"""Local Russian ASR, transparent acoustic features, and provisional diarization.

Times are seconds from the first decoded audio sample of the supplied recording.
Speaker clustering and question/answer pairs require expert confirmation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.fft import dct

_DLL_HANDLES = []
_SPEAKER_EXTRACTOR = None
SR = 16000


def setup_cuda(root: Path) -> None:
    """Keep Windows DLL directories local to this process."""
    if os.name != "nt":
        return
    candidates = [
        root / "vendor/nvidia/cudnn/bin",
        Path(os.environ.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1")) / "bin",
    ]
    for directory in candidates:
        if directory.is_dir():
            _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ["PATH"]


def finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def write_json(path: Path, data: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def timecode(seconds):
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def words_and_syllables(text):
    words = re.findall(r"[А-Яа-яЁёA-Za-z]+", text)
    # Orthographic vowel count is explicitly an estimate, not phoneme alignment.
    syllables = len(re.findall(r"[аеёиоуыэюяaeiouy]", text.lower()))
    return len(words), syllables


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([float(start), float(end)])
        else:
            merged[-1][1] = max(merged[-1][1], float(end))
    return merged


def interval_duration(intervals, start, end):
    return sum(max(0, min(end, b) - max(start, a)) for a, b in intervals)


def split_utterances(chunks):
    """Split decoder chunks at sentence boundaries or >=650-ms word gaps."""
    utterances = []
    for chunk in chunks:
        words = chunk.get("words") or []
        if not words:
            utterances.append(dict(chunk, id=len(utterances)))
            continue
        current = []
        for i, word in enumerate(words):
            current.append(word)
            gap = words[i + 1]["start"] - word["end"] if i + 1 < len(words) else float("inf")
            sentence_end = bool(re.search(r"[.!?…][»\"']?$", word["word"].strip()))
            if sentence_end or gap >= 0.65 or word["end"] - current[0]["start"] >= 15:
                text = "".join(w["word"] for w in current).strip()
                count, syllables = words_and_syllables(text)
                duration = max(current[-1]["end"] - current[0]["start"], 0.01)
                utterances.append({
                    "id": len(utterances), "source_chunk_id": chunk["id"],
                    "start": current[0]["start"], "end": current[-1]["end"],
                    "text": text, "words": current, "avg_logprob": chunk.get("avg_logprob"),
                    "no_speech_prob": chunk.get("no_speech_prob"),
                    "words_per_minute": round(count * 60 / duration, 3),
                    "syllables_per_second_est": round(syllables / duration, 3),
                })
                current = []
    return utterances


def bound_timestamps(segments, duration):
    """Limit model-estimated edge times to available audio, retaining originals."""
    for segment in segments:
        for item in [segment, *segment.get("words", [])]:
            start, end = item["start"], item["end"]
            if start < 0 or end > duration:
                item["raw_model_start"] = start
                item["raw_model_end"] = end
                item["timing_clipped_to_recording"] = True
                item["start"] = min(max(0, start), duration)
                item["end"] = min(max(item["start"], end), duration)
        count, syllables = words_and_syllables(segment["text"])
        length = max(segment["end"] - segment["start"], 0.01)
        segment["words_per_minute"] = round(60 * count / length, 3)
        segment["syllables_per_second_est"] = round(syllables / length, 3)


def acoustic_windows(audio: np.ndarray, speech_intervals, step=1.0, window=5.0):
    """Praat F0/formants plus relative digital energy and spectral descriptors."""
    import parselmouth

    sound = parselmouth.Sound(audio, sampling_frequency=SR)
    pitch = sound.to_pitch_ac(time_step=0.02, pitch_floor=60, pitch_ceiling=500)
    pitch_t = pitch.xs()
    pitch_f = pitch.selected_array["frequency"]
    formants = sound.to_formant_burg(time_step=0.05, max_number_of_formants=5, maximum_formant=5500)
    # Retain measured formant values only when the nearest pitch frame is voiced.
    ft = formants.xs()
    nearest = np.clip(np.searchsorted(pitch_t, ft), 0, len(pitch_f) - 1)
    voiced_formants = pitch_f[nearest] > 0
    fs = np.array([[formants.get_value_at_time(i, t) for i in (1, 2, 3)] for t in ft])
    fs[~voiced_formants] = np.nan
    rows = []
    for start in np.arange(0, len(audio) / SR, step):
        end = min(start + window, len(audio) / SR)
        if end - start < 0.5:
            break
        y = audio[int(start * SR):int(end * SR)]
        rms = np.sqrt(np.mean(y.astype(np.float64) ** 2))
        freqs, psd = signal.welch(y, SR, nperseg=min(len(y), 2048))
        psd = np.maximum(psd, 1e-20)
        band = (freqs >= 100) & (freqs <= 4000)
        centroid = np.sum(freqs * psd) / np.sum(psd)
        tilt = np.polyfit(np.log2(freqs[band]), 10 * np.log10(psd[band]), 1)[0]
        pf = pitch_f[(pitch_t >= start) & (pitch_t < end)]
        voiced = pf[pf > 0]
        fmask = (ft >= start) & (ft < end)
        formant_values = fs[fmask]
        medians = [
            finite(np.median(col[np.isfinite(col)])) if np.isfinite(col).any() else None
            for col in formant_values.T
        ]
        rows.append({
            "start_s": round(float(start), 3), "end_s": round(float(end), 3),
            "time_s": round(float((start + end) / 2), 3),
            "rms_dbfs": finite(20 * np.log10(max(rms, 1e-10))),
            "rms": finite(rms),
            "pitch_median_hz": finite(np.median(voiced)) if len(voiced) else None,
            "pitch_iqr_hz": finite(np.subtract(*np.percentile(voiced, [75, 25]))) if len(voiced) else None,
            "voiced_fraction": finite(len(voiced) / max(len(pf), 1)),
            "speech_fraction": finite(interval_duration(speech_intervals, start, end) / (end - start)),
            "spectral_centroid_hz": finite(centroid),
            "spectral_tilt_db_per_octave": finite(tilt),
            "f1_median_hz": medians[0], "f2_median_hz": medians[1], "f3_median_hz": medians[2],
            "clipping_fraction": finite(np.mean(np.abs(y) >= 0.999)),
        })
    return rows


def speaker_vector(y):
    """Simple MFCC statistics for explicitly provisional two-speaker clustering."""
    if len(y) < SR * 0.4:
        return None
    _, _, spectrum = signal.stft(y, SR, nperseg=400, noverlap=240, boundary=None)
    power = np.abs(spectrum) ** 2
    # 32 triangular mel filters over 80--7600 Hz.
    mel_edges = np.linspace(2595 * np.log10(1 + 80 / 700), 2595 * np.log10(1 + 7600 / 700), 34)
    hz_edges = 700 * (10 ** (mel_edges / 2595) - 1)
    frequency = np.fft.rfftfreq(400, 1 / SR)
    filters = np.array([
        np.maximum(0, np.minimum((frequency - a) / (b - a), (c - frequency) / (c - b)))
        for a, b, c in zip(hz_edges[:-2], hz_edges[1:-1], hz_edges[2:])
    ])
    mfcc = dct(np.log(np.maximum(filters @ power, 1e-12)), axis=0, norm="ortho")[1:14]
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])


def diarize(segments, audio, override_path: Path | None):
    """Labels speakers A/B; roles are only set by explicit user annotation."""
    global _SPEAKER_EXTRACTOR
    neural_model = Path(__file__).resolve().parents[2] / "models/speaker/wespeaker_en_voxceleb_resnet34.onnx"
    if neural_model.exists():
        from .speakers import make_extractor, refine_speakers
        if _SPEAKER_EXTRACTOR is None:
            _SPEAKER_EXTRACTOR = make_extractor(neural_model)
        return refine_speakers(segments, audio, _SPEAKER_EXTRACTOR, override_path)
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    eligible, vectors = [], []
    for i, segment in enumerate(segments):
        y = audio[int(segment["start"] * SR):int(segment["end"] * SR)]
        vector = speaker_vector(y)
        if vector is not None:
            eligible.append(i)
            vectors.append(vector)
    score = None
    if len(vectors) >= 4:
        values = StandardScaler().fit_transform(vectors)
        model = KMeans(n_clusters=2, random_state=42, n_init=20).fit(values)
        labels = model.labels_
        # Make A consistently the first clustered speaker, without inferring their role.
        labels = labels ^ labels[0]
        if 1 < len(np.unique(labels)) < len(labels):
            score = finite(silhouette_score(values, labels))
        for index, label in zip(eligible, labels):
            segments[index]["speaker_id"] = "A" if label == 0 else "B"
    for segment in segments:
        segment.setdefault("speaker_id", "unknown")
        segment["speaker_role"] = "unconfirmed"
        segment["speaker_label_status"] = "provisional_acoustic_cluster"
    if override_path and override_path.exists():
        overrides = json.loads(override_path.read_text(encoding="utf-8"))
        for annotation in overrides.get("intervals", []):
            for segment in segments:
                midpoint = (segment["start"] + segment["end"]) / 2
                if annotation["start"] <= midpoint < annotation["end"]:
                    segment["speaker_id"] = annotation["speaker_id"]
                    segment["speaker_role"] = annotation.get("speaker_role", "unconfirmed")
                    segment["speaker_label_status"] = "manual"
    return {"method": "MFCC statistics + 2-means; provisional, not neural speaker verification",
            "silhouette_score": score, "roles_confirmed": all(s["speaker_role"] != "unconfirmed" for s in segments),
            "warning": "Clusters may split one speaker or merge speakers; overlap is not separated. Confirm roles manually."}


def question_answer_candidates(segments, speech):
    pairs = []
    for i, question in enumerate(segments[:-1]):
        if "?" not in question["text"]:
            continue
        answer = segments[i + 1]
        if answer["start"] < question["end"]:
            continue
        words, syllables = words_and_syllables(answer["text"])
        duration = max(answer["end"] - answer["start"], 1e-6)
        voiced_duration = interval_duration(speech, answer["start"], answer["end"])
        pairs.append({
            "question_segment_id": question["id"], "answer_segment_id": answer["id"],
            "question_start_s": question["start"], "question_end_s": question["end"],
            "answer_start_s": answer["start"], "answer_end_s": answer["end"],
            "latency_s": round(answer["start"] - question["end"], 3),
            "answer_duration_s": round(duration, 3),
            "within_answer_nonspeech_s": round(max(0, duration - voiced_duration), 3),
            "words_per_minute": round(60 * words / duration, 3),
            "syllables_per_second_est": round(syllables / duration, 3),
            "status": "candidate_requires_expert_confirmation",
            "question": question["text"], "answer": answer["text"],
        })
    return pairs


def transcribe_file(model, path: Path, out: Path, args):
    from faster_whisper import BatchedInferencePipeline
    from faster_whisper.vad import get_speech_timestamps, VadOptions
    import imageio_ffmpeg

    started = time.time()
    recording_id = path.stem
    participant = recording_id.split("_", 1)[-1]
    target = out / f"{recording_id}.segments.json"
    if args.features_only:
        if args.skip_existing and (out / f"{recording_id}.summary.json").exists():
            print(f"SKIP_COMPLETE {recording_id}", flush=True)
            return
        deadline = time.time() + 1800
        while not target.exists() or not (out / f"{recording_id}.txt").exists():
            if not args.wait_for_asr or time.time() > deadline:
                raise RuntimeError(f"ASR output is not ready for {recording_id}")
            time.sleep(2)
    decoded = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path),
         "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    audio = np.frombuffer(decoded.stdout, dtype=np.float32).copy()
    duration = len(audio) / SR
    print(f"START {recording_id} duration={duration:.2f}s", flush=True)
    if (args.skip_existing or args.features_only) and target.exists():
        result = json.loads(target.read_text(encoding="utf-8"))
        segments = result["segments"]
    else:
        pipeline = BatchedInferencePipeline(model=model)
        generated, info = pipeline.transcribe(
            audio, language="ru", batch_size=args.batch_size,
            beam_size=args.beam_size, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 200},
            word_timestamps=True, condition_on_previous_text=False,
        )
        segments = []
        for segment in generated:
            words, syllables = words_and_syllables(segment.text)
            length = max(segment.end - segment.start, 0.01)
            segments.append({
                "id": int(segment.id), "start": float(segment.start), "end": float(segment.end),
                "text": segment.text.strip(), "avg_logprob": finite(segment.avg_logprob),
                "no_speech_prob": finite(segment.no_speech_prob),
                "words_per_minute": round(words / length * 60, 3),
                "syllables_per_second_est": round(syllables / length, 3),
                "words": [{"start": float(w.start), "end": float(w.end), "word": w.word,
                           "probability": finite(w.probability)} for w in (segment.words or [])],
            })
            if len(segments) % 50 == 0:
                print(f"ASR_PROGRESS {recording_id} t={segment.end:.1f}/{duration:.1f}s segments={len(segments)}", flush=True)
        result = {
            "schema_version": 1, "recording_id": recording_id, "participant_id": participant,
            "source_file": path.name, "duration_s": duration, "language": info.language,
            "model": Path(args.model).name, "model_path": str(Path(args.model).resolve()),
            "model_sha256": args.model_sha256, "device": args.device,
            "compute_type": args.compute_type, "timebase": "audio_seconds_from_first_decoded_sample",
            "decoding": {"backend": "faster-whisper BatchedInferencePipeline", "beam_size": args.beam_size,
                         "batch_size": args.batch_size, "word_timestamps": True,
                         "condition_on_previous_text": False, "initial_prompt": None,
                         "vad_filter": True, "min_silence_duration_ms": 300, "speech_pad_ms": 200},
            "asr_status": "automatic_unreviewed", "segments": segments,
        }
        write_json(out / f"{recording_id}.raw.json", result)
        write_json(target, result)
        print(f"ASR_READY {target} segments={len(segments)} elapsed={time.time()-started:.1f}s", flush=True)
    if not result.get("utterance_segmentation"):
        segments = split_utterances(segments)
        result["segments"] = segments
        result["utterance_segmentation"] = "word_timestamps_sentence_punctuation_or_650ms_gap_or_15s"
        # Previous coarse-chunk derived labels must be recomputed after splitting.
        result.pop("qa_candidates", None)
        result.pop("diarization", None)
    bound_timestamps(segments, duration)
    write_json(target, result)
    # Save transcript before acoustic extraction so reference matching can begin.
    (out / f"{recording_id}.txt").write_text(
        "\n".join(f"[{timecode(s['start'])} --> {timecode(s['end'])}] {s['text']}" for s in segments),
        encoding="utf-8",
    )
    if args.asr_only:
        print(f"TRANSCRIPT_READY {recording_id} utterances={len(segments)}", flush=True)
        return
    vad = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=300, speech_pad_ms=0), sampling_rate=SR)
    speech = merge_intervals([(v["start"] / SR, v["end"] / SR) for v in vad])
    override = args.overrides / f"{recording_id}.json" if args.overrides else None
    result["diarization"] = diarize(segments, audio, override)
    result["speech_intervals"] = speech
    result["qa_candidates"] = question_answer_candidates(segments, speech)
    result["speech_duration_s"] = round(sum(b - a for a, b in speech), 3)
    result["warnings"] = [
        "ASR text, timing, speaker clustering and question/answer candidates are automatic and require review.",
        "Orthographic vowel count estimates syllables; it is not a phonetic transcription.",
        "F0 and formants include all recorded speakers; dBFS is relative digital level, not calibrated sound pressure.",
        "Acoustic features do not establish stress or deception; no lie probability is produced.",
    ]
    write_json(target, result)
    features = acoustic_windows(audio, speech)
    feature_path = out / f"{recording_id}.features.csv"
    with feature_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(features[0]))
        writer.writeheader()
        writer.writerows(features)
    write_json(out / f"{recording_id}.summary.json", {
        "recording_id": recording_id, "participant_id": participant, "duration_s": duration,
        "speech_duration_s": result["speech_duration_s"], "segment_count": len(segments),
        "qa_candidate_count": len(result["qa_candidates"]), "feature_window_s": 5,
        "feature_step_s": 1, "elapsed_s": round(time.time() - started, 2),
        "diarization": result["diarization"], "warnings": result["warnings"],
    })
    print(f"DONE {recording_id} elapsed={time.time()-started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("../1_30"))
    parser.add_argument("--output", type=Path, default=Path("outputs/audio"))
    parser.add_argument("--model", default="models/faster-whisper-large-v3")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overrides", type=Path, default=Path("data/private/annotations/speakers"))
    parser.add_argument("--limit", type=int)
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--asr-only", action="store_true")
    stages.add_argument("--features-only", action="store_true")
    parser.add_argument("--wait-for-asr", action="store_true")
    args = parser.parse_args()
    import hashlib
    model_file = Path(args.model) / "model.bin"
    args.model_sha256 = None
    if model_file.is_file():
        with model_file.open("rb") as handle:
            args.model_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    sys.stdout.reconfigure(encoding="utf-8")
    setup_cuda(Path(__file__).resolve().parents[2])
    from faster_whisper import WhisperModel
    args.output.mkdir(parents=True, exist_ok=True)
    model = None if args.features_only else WhisperModel(
        args.model, device=args.device, compute_type=args.compute_type, cpu_threads=6
    )
    paths = [args.input] if args.input.is_file() else sorted(args.input.glob("*.MTS"))
    for path in paths[:args.limit]:
        transcribe_file(model, path, args.output, args)


if __name__ == "__main__":
    main()
