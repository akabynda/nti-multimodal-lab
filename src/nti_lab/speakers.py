"""Optional neural two-speaker diarization after ASR/acoustic extraction."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import json
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .audio import SR, finite, write_json


def make_extractor(model: Path, threads=2):
    import sherpa_onnx
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model), num_threads=threads, provider="cpu", debug=False,
    )
    if not config.validate():
        raise ValueError(f"Invalid speaker model: {model}")
    return sherpa_onnx.SpeakerEmbeddingExtractor(config)


def refine_speakers(segments, audio, extractor, override_path=None):
    eligible, vectors = [], []
    for index, segment in enumerate(segments):
        segment["speaker_id"] = "unknown"
        segment["speaker_role"] = "unconfirmed"
        segment["speaker_label_status"] = "provisional_neural_embedding"
        start, end = segment["start"], segment["end"]
        if end - start < 0.7:
            continue
        # Long utterances can contain multiple speakers; cap the sample to a centre crop.
        length = min(6, end - start)
        middle = (start + end) / 2
        y = audio[int((middle - length / 2) * SR):int((middle + length / 2) * SR)]
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=SR, waveform=np.ascontiguousarray(y))
        stream.input_finished()
        if not extractor.is_ready(stream):
            continue
        vector = np.array(extractor.compute(stream), dtype=np.float64)
        norm = np.linalg.norm(vector)
        if norm == 0 or not np.isfinite(vector).all():
            continue
        eligible.append(index)
        vectors.append(vector / norm)
    if len(vectors) < 4:
        return {"method": "WeSpeaker ResNet34", "status": "insufficient_utterances"}
    values = np.array(vectors)
    labels = KMeans(n_clusters=2, random_state=42, n_init=30).fit_predict(values)
    labels = labels ^ labels[0]
    centres = np.array([values[labels == label].mean(axis=0) for label in (0, 1)])
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    similarity = values @ centres.T
    for row, (index, label) in enumerate(zip(eligible, labels)):
        segment = segments[index]
        segment["speaker_id"] = "A" if label == 0 else "B"
        segment["speaker_similarity_margin"] = finite(similarity[row, label] - similarity[row, 1 - label])
        segment["speaker_short_utterance"] = segment["end"] - segment["start"] < 1.5
    # Role candidates rely on verbal question patterns; confirmed roles still require annotation.
    ratios = {}
    counts = {}
    for speaker in ("A", "B"):
        own = [s for s in segments if s["speaker_id"] == speaker]
        counts[speaker] = len(own)
        ratios[speaker] = sum("?" in s["text"] for s in own) / max(len(own), 1)
    interviewer = max(ratios, key=ratios.get)
    balanced = min(counts.values()) >= max(10, 0.1 * sum(counts.values()))
    role_hint_supported = balanced and abs(ratios["A"] - ratios["B"]) >= 0.15
    for segment in segments:
        speaker = segment["speaker_id"]
        segment["speaker_role_candidate"] = (
            "interviewer" if speaker == interviewer else "respondent"
        ) if speaker in ("A", "B") and role_hint_supported else "unconfirmed"
    if override_path and override_path.exists():
        overrides = json.loads(override_path.read_text(encoding="utf-8"))
        for annotation in overrides.get("intervals", []):
            for segment in segments:
                if annotation["start"] <= (segment["start"] + segment["end"]) / 2 < annotation["end"]:
                    segment["speaker_id"] = annotation["speaker_id"]
                    segment["speaker_role"] = annotation.get("speaker_role", "unconfirmed")
                    segment["speaker_label_status"] = "manual"
    return {
        "method": "WeSpeaker ResNet34 unit-normalized speaker embeddings + KMeans, 2 speakers",
        "algorithm_id": "wespeaker_cosine_kmeans_v2",
        "model": "wespeaker_en_voxceleb_resnet34.onnx",
        "model_sha256": "5ef208a9da1453335308a6b6f4e6dfbd7e183a38b604de0a57664f45d257fe94",
        "embedding_dim": int(values.shape[1]), "silhouette_score": finite(silhouette_score(values, labels, metric="cosine")),
        "question_ratio_by_speaker": ratios,
        "utterance_count_by_speaker": counts,
        "cluster_balance_check": "passed" if balanced else "review_required_unbalanced_clusters",
        "role_candidate_basis": "question-mark frequency; not a confirmed identity or role",
        "roles_confirmed": all(s["speaker_role"] != "unconfirmed" for s in segments),
        "warning": "English-trained model applied to Russian speech without annotated diarization evaluation; overlap and within-utterance turns are not resolved; short utterances are uncertain.",
    }


def process_recording(path, args):
    start = time.time()
    summary = args.output / f"{path.stem}.summary.json"
    while not summary.exists():
        if not args.wait_for_acoustics or time.time() - start > 2400:
            raise RuntimeError(f"Acoustics not ready: {path.stem}")
        time.sleep(2)
    transcript = args.output / f"{path.stem}.segments.json"
    data = json.loads(transcript.read_text(encoding="utf-8"))
    if (data.get("diarization", {}).get("model") == args.model.name and
            data.get("diarization", {}).get("algorithm_id") == "wespeaker_cosine_kmeans_v2"):
        print(f"SKIP_SPEAKERS {path.stem}", flush=True)
        return
    extractor = make_extractor(args.model)
    decoded = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path), "-vn",
         "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    audio = np.frombuffer(decoded.stdout, dtype=np.float32)
    meta = refine_speakers(data["segments"], audio, extractor, args.overrides / f"{path.stem}.json")
    data["diarization"] = meta
    write_json(transcript, data)
    summary_data = json.loads(summary.read_text(encoding="utf-8"))
    summary_data["diarization"] = meta
    write_json(summary, summary_data)
    print(f"SPEAKERS_READY {path.stem} silhouette={meta.get('silhouette_score')} elapsed={time.time()-start:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("../1_30"))
    parser.add_argument("--output", type=Path, default=Path("outputs/audio"))
    parser.add_argument("--model", type=Path, default=Path("models/speaker/wespeaker_en_voxceleb_resnet34.onnx"))
    parser.add_argument("--overrides", type=Path, default=Path("data/annotations/speakers"))
    parser.add_argument("--wait-for-acoustics", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    paths = [args.input] if args.input.is_file() else sorted(args.input.glob("*.MTS"))
    paths = paths[:args.limit]
    if args.workers == 1:
        for path in paths:
            process_recording(path, args)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(process_recording, paths, repeat(args)))


if __name__ == "__main__":
    main()
