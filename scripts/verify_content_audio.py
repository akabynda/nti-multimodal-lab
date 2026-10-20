"""Repeat selected content-candidate ASR on original audio; never certify listening.

Example:
python scripts/verify_content_audio.py --participants Participant01
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf

from nti_lab.audio import SR, setup_cuda, write_json


def prepare_clips(candidates, duration_by_recording):
    clips = []
    for row in sorted(candidates, key=lambda r: (r["recording_id"], r["context_start_s"])):
        start = max(0, row["context_start_s"] - 3)
        end = min(duration_by_recording[row["recording_id"]], row["context_end_s"] + 5)
        if clips and clips[-1]["recording_id"] == row["recording_id"] and start <= clips[-1]["end_s"]:
            clips[-1]["end_s"] = max(clips[-1]["end_s"], end)
            clips[-1]["comparison_ids"].append(row["comparison_id"])
        else:
            clips.append({"recording_id": row["recording_id"], "start_s": start, "end_s": end,
                          "comparison_ids": [row["comparison_id"]]})
    for i, clip in enumerate(clips, 1):
        clip["clip_id"] = f"recheck_{i:02}"
    return clips


def decode_pass(model, audio, offset, beam_size):
    generated, info = model.transcribe(
        audio, language="ru", beam_size=beam_size, temperature=0.0,
        condition_on_previous_text=False, vad_filter=False, word_timestamps=True,
        initial_prompt=None, no_speech_threshold=0.6,
    )
    segments = []
    for segment in generated:
        segments.append({
            "start_s": round(offset + segment.start, 3),
            "end_s": round(offset + segment.end, 3),
            "text": segment.text.strip(), "avg_logprob": segment.avg_logprob,
            "no_speech_prob": segment.no_speech_prob,
            "words": [{"start_s": round(offset + w.start, 3), "end_s": round(offset + w.end, 3),
                       "text": w.word.strip(), "probability": w.probability}
                      for w in (segment.words or [])],
        })
    return {"beam_size": beam_size, "text": " ".join(s["text"] for s in segments), "segments": segments,
            "language": info.language}


def apply_assessments(report, path):
    assessment = json.loads(path.read_text(encoding="utf-8"))
    if assessment["source_comparisons_sha256"] != report["source_comparisons_sha256"]:
        raise ValueError("Assessments refer to a different content-comparison snapshot")
    by_id = {row["comparison_id"]: row for row in assessment["candidate_assessments"]}
    if set(by_id) != {row["comparison_id"] for row in report["candidate_checks"]}:
        raise ValueError("Assessment IDs do not match the decoded candidates")
    for check in report["candidate_checks"]:
        check.update(by_id[check["comparison_id"]])
        check["audio_verified"] = False
        check["manual_listening_performed"] = False
        check["review_required"] = True
    report["assessment_method"] = assessment["assessment_method"]
    report["summary"] = assessment["summary"]
    report["clip_disagreement_notes"] = assessment["clip_disagreement_notes"]
    report["primary_model_sha256"] = assessment.get("primary_model_sha256", report.get("primary_model_sha256"))
    report["status"] = "repeat_asr_and_assistant_textual_comparison_complete_not_human_audio_verified"
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--participants", nargs="+")
    parser.add_argument("--model", default="models/faster-whisper-large-v3")
    parser.add_argument("--comparisons", type=Path, default=Path("outputs/research/content_comparison.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/research/audio_recheck.json"))
    parser.add_argument("--input", type=Path, default=Path("../1_30"))
    parser.add_argument("--assessment-file", type=Path)
    parser.add_argument("--assessment-only", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.assessment_only:
        if not args.assessment_file:
            parser.error("--assessment-only requires --assessment-file")
        report = json.loads(args.output.read_text(encoding="utf-8"))
        write_json(args.output, apply_assessments(report, args.assessment_file))
        print("ASSESSMENTS_APPLIED; audio_verified remains false", flush=True)
        return
    if not args.participants:
        parser.error("--participants is required for audio decoding")
    root = Path(__file__).resolve().parents[1]
    setup_cuda(root)
    from faster_whisper import WhisperModel

    comparison_bytes = args.comparisons.read_bytes()
    comparisons = json.loads(comparison_bytes)
    candidates = [row for row in comparisons["comparisons"]
                  if row["participant_id"] in args.participants and row["relation"] == "contradiction_candidate"]
    transcripts = {r["recording_id"]: json.loads((root / f"outputs/audio/{r['recording_id']}.segments.json").read_text(encoding="utf-8"))
                   for r in candidates}
    clips = prepare_clips(candidates, {key: value["duration_s"] for key, value in transcripts.items()})
    clip_dir = args.output.parent / "audio_recheck_clips"
    clip_dir.mkdir(exist_ok=True, parents=True)
    report = {
        "schema_version": 1,
        "method": "Repeat automatic transcription of clips decoded directly from original MTS audio",
        "primary_model": "Systran/faster-whisper-large-v3 (full model, not turbo)",
        "original_model": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "settings": {"device": "cuda", "compute_type": "int8_float16", "language": "ru",
                     "beam_sizes": [10, 1], "temperature": 0, "vad_filter": False,
                     "condition_on_previous_text": False, "initial_prompt": None,
                     "sample_rate_hz": SR},
        "independence_limit": "Both repeated decodes use one full Whisper checkpoint; original and repeat checkpoints belong to the Whisper family. Agreement is not independent human verification.",
        "manual_listening_performed": False, "audio_verified": False,
        "source_comparisons_sha256": hashlib.sha256(comparison_bytes).hexdigest(),
        "source_comparisons_modified": False,
        "candidate_count": len(candidates), "unique_clip_count": len(clips),
        "clips": [], "candidate_checks": [],
    }
    model = WhisperModel(args.model, device="cuda", compute_type="int8_float16", cpu_threads=6)
    for clip in clips:
        start = time.time()
        wav_path = clip_dir / f"{clip['clip_id']}_{clip['recording_id']}.wav"
        source = args.input / f"{clip['recording_id']}.MTS"
        subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-ss", str(clip["start_s"]),
            "-i", str(source), "-t", str(clip["end_s"] - clip["start_s"]),
            "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", "-y", str(wav_path),
        ], check=True)
        audio, sr = sf.read(wav_path, dtype="float32")
        assert sr == SR
        clip["local_audio_file"] = str(wav_path.resolve())
        clip["audio_sha256"] = hashlib.sha256(wav_path.read_bytes()).hexdigest()
        clip["source_file"] = source.name
        clip["passes"] = [decode_pass(model, audio, clip["start_s"], beam) for beam in (10, 1)]
        report["clips"].append(clip)
        write_json(args.output, report)
        print(f"RECHECK_READY {clip['clip_id']} ids={','.join(clip['comparison_ids'])} elapsed={time.time()-start:.1f}s", flush=True)
    for original in candidates:
        clip = next(c for c in report["clips"] if original["comparison_id"] in c["comparison_ids"])
        evidence = [s for s in transcripts[original["recording_id"]]["segments"]
                    if s["end"] >= original["context_start_s"] and s["start"] <= original["context_end_s"]]
        report["candidate_checks"].append({
            "comparison_id": original["comparison_id"], "participant_id": original["participant_id"],
            "clip_id": clip["clip_id"], "original_spoken_excerpt": original["spoken_excerpt"],
            "original_source_excerpt": original["source_excerpt"],
            "candidate_start_s": original["start_s"], "candidate_end_s": original["end_s"],
            "speaker_evidence_method": "Previously computed neural clustering plus textual dialogue position; not human-confirmed speaker identity",
            "speaker_evidence": [{"start_s": s["start"], "end_s": s["end"], "text": s["text"],
                                 "speaker_id": s.get("speaker_id"), "role_candidate": s.get("speaker_role_candidate"),
                                 "role_confirmed": s.get("speaker_role", "unconfirmed")}
                                for s in evidence],
            "audio_verified": False, "manual_listening_performed": False, "review_required": True,
            "assessment_status": "repeat_asr_ready_for_textual_comparison",
            "video_truth_label": None,
        })
    report["status"] = "repeat_decoding_complete_textual_assessment_pending"
    if args.assessment_file:
        apply_assessments(report, args.assessment_file)
    write_json(args.output, report)
    print(f"DONE candidates={len(candidates)} unique_clips={len(clips)}", flush=True)


if __name__ == "__main__":
    main()
