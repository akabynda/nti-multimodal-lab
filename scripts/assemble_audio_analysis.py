"""Analyze new utterance boundaries while retaining unchanged acoustic windows."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from nti_lab.audio import SR, acoustic_windows, diarize, merge_intervals, question_answer_candidates, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("../1_30"))
    parser.add_argument("--asr", type=Path, default=Path("outputs/audio-large-v3"))
    parser.add_argument("--previous", type=Path, default=Path("outputs/audio"))
    parser.add_argument("--output", type=Path, default=Path("outputs/audio-large-v3/analysis"))
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    from faster_whisper.vad import get_speech_timestamps, VadOptions
    for path in sorted(args.asr.glob("*.segments.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        with path.open("rb") as handle:
            data["source_asr_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
        rid = data["recording_id"]
        target = args.output / path.name
        if target.exists():
            continue
        decoded = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(args.input / data["source_file"]),
                                  "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
                                 check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        audio = np.frombuffer(decoded.stdout, dtype=np.float32).copy()
        assert abs(len(audio) / SR - data["duration_s"]) < 0.01
        old_path = args.previous / path.name
        old = json.loads(old_path.read_text(encoding="utf-8")) if old_path.exists() else {}
        old_features = args.previous / f"{rid}.features.csv"
        feature_path = args.output / old_features.name
        reuse = old.get("source_file") == data["source_file"] and abs(old.get("duration_s", -1) - data["duration_s"]) < 0.01 and old_features.exists() and old.get("speech_intervals")
        if reuse:
            speech = old["speech_intervals"]
            shutil.copy2(old_features, feature_path)
            data["acoustic_feature_provenance"] = {"method": "Reused unchanged 5s acoustic windows and VAD from the same original recording and decoded duration; ASR text is not an input to these features.", "source": str(old_features.resolve())}
        else:
            vad = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=300, speech_pad_ms=0), sampling_rate=SR)
            speech = merge_intervals([(v["start"] / SR, v["end"] / SR) for v in vad])
            features = acoustic_windows(audio, speech)
            with feature_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(features[0]))
                writer.writeheader()
                writer.writerows(features)
            data["acoustic_feature_provenance"] = {"method": "New original-audio acoustic and VAD extraction"}
        data["speech_intervals"] = speech
        data["speech_duration_s"] = round(sum(b - a for a, b in speech), 3)
        data["diarization"] = diarize(data["segments"], audio, Path("data/private/annotations/speakers") / f"{rid}.json")
        data["qa_candidates"] = question_answer_candidates(data["segments"], speech)
        data["warnings"] = ["Automatic ASR, provisional speaker clusters and question/answer pairs require review.",
                            "Acoustic features describe the recording and do not establish deception, stress or psychological traits."]
        write_json(target, data)
        write_json(args.output / f"{rid}.summary.json", {
            "recording_id": rid, "participant_id": data["participant_id"], "duration_s": data["duration_s"],
            "speech_duration_s": data["speech_duration_s"], "segment_count": len(data["segments"]),
            "qa_candidate_count": len(data["qa_candidates"]), "feature_window_s": 5, "feature_step_s": 1,
            "diarization": data["diarization"], "warnings": data["warnings"],
            "acoustic_feature_provenance": data["acoustic_feature_provenance"],
        })
        print(f"DONE {rid} reused_acoustics={bool(reuse)}", flush=True)


if __name__ == "__main__":
    main()
