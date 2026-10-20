"""Repeat requested original-audio intervals without prompts or reference text."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from nti_lab.audio import SR, setup_cuda, write_json
from verify_content_audio import decode_pass


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("../1_30"))
    parser.add_argument("--output", type=Path, default=Path("outputs/audio-large-v3"))
    parser.add_argument("--requests", type=Path, default=Path("outputs/audio-large-v3/review_requests.json"))
    parser.add_argument("--model", type=Path, default=Path("models/faster-whisper-large-v3"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    setup_cuda(Path(__file__).resolve().parents[1])
    from faster_whisper import WhisperModel
    requests = json.loads(args.requests.read_text(encoding="utf-8"))
    destination = args.output / "rechecks"
    destination.mkdir(exist_ok=True)
    model = WhisperModel(str(args.model), device=args.device, compute_type=args.compute_type, cpu_threads=args.cpu_threads)
    model_hash = digest(args.model / "model.bin")
    for row in requests["clips"]:
        target = destination / f"{row['clip_id']}.json"
        if target.exists():
            saved = json.loads(target.read_text(encoding="utf-8"))
            if saved.get("request") == row and saved.get("model_sha256") == model_hash:
                print(f"SKIP {row['clip_id']}", flush=True)
                continue
            raise ValueError(f"Existing clip has incompatible provenance: {target}")
        source = args.input / f"{row['recording_id']}.MTS"
        decoded = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-ss", str(row["start_s"]),
                                  "-i", str(source), "-t", str(row["end_s"] - row["start_s"]), "-vn", "-ac", "1",
                                  "-ar", str(SR), "-f", "f32le", "pipe:1"],
                                 check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        audio = np.frombuffer(decoded.stdout, dtype=np.float32).copy()
        passes = [decode_pass(model, audio, row["start_s"], beam) for beam in [10, 1]]
        result = {"request": row, "source_file": str(source.resolve()),
                  "decoded_float32_audio_sha256": hashlib.sha256(decoded.stdout).hexdigest(),
                  "source_asr_sha256": digest(args.output / f"{row['recording_id']}.segments.json"),
                  "model": args.model.name, "model_sha256": model_hash, "device": args.device, "compute_type": args.compute_type,
                  "method": "Original recording, cropped context, independent non-batched decodes with beam sizes 10 and 1, temperature 0, no VAD, no initial prompt or previous-text conditioning.",
                  "reference_stories_used": False, "audio_verified": False, "manual_listening_performed": False,
                  "passes": passes}
        write_json(target, result)
        print(f"DONE {row['clip_id']} {len(audio)/SR:.1f}s {passes[0]['text']}", flush=True)


if __name__ == "__main__":
    main()
