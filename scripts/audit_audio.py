"""Validate all available audio artefacts without rerunning ML inference."""
import csv
import json
import argparse
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=root / "outputs/audio")
parser.add_argument("--expected-recordings", type=int)
parser.add_argument("--asr-only", action="store_true")
args = parser.parse_args()
output = args.output
rows = []
for path in sorted(output.glob("*.segments.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = data["segments"]
    duration = data["duration_s"]
    assert segments, path
    assert len({s["id"] for s in segments}) == len(segments), path
    previous = -1
    for segment in segments:
        assert 0 <= segment["start"] <= segment["end"] <= duration + 0.1, (path, segment["id"])
        assert segment["start"] >= previous - 0.02, (path, segment["id"])
        previous = segment["start"]
        for word in segment.get("words", []):
            assert segment["start"] - 0.02 <= word["start"] <= word["end"] <= segment["end"] + 0.02, (path, segment["id"])
    feature_path = output / f"{data['recording_id']}.features.csv"
    features = list(csv.DictReader(feature_path.open(encoding="utf-8"))) if feature_path.exists() else []
    if not args.asr_only:
        assert features, f"Acoustic features missing: {path}"
        assert data.get("diarization"), f"Diarization metadata missing: {path}"
    assert not any(s.get("speaker_role", "unconfirmed") != "unconfirmed" and
                   s.get("speaker_label_status") != "manual" for s in segments), path
    for row in features:
        assert 0 <= float(row["start_s"]) < float(row["end_s"]) <= duration + 0.01, path
        assert 0 <= float(row["speech_fraction"]) <= 1.00001, path
        assert 0 <= float(row["voiced_fraction"]) <= 1, path
    rows.append({
        "recording_id": data["recording_id"],
        "duration_s": duration, "utterances": len(segments),
        "words": sum(len(s.get("words", [])) for s in segments),
        "first_utterance_s": segments[0]["start"], "last_utterance_s": segments[-1]["end"],
        "feature_windows": len(features), "qa_candidates": len(data.get("qa_candidates", [])),
        "diarization_method": data.get("diarization", {}).get("method"),
        "cluster_balance_check": data.get("diarization", {}).get("cluster_balance_check"),
        "unknown_speaker_utterances": sum(s.get("speaker_id") == "unknown" for s in segments),
    })
assert rows, "No ASR recordings found"
if args.expected_recordings is not None:
    assert len(rows) == args.expected_recordings, f"Expected {args.expected_recordings} recordings, found {len(rows)}"
report = {"recordings": rows, "total_duration_s": sum(r["duration_s"] for r in rows),
          "total_utterances": sum(r["utterances"] for r in rows), "status": "timings_and_ranges_verified"}
(output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
