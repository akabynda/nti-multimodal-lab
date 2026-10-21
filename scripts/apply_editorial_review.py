"""Publish the reviewed dialogue while retaining immutable source and decision files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nti_lab.audio import timecode
from nti_lab.editorial import ROLE_NAMES, apply_editorial


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def publish(source_dir, decisions_dir, output):
    prepared = []
    for path in sorted(source_dir.glob("*.segments.json")):
        source = read(path)
        decision_path = decisions_dir / f"{source['recording_id']}.json"
        decision = read(decision_path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != decision["source_sha256"]:
            raise ValueError(f"Source checksum mismatch: {path.name}")
        result = apply_editorial(source, decision)
        result["editorial_review"]["decision_sha256"] = hashlib.sha256(decision_path.read_bytes()).hexdigest()
        prepared.append((path, result))
    if not prepared:
        raise ValueError("No source transcripts")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for path, result in prepared:
        write(output / path.name, result)
        rid = result["recording_id"]
        lines = [f"[{timecode(t['start'])} --> {timecode(t['end'])}] {ROLE_NAMES[t['speaker_role']]}: {t['text']}"
                 for t in result["turns"]]
        (output / f"{rid}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        summary_path = output / f"{rid}.summary.json"
        if summary_path.exists():
            summary = read(summary_path)
            summary["editorial_review"] = result["editorial_review"]
            summary["segment_count"] = len(result["segments"])
            summary["qa_candidate_count"] = len(result["qa_candidates"])
            write(summary_path, summary)
        records.append({"recording_id": rid, **result["editorial_review"],
                        "sha256": hashlib.sha256((output / path.name).read_bytes()).hexdigest()})
    summary = {"recordings": len(records), "reviewed_segments": sum(r["reviewed_segments"] for r in records),
               "turns": sum(r["turns"] for r in records), "text_corrections": sum(r["text_corrections"] for r in records),
               "merge_groups": sum(r["merge_groups"] for r in records), "split_segments": sum(r["split_segments"] for r in records),
               "timing_corrections": sum(r["timing_corrections"] for r in records),
               "recovered_segments": sum(r["recovered_segments"] for r in records),
               "unresolved_text_fragments": sum(len(r["unresolved"]) for r in records),
               "unknown_role_turns": sum(r["role_counts"]["unknown"] for r in records),
               "records": records, "source_directory": str(source_dir.resolve()),
               "decisions_directory": str(decisions_dir.resolve())}
    write(output / "editorial_review.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/audio"))
    args = parser.parse_args()
    summary = publish(args.source, args.decisions, args.output)
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, ensure_ascii=False, indent=2))
