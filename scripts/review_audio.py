"""Compare ASR versions and apply explicit, evidence-linked editorial corrections.

This tool never reads reference stories and never guesses replacement text.
All review material is written under outputs/; unedited ASR remains preserved.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
from difflib import SequenceMatcher
from pathlib import Path

from nti_lab.audio import timecode, write_json, question_answer_candidates, words_and_syllables


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def tokens(text):
    return re.findall(r"[\w]+", text.lower().replace("ё", "е"))


def compare(old_dir, new_dir):
    recordings = []
    for path in sorted(new_dir.glob("*.segments.json")):
        new = json.loads(path.read_text(encoding="utf-8"))
        old_path = old_dir / path.name
        old = json.loads(old_path.read_text(encoding="utf-8")) if old_path.exists() else {"segments": []}
        rows = []
        for s in new["segments"]:
            near = [p for p in old["segments"] if p["start"] < s["end"] + 0.15 and p["end"] > s["start"] - 0.15]
            old_words = [w["word"] for p in near for w in p.get("words", [])
                         if s["start"] - 0.15 <= (w["start"] + w["end"]) / 2 <= s["end"] + 0.15]
            before = " ".join(old_words).strip() or " ".join(p["text"] for p in near)
            a, b = tokens(before), tokens(s["text"])
            similarity = SequenceMatcher(None, a, b, autojunk=False).ratio()
            reasons = []
            low_words = [w["word"] for w in s.get("words", []) if w.get("probability", 1) < 0.4]
            if low_words:
                reasons.append("low_word_probability")
            if s.get("avg_logprob") is not None and s["avg_logprob"] < -0.8:
                reasons.append("low_chunk_logprob")
            if a and len(b) >= 3 and similarity < 0.6:
                reasons.append("large_revision_from_turbo")
            if any(x in s["text"].lower() for x in ["продолжение следует", "субтитр", "подписывайтесь", "спасибо за просмотр"]):
                reasons.append("possible_asr_boilerplate")
            if len(b) >= 8 and len(set(b)) / len(b) < 0.35:
                reasons.append("possible_repetition")
            old_neg = sum(x in {"не", "нет", "никогда", "ни"} for x in a)
            new_neg = sum(x in {"не", "нет", "никогда", "ни"} for x in b)
            if a and b and old_neg != new_neg:
                reasons.append("negation_count_differs_check_alignment_and_audio")
            rows.append({"segment_id": s["id"], "start_s": s["start"], "end_s": s["end"],
                         "turbo_text": before, "large_v3_text": s["text"],
                         "token_similarity": round(similarity, 4), "flags": reasons,
                         "low_probability_words": low_words})
        recordings.append({"recording_id": new["recording_id"], "source_sha256": sha256(path),
                           "old_source_sha256": sha256(old_path) if old_path.exists() else None,
                           "segments": rows})
    report = {"method": "Time-aligned token comparison, for triage only; alignment and confidence are not correctness labels.",
              "reference_stories_used": False, "recordings": recordings}
    write_json(new_dir / "transcript_comparison.json", report)
    return report


def apply_review(new_dir, ledger_path):
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("reference_stories_used") is not False:
        raise ValueError("Review must explicitly exclude reference stories")
    destination = new_dir / "reviewed"
    destination.mkdir(exist_ok=True)
    review_by_id = {r["recording_id"]: r for r in ledger["recordings"]}
    totals = {"recordings": 0, "segments": 0, "words": 0, "reviewed_segments": 0,
              "correction_count": 0, "unresolved_count": 0, "recovered_segments": 0}
    details, examples = [], []
    for path in sorted(new_dir.glob("*.segments.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        recording_id = data["recording_id"]
        review = review_by_id.get(recording_id, {})
        if review and review["source_sha256"] != sha256(path):
            raise ValueError(f"Review refers to different ASR snapshot: {recording_id}")
        edits = {e["segment_id"]: e for e in review.get("corrections", [])}
        additions = review.get("additions", [])
        reviewed_ids = set(review.get("reviewed_segment_ids", []))
        unresolved = {u["segment_id"]: u for u in review.get("unresolved", [])}
        valid_ids = {s["id"] for s in data["segments"]}
        if not (set(edits) | reviewed_ids | set(unresolved)) <= valid_ids:
            raise ValueError(f"Unknown segment in review: {recording_id}")
        analysis_path = new_dir / "analysis" / path.name
        analysis = json.loads(analysis_path.read_text(encoding="utf-8")) if analysis_path.exists() else None
        if analysis and [(s["id"], s["text"], s["start"], s["end"]) for s in analysis["segments"]] != [(s["id"], s["text"], s["start"], s["end"]) for s in data["segments"]]:
            raise ValueError("Acoustic/diarization analysis refers to different source utterances")
        result = copy.deepcopy(analysis or data)
        for s in result["segments"]:
            s["text_review_status"] = "assistant_text_review" if s["id"] in reviewed_ids else "not_reviewed"
            if s["id"] in unresolved:
                s["text_review_status"] = "unresolved_requires_listening"
                s["text_review_note"] = unresolved[s["id"]]["reason"]
            if s["id"] not in edits:
                continue
            e = edits[s["id"]]
            if e["before"] != s["text"] or not e.get("evidence") or not e.get("reason"):
                raise ValueError(f"Invalid correction provenance: {recording_id}/{s['id']}")
            if e.get("audio_verified") is not False or e.get("method") != "repeated_local_asr_and_assistant_text_review":
                raise ValueError("Automated review cannot certify human listening")
            evidence_path = new_dir / e["evidence"]["file"]
            if sha256(evidence_path) != e["evidence"]["sha256"]:
                raise ValueError("Correction evidence checksum mismatch")
            s["original_asr_text"] = s["text"]
            s["text"] = e["after"]
            if e.get("replacement_word_times"):
                s["original_asr_words"] = s["words"]
                s["original_asr_start"], s["original_asr_end"] = s["start"], s["end"]
                s["words"] = copy.deepcopy(e["replacement_word_times"])
                s["start"], s["end"] = s["words"][0]["start"], s["words"][-1]["end"]
                s["speaker_id"], s["speaker_role"], s["speaker_label_status"] = "unknown", "unconfirmed", "unconfirmed"
                s["speaker_role_candidate"] = "unconfirmed"
            words, syllables = words_and_syllables(s["text"])
            duration = max(s["end"] - s["start"], 0.01)
            s["words_per_minute"] = round(60 * words / duration, 3)
            s["syllables_per_second_est"] = round(syllables / duration, 3)
            s["text_review_status"] = "corrected_with_repeat_asr_evidence"
            s["text_correction"] = e
            s["word_alignment_status"] = ("repeat_ASR_word_times_not_human_verified" if e.get("replacement_word_times")
                                          else "original_ASR_word_times_not_realigned_after_editorial_text_change")
            examples.append({"recording_id": recording_id, "segment_id": s["id"], "start_s": s["start"],
                             "before": e["before"], "after": e["after"], "reason": e["reason"]})
        for i, addition in enumerate(additions):
            if addition.get("audio_verified") is not False or addition.get("method") != "repeated_local_asr_and_assistant_text_review":
                raise ValueError("Recovered speech must retain automatic-review status")
            if sha256(new_dir / addition["evidence"]["file"]) != addition["evidence"]["sha256"]:
                raise ValueError("Recovered speech evidence checksum mismatch")
            s = copy.deepcopy(addition["segment"])
            s["id"] = 10000 + i
            assert s["id"] not in valid_ids and 0 <= s["start"] <= s["end"] <= data["duration_s"]
            s.update({"speaker_id": "unknown", "speaker_role": "unconfirmed", "speaker_label_status": "unconfirmed",
                      "text_review_status": "speech_recovered_by_repeat_asr", "source_chunk_id": None,
                      "recovery_provenance": addition, "word_alignment_status": "repeat_ASR_word_times_not_human_verified"})
            words, syllables = words_and_syllables(s["text"])
            duration = max(s["end"] - s["start"], 0.01)
            s["words_per_minute"] = round(60 * words / duration, 3)
            s["syllables_per_second_est"] = round(syllables / duration, 3)
            result["segments"].append(s)
            examples.append({"recording_id": recording_id, "segment_id": s["id"], "start_s": s["start"],
                             "before": "[omitted by original ASR]", "after": s["text"], "reason": addition["reason"]})
        result["segments"].sort(key=lambda s: (s["start"], s["end"], s["id"]))
        result["asr_status"] = "automatic_asr_with_assistant_text_review_not_human_audio_verified"
        if "speech_intervals" in result:
            result["qa_candidates"] = question_answer_candidates(result["segments"], result["speech_intervals"])
        record_review_hash = hashlib.sha256(json.dumps(review, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        result["text_review"] = {"ledger": str(ledger_path.resolve()), "ledger_record_sha256": record_review_hash,
                                 "source_asr_sha256": sha256(path), "audio_verified": False,
                                 "manual_listening_performed": False, "reference_stories_used": False,
                                 "reviewed_segments": len(reviewed_ids), "corrections": len(edits), "recovered_segments": len(additions),
                                 "unresolved": len(unresolved)}
        write_json(destination / path.name, result)
        feature_path = new_dir / "analysis" / f"{recording_id}.features.csv"
        if feature_path.exists():
            shutil.copy2(feature_path, destination / feature_path.name)
        summary_path = new_dir / "analysis" / f"{recording_id}.summary.json"
        if summary_path.exists():
            audio_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            audio_summary["qa_candidate_count"] = len(result.get("qa_candidates", []))
            audio_summary["segment_count"] = len(result["segments"])
            audio_summary["text_review"] = result["text_review"]
            write_json(destination / summary_path.name, audio_summary)
        lines = ["Whisper large-v3 · редакторская проверка и повторное распознавание спорных фрагментов", ""]
        for s in result["segments"]:
            marker = " [требует прослушивания]" if s["id"] in unresolved else ""
            if s["id"] in edits:
                marker += " [исправлено по повторному ASR]"
            if s["id"] >= 10000:
                marker += " [восстановлено повторным ASR; говорящий не подтверждён]"
            lines.append(f"[{timecode(s['start'])} --> {timecode(s['end'])}] {s['text']}{marker}")
        (destination / f"{recording_id}.txt").write_text("\n".join(lines), encoding="utf-8")
        count = {"recording_id": recording_id, "segments": len(result["segments"]),
                 "words": sum(len(s.get("words", [])) for s in data["segments"]),
                 "reviewed_segments": len(reviewed_ids), "correction_count": len(edits) + len(additions),
                 "unresolved_count": len(unresolved), "recovered_segments": len(additions)}
        details.append(count)
        totals["recordings"] += 1
        for key in count.keys() - {"recording_id"}:
            totals[key] += count[key]
    summary = {"model": "faster-whisper-large-v3", **totals, "recording_details": details,
               "audio_verified": False, "manual_listening_performed": False,
               "reference_stories_used": False, "review_method": "Assistant read-through and local repeated ASR; no claim of error-free or human-listened transcript.",
               "correction_examples": examples, "word_count_definition": "Original timed words before editorial corrections",
               "reviewed_output": str(destination.resolve())}
    summary["text_correction_count"] = sum(e["before"] != e["after"] for r in ledger["recordings"] for e in r.get("corrections", []))
    summary["timing_only_count"] = sum(e["before"] == e["after"] for r in ledger["recordings"] for e in r.get("corrections", []))
    summary["raw_segments"] = sum(len(r.get("reviewed_segment_ids", [])) for r in ledger["recordings"])
    summary["correction_count_definition"] = "Text replacements + timing-only corrections + recovered segments"
    summary["ledger_sha256"] = sha256(ledger_path)
    checks = [json.loads(p.read_text(encoding="utf-8")) for p in (new_dir / "rechecks").glob("*.json")]
    summary["repeat_review"] = {"clips": len(checks), "decode_passes": sum(len(c["passes"]) for c in checks),
                                "audio_duration_s": round(sum(c["request"]["end_s"] - c["request"]["start_s"] for c in checks), 3),
                                "beam_sizes": [10, 1], "initial_prompt": None, "reference_stories_used": False}
    write_json(new_dir / "validation_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, default=Path("outputs/audio"))
    parser.add_argument("--new", type=Path, default=Path("outputs/audio-large-v3"))
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    promotion = args.new / "promotion.json"
    if args.old == Path("outputs/audio") and promotion.exists():
        args.old = Path(json.loads(promotion.read_text(encoding="utf-8"))["archive"])
    comparison = compare(args.old, args.new)
    print(f"Compared {len(comparison['recordings'])} recordings")
    if args.ledger:
        summary = apply_review(args.new, args.ledger)
        print(json.dumps({k: v for k, v in summary.items() if k not in {"correction_examples", "recording_details"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
