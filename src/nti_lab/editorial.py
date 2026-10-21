"""Apply a complete, explicit dialogue review without changing its ASR source."""
from __future__ import annotations

import copy
import re

from .audio import words_and_syllables

ROLES = {"interviewer", "respondent", "unknown"}
ROLE_NAMES = {"interviewer": "Интервьюер", "respondent": "Респондент", "unknown": "Говорящий"}


def join_fragments(texts):
    text = ""
    for part in texts:
        part = part.strip()
        if text and re.match(r"^-(?:то|нибудь|либо|таки)\b", part, re.I):
            text = text.rstrip() + part
        else:
            text += (" " if text else "") + part
    return text


def dialogue_pairs(turns):
    """Collect whole answers using reviewed roles, retaining source segment IDs."""
    blocks = []
    for turn in turns:
        if blocks and blocks[-1]["speaker_role"] == turn["speaker_role"]:
            block = blocks[-1]
            block["text"] += " " + turn["text"]
            block["end"] = turn["end"]
            block["segment_ids"].extend(turn["segment_ids"])
            block["turn_ids"].append(turn["id"])
        else:
            blocks.append(copy.deepcopy(turn))
            blocks[-1]["turn_ids"] = [turn["id"]]
    pairs = []
    for question, answer in zip(blocks, blocks[1:]):
        if question["speaker_role"] != "interviewer" or answer["speaker_role"] != "respondent":
            continue
        if "?" not in question["text"] and not re.search(r"расскаж|опиш|назов|привед|представ|вспомн", question["text"], re.I):
            continue
        pairs.append({
            "question_segment_id": question["segment_ids"][0],
            "answer_segment_id": answer["segment_ids"][0],
            "question_segment_ids": question["segment_ids"], "answer_segment_ids": answer["segment_ids"],
            "question_turn_ids": question["turn_ids"], "answer_turn_ids": answer["turn_ids"],
            "question_start_s": question["start"], "question_end_s": question["end"],
            "answer_start_s": answer["start"], "answer_end_s": answer["end"],
            "latency_s": round(answer["start"] - question["end"], 3),
            "answer_duration_s": round(answer["end"] - answer["start"], 3),
            "question": question["text"], "answer": answer["text"],
            "method": "editorially_reviewed_dialogue_roles",
        })
    return pairs


def apply_editorial(source, decision):
    if source["recording_id"] != decision["recording_id"]:
        raise ValueError("Recording ID mismatch")
    result = copy.deepcopy(source)
    segments = result["segments"]
    ids = [s["id"] for s in segments]
    reviewed = decision["reviewed_segment_ids"]
    roles = decision["roles"]
    if len(set(ids)) != len(ids) or sorted(reviewed) != sorted(ids):
        raise ValueError("Review must cover every source segment exactly once")
    if sorted(r["segment_id"] for r in roles) != sorted(ids):
        raise ValueError("Roles must cover every source segment exactly once")
    by_id = {s["id"]: s for s in segments}
    timed = set()
    for edit in decision.get("timing_corrections", []):
        s = by_id.get(edit["segment_id"])
        if s is None or s["id"] in timed or (s["start"], s["end"]) != (edit["before_start"], edit["before_end"]):
            raise ValueError("Timing correction does not match source")
        if not edit.get("reason") or not edit.get("evidence") or not edit.get("words"):
            raise ValueError("Timing correction requires aligned evidence")
        if not 0 <= edit["start"] < edit["end"] <= source["duration_s"]:
            raise ValueError("Timing correction outside recording")
        if any(not edit["start"] <= w["start"] <= w["end"] <= edit["end"] for w in edit["words"]):
            raise ValueError("Corrected word outside segment")
        timed.add(s["id"])
        s["pre_editorial_timing"] = {"start": s["start"], "end": s["end"], "words": s.get("words", [])}
        s["start"], s["end"], s["words"] = edit["start"], edit["end"], copy.deepcopy(edit["words"])
        s["editorial_timing_correction"] = copy.deepcopy(edit)
        s["word_alignment_status"] = "repeat_ASR_word_times_editorially_reviewed"
    for row in roles:
        if row["role"] not in ROLES:
            raise ValueError("Invalid dialogue role")
        s = by_id[row["segment_id"]]
        s["original_speaker_role"] = s.get("speaker_role", "unconfirmed")
        s["speaker_role"] = row["role"]
        s["speaker_label_status"] = "editorial_context_review"
        s["speaker_review_reason"] = row.get("reason", "Role reviewed in the complete dialogue")
    edited = set()
    for edit in decision.get("corrections", []):
        s = by_id.get(edit["segment_id"])
        if s is None or s["id"] in edited or s["text"] != edit["before"] or not edit.get("reason") or not edit["after"].strip():
            raise ValueError("Invalid or duplicate text correction")
        edited.add(s["id"])
        s["pre_editorial_text"] = s["text"]
        s["text"] = edit["after"].strip()
        s["editorial_correction"] = copy.deepcopy(edit)
        normalize = lambda text: re.sub(r"\W+", "", text).casefold()
        aligned = s["id"] in timed and normalize(" ".join(w["word"] for w in s["words"])) == normalize(s["text"])
        if aligned:
            s["word_alignment_status"] = "repeat_ASR_word_times_editorially_reviewed"
        elif s["id"] in timed:
            s["word_alignment_status"] = "repeat_ASR_word_times_not_realigned_after_editorial_text_change"
        else:
            s["word_alignment_status"] = "original_ASR_word_times_not_realigned_after_editorial_text_change"
        words, syllables = words_and_syllables(s["text"])
        duration = max(s["end"] - s["start"], .01)
        s["words_per_minute"] = round(60 * words / duration, 3)
        s["syllables_per_second_est"] = round(syllables / duration, 3)
    source_count = len(ids)
    recovered = decision.get("recovered_segments", [])
    for addition in recovered:
        if addition["id"] in by_id or addition["role"] not in ROLES or not addition.get("evidence") or not addition.get("reason"):
            raise ValueError("Invalid recovered segment or missing evidence")
        if not addition["text"].strip() or not 0 <= addition["start"] < addition["end"] <= source["duration_s"]:
            raise ValueError("Invalid recovered segment text or timing")
        if not addition.get("words") or any(not addition["start"] <= w["start"] <= w["end"] <= addition["end"] for w in addition["words"]):
            raise ValueError("Recovered segment requires aligned words")
        s = {k: copy.deepcopy(addition[k]) for k in ("id", "start", "end", "text", "words")}
        s.update({"speaker_id": "unknown", "speaker_role": addition["role"],
                  "speaker_label_status": "editorial_context_review", "editorial_recovery": copy.deepcopy(addition),
                  "word_alignment_status": "repeat_ASR_word_times_editorially_reviewed"})
        segments.append(s)
        by_id[s["id"]] = s
    segments.sort(key=lambda s: (s["start"], s["end"], s["id"]))
    ids = [s["id"] for s in segments]
    for s in segments:
        words, syllables = words_and_syllables(s["text"])
        duration = max(s["end"] - s["start"], .01)
        s["words_per_minute"] = round(60 * words / duration, 3)
        s["syllables_per_second_est"] = round(syllables / duration, 3)
    # Splits are display-level turns: source segments and identifiers stay available.
    splits = {s["segment_id"]: s["parts"] for s in decision.get("split_segments", [])}
    if len(splits) != len(decision.get("split_segments", [])) or not set(splits) <= set(ids):
        raise ValueError("Invalid split IDs")
    groups, grouped = {}, set()
    for group in decision.get("merge_groups", []):
        if len(group) < 2 or any(i not in by_id or i in grouped or i in splits for i in group):
            raise ValueError("Merge groups must be disjoint source segments")
        start = ids.index(group[0])
        if ids[start:start + len(group)] != group:
            raise ValueError("Only adjacent source segments can be merged")
        if len({by_id[i]["speaker_role"] for i in group}) != 1:
            raise ValueError("Cannot merge different dialogue roles")
        groups[group[0]] = group
        grouped.update(group)
    turns = []
    for s in segments:
        sid = s["id"]
        if sid in splits:
            parts = splits[sid]
            if len(parts) < 2 or " ".join(p["text"].strip() for p in parts) != s["text"]:
                raise ValueError("Split must preserve the complete corrected text")
            previous = s["start"]
            for n, part in enumerate(parts):
                if part["role"] not in ROLES or not s["start"] <= part["start"] < part["end"] <= s["end"] or part["start"] < previous:
                    raise ValueError("Invalid split timing or role")
                previous = part["end"]
                turns.append({"id": f"{sid}.{n}", "segment_ids": [sid], "start": part["start"],
                              "end": part["end"], "speaker_role": part["role"], "text": part["text"].strip()})
        elif sid not in grouped or sid in groups:
            group = groups.get(sid, [sid])
            turns.append({"id": sid, "segment_ids": group, "start": s["start"], "end": by_id[group[-1]]["end"],
                          "speaker_role": s["speaker_role"], "text": join_fragments(by_id[i]["text"] for i in group)})
    unresolved = decision.get("unresolved", [])
    if not {u["segment_id"] for u in unresolved} <= set(ids):
        raise ValueError("Unknown unresolved segment")
    unresolved_by_id = {u["segment_id"]: u["reason"] for u in unresolved}
    for s in segments:
        s["pre_editorial_text_review_status"] = s.get("text_review_status")
        s["text_review_status"] = "editorial_context_reviewed"
        if s["id"] in unresolved_by_id:
            s["text_review_status"] = "editorial_requires_audio_clarification"
            s["editorial_review_note"] = unresolved_by_id[s["id"]]
    result["turns"] = turns
    result["pre_editorial_qa_candidates"] = result.get("qa_candidates", [])
    result["qa_candidates"] = dialogue_pairs(turns)
    result["editorial_review"] = {
        "source_sha256": decision["source_sha256"], "reviewed_segments": source_count, "recovered_segments": len(recovered),
        "text_corrections": len(edited), "timing_corrections": len(timed), "merge_groups": len(groups), "split_segments": len(splits),
        "turns": len(turns), "role_counts": {role: sum(t["speaker_role"] == role for t in turns) for role in ROLES},
        "unresolved": unresolved, "method": "complete_assistant_contextual_readthrough",
        "reference_stories_used": False, "manual_listening_performed": False,
    }
    return result
