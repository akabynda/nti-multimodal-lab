"""Editorial corrections must remain traceable to immutable ASR snapshots."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("audio_review", Path(__file__).parents[1] / "scripts/review_audio.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def fixture_files(tmp_path):
    source = tmp_path / "00001_Participant01.segments.json"
    data = {"recording_id": "00001_Participant01", "participant_id": "Participant01", "duration_s": 10,
            "segments": [{"id": 0, "start": 1, "end": 2, "text": "Станциями.",
                          "words": [{"start": 1, "end": 2, "word": "Станциями.", "probability": 0.5}]}]}
    source.write_text(json.dumps(data), encoding="utf-8")
    evidence = tmp_path / "repeat.json"
    evidence.write_text('{"synthetic_test_evidence": true}', encoding="utf-8")
    ledger = {"reference_stories_used": False, "recordings": [{
        "recording_id": data["recording_id"], "source_sha256": review.sha256(source), "reviewed_segment_ids": [0],
        "corrections": [{"segment_id": 0, "before": "Станциями.", "after": "Танцами.",
                         "reason": "Synthetic provenance test", "method": "repeated_local_asr_and_assistant_text_review",
                         "audio_verified": False, "evidence": {"file": evidence.name, "sha256": review.sha256(evidence)}}],
        "unresolved": []}]}
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    return source, ledger_path, evidence


def test_edit_preserves_raw_asr_and_never_certifies_listening(tmp_path):
    source, ledger, evidence = fixture_files(tmp_path)
    original = source.read_bytes()
    summary = review.apply_review(tmp_path, ledger)
    revised = json.loads((tmp_path / "reviewed" / source.name).read_text(encoding="utf-8"))
    assert source.read_bytes() == original
    assert revised["segments"][0]["text"] == "Танцами."
    assert revised["segments"][0]["original_asr_text"] == "Станциями."
    assert revised["text_review"]["audio_verified"] is False
    assert revised["text_review"]["manual_listening_performed"] is False
    assert summary["correction_count"] == 1


def test_edit_refuses_changed_evidence(tmp_path):
    source, ledger, evidence = fixture_files(tmp_path)
    evidence.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        review.apply_review(tmp_path, ledger)


def test_edit_refuses_changed_source_asr(tmp_path):
    source, ledger, evidence = fixture_files(tmp_path)
    source.write_text(source.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="different ASR snapshot"):
        review.apply_review(tmp_path, ledger)
