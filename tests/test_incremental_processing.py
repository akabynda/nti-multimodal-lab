"""Adding a recording must preserve existing participants and accept media names."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selected_bio_input_merges_and_preserves_existing_results(tmp_path, monkeypatch):
    module = load_script("analyze_bio")
    data, output = tmp_path / "data", tmp_path / "bio"
    data.mkdir()
    output.mkdir()
    original = data / "First.bcrx"
    original.write_bytes(b"first source")
    incoming = data / "arbitrary-session-name.BCRX"
    incoming.write_bytes(b"second source")
    calls = []

    def record(path, destination, subject_id):
        folder = destination / subject_id
        folder.mkdir(exist_ok=True)
        metadata = {"subject_id": subject_id, "window_count": 1,
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        pd.DataFrame([{"subject_id": subject_id, "hr_bpm": 80}]).to_csv(folder / "metrics.csv", index=False)
        for name in ("signals.npz", "timeseries.json", "rr_intervals.csv"):
            (folder / name).write_bytes(b"saved result")
        calls.append(subject_id)

    record(original, output, "First")
    old_files = {p.name: p.read_bytes() for p in (output / "First").iterdir()}
    calls.clear()
    monkeypatch.setattr(module, "analyze_recording", record)
    monkeypatch.setattr(module, "aggregate_plot", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["analyze_bio.py", "--data", str(data), "--output", str(output),
                                     "--input", str(incoming), "--subject-id", "Second"])
    module.main()
    assert calls == ["Second"]
    assert {r["subject_id"] for r in json.loads((output / "summary.json").read_text())["recordings"]} == {"First", "Second"}
    assert len(pd.read_csv(output / "windows.csv")) == 2
    assert old_files == {p.name: p.read_bytes() for p in (output / "First").iterdir()}
    module.main()
    assert calls == ["Second"]  # A complete unchanged input is kept on another run.


def test_media_resolution_accepts_mp4_and_requires_explicit_duplicate_path(tmp_path):
    module = load_script("estimate_sync")
    nested = tmp_path / "download"
    nested.mkdir()
    media = nested / "interview.mp4"
    media.write_bytes(b"video placeholder")
    video = {"source_file": media.name, "recording_id": "arbitrary-recording-id"}
    assert module.resolve_media_source(video, tmp_path) == media
    duplicate = tmp_path / "other"
    duplicate.mkdir()
    duplicate_media = duplicate / media.name
    duplicate_media.write_bytes(b"another source")
    with pytest.raises(ValueError, match="Ambiguous"):
        module.resolve_media_source(video, tmp_path)
    video["source_path"] = str(media)
    assert module.resolve_media_source(video, tmp_path) == media
