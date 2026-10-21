import json

from fastapi.testclient import TestClient

from nti_lab import app as module


def analysis_client(tmp_path, monkeypatch):
    folder = tmp_path / "outputs/research/multimodal"
    folder.mkdir(parents=True)
    monkeypatch.setattr(module, "OUTPUT", tmp_path / "outputs")
    return TestClient(module.app), folder


def test_analysis_missing_or_invalid_summary_is_not_run(tmp_path, monkeypatch):
    client, folder = analysis_client(tmp_path, monkeypatch)
    assert client.get("/api/analysis").json() == {"status": "not_run"}
    for content in ("{", "[]", '{"status":"not_run"}'):
        (folder / "summary.json").write_text(content, encoding="utf-8")
        assert client.get("/api/analysis").json() == {"status": "not_run"}


def test_analysis_exposes_only_aggregate_fields_and_preserves_missing_values(tmp_path, monkeypatch):
    client, folder = analysis_client(tmp_path, monkeypatch)
    summary = {
        "status": "complete", "findings_ru": ["Результат"],
        "dataset": {"participants": 9, "trials": 37, "agreements": 27, "contradictions": 10,
                    "participant_ids": ["private"]},
        "models": [{"id": "av", "name": "Аудио и видео",
                    "metrics": {"brier": 0, "balanced_accuracy": None, "roc_auc": .5,
                                "participant_predictions": ["private"]},
                    "predictions": ["private"]}],
        "profile_increment": {"delta_brier": 0, "ci_low": None, "ci_high": .1,
                              "participant_ids": ["private"]},
        "associations": [{"profile_label": "Эмоц+", "feature_label": "Темп", "n": 9,
                          "rho": 0, "p_holm": None, "participant_values": ["private"]}],
        "explanation_ru": "Согласование / противоречие эталону.",
        "participant_predictions": ["private"],
    }
    (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    response = client.get("/api/analysis")
    assert response.status_code == 200
    result = response.json()
    assert result == {
        "status": "complete", "findings_ru": ["Результат"],
        "dataset": {"participants": 9, "trials": 37, "agreements": 27, "contradictions": 10},
        "models": [{"id": "av", "name": "Аудио и видео",
                    "metrics": {"brier": 0, "balanced_accuracy": None, "roc_auc": .5}}],
        "profile_increment": {"delta_brier": 0, "ci_low": None, "ci_high": .1},
        "associations": [{"profile_label": "Эмоц+", "feature_label": "Темп", "n": 9,
                          "rho": 0, "p_holm": None}],
        "explanation_ru": "Согласование / противоречие эталону.",
    }


def test_analysis_downloads_allowlisted_results_only(tmp_path, monkeypatch):
    client, folder = analysis_client(tmp_path, monkeypatch)
    for filename in ("summary.json", "participant_features.csv", "trial_features.csv",
                     "associations.csv", "model_predictions.csv", "analysis.md",
                     "analysis-results.zip", "associations.png", "model_comparison.png"):
        (folder / filename).write_bytes(b"result")
        response = client.get(f"/analysis/{filename}")
        assert response.status_code == 200
        assert response.content == b"result"
        assert "attachment" in response.headers["content-disposition"]
    (folder / "private.json").write_text("private", encoding="utf-8")
    for url in ("/analysis/private.json", "/analysis/%2E%2E%2Fprivate.json",
                "/analysis/%2E%2E%5Cprivate.json", "/analysis/SUMMARY.JSON"):
        assert client.get(url).status_code == 404


def test_analysis_missing_allowed_download_is_404(tmp_path, monkeypatch):
    client, _ = analysis_client(tmp_path, monkeypatch)
    assert client.get("/analysis/analysis-results.zip").status_code == 404
