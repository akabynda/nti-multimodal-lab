"""Build reference catalog, link supported profiles, and compare available ASR output."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nti_lab.research import (build_catalog, compare_stories, content_review_report, grouped_validation, materialize_content_review,
                              profile_roster, temperament_hypotheses)
from nti_lab.story_topics import display_story_titles


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def comparison_plot(output: Path, stories: list[dict], matches: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    participants = sorted({s["participant_id"] for s in stories})
    values = np.full((len(participants), 6), np.nan)
    for row in matches:
        column = int(row["candidate_story_id"].rsplit("_", 1)[-1]) - 1
        values[participants.index(row["participant_id"]), column] = row["similarity"]
    with plt.rc_context({"font.size": 10, "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        palette = plt.get_cmap("Blues").copy()
        palette.set_bad("#eeeeee")
        pic = ax.imshow(values, cmap=palette, vmin=0, vmax=1, aspect="auto")
        for i in range(len(participants)):
            for j in range(6):
                ax.text(j, i, "N/A" if np.isnan(values[i, j]) else f"{values[i, j]:.2f}",
                        ha="center", va="center", color="white" if values[i, j] > .65 else "#111111")
        ax.set_yticks(range(len(participants)), [f"P{i+1:02d}" for i in range(len(participants))])
        ax.set_xticks(range(6), [f"Текст {i+1}" for i in range(6)])
        ax.set(xlabel="Порядок историй в исходном документе каждого участника",
               ylabel="Респондент", title="Максимальное лексическое сходство с 45-секундным окном ASR")
        fig.colorbar(pic, ax=ax, label="Косинусное сходство TF-IDF (0–1)")
        fig.savefig(output / "story_similarity.png", dpi=180, facecolor="white")
        fig.savefig(output / "story_similarity.svg", facecolor="white")
        plt.close(fig)
    pd.DataFrame(values, index=[f"P{i+1:02d}" for i in range(len(participants))],
                 columns=[f"story_{i+1:02d}" for i in range(6)]).to_csv(output / "story_similarity_values.csv")
    pd.DataFrame({"plot_id": [f"P{i+1:02d}" for i in range(len(participants))],
                  "participant_id": participants}).to_csv(output / "plot_participant_key.csv", index=False)
    (output / "story_similarity_alt.txt").write_text(
        f"Матрица {len(participants)} участников и 6 письменных историй. Числа показывают максимальное лексическое "
        "сходство с окном ASR длиной 45 секунд. N/A означает отсутствие стенограммы. "
        "Это описательные значения, не вероятности правды или лжи. Порядок тем у участников разный.",
        encoding="utf-8")


def run(root: Path = ROOT, workbook: Path | None = None, audio_dir: Path | None = None,
        notes_path: Path | None = None):
    root = root.resolve()
    private = root / "data/private"
    output = root / "outputs/research"
    output.mkdir(parents=True, exist_ok=True)
    finalized_notes = private / "comparison_notes_large_v3.json"
    if notes_path is None and finalized_notes.exists():
        notes_path = finalized_notes
    if audio_dir is None and notes_path is not None and notes_path.exists():
        selected = json.loads(notes_path.read_text(encoding="utf-8"))
        audio_dir = root / selected.get("asr_dir", "outputs/audio")
    audio_dir = (audio_dir or root / "outputs/audio").resolve()
    source = json.loads((private / "source_documents.json").read_text(encoding="utf-8"))
    spec = json.loads((private / "catalog_spec.json").read_text(encoding="utf-8"))
    truth_path = private / "reference_truth.json"
    reference_truth = json.loads(truth_path.read_text(encoding="utf-8")) if truth_path.exists() else None
    stories = build_catalog(source["documents"], spec, reference_truth)
    write_json(private / "stories.json", {
        "schema_version": 1, "source_folder_url": source["source_folder_url"],
        "source_folder_urls": source.get("source_folder_urls", [source["source_folder_url"]]),
        "source_document_count": len(source["documents"]), "stories": stories})
    text_dir = private / "stories"
    text_dir.mkdir(exist_ok=True)
    for story in stories:
        (text_dir / (story["story_id"] + ".txt")).write_text(story["title"] + "\n\n" + story["text"], encoding="utf-8")
    roster, all_profiles = profile_roster(workbook or root.parent / "Результаты.xlsx", spec)
    roster.to_csv(private / "participant_profiles.csv", index=False, encoding="utf-8-sig")
    roster.to_csv(output / "profile_link_audit.csv", index=False, encoding="utf-8-sig")
    catalog = pd.DataFrame([display_story_titles(s) for s in stories]).drop(columns="text")
    catalog.to_csv(output / "story_catalog.csv", index=False, encoding="utf-8-sig")
    matches, windows, files = [], [], []
    for path in sorted(audio_dir.glob("*.segments.json")):
        recording_id = path.name.removesuffix(".segments.json")
        transcript = json.loads(path.read_text(encoding="utf-8"))
        match, window = compare_stories(stories, transcript, recording_id)
        matches.extend(match)
        windows.extend(window)
        files.append({"recording_id": recording_id, "segments": len(transcript.get("segments", [])),
                      "reference_matches": len(match), "transcript_path": str(path.relative_to(root))})
    display_matches = [display_story_titles(r) for r in matches]
    display_windows = [display_story_titles(r) for r in windows]
    pd.DataFrame(display_matches, columns=list(display_matches[0]) if display_matches else ["recording_id", "candidate_story_id", "similarity", "review_required"]).to_csv(
        output / "story_matches.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(display_windows, columns=list(display_windows[0]) if display_windows else ["recording_id", "candidate_story_id", "start_s", "end_s"]).to_csv(
        output / "window_matches.csv", index=False, encoding="utf-8-sig")
    write_json(output / "story_matches.json", {"schema_version": 1, "matches": matches, "transcripts": files})
    note_path = notes_path or private / "comparison_notes.json"
    content_review = []
    participant_narratives = {}
    if note_path.exists():
        notes_doc = json.loads(note_path.read_text(encoding="utf-8"))
        participant_narratives = notes_doc.get("participant_narratives", {})
        delegated_path = output / "manual_review_first4.json"
        if notes_path is None and delegated_path.exists():
            delegated = json.loads(delegated_path.read_text(encoding="utf-8"))
            reviewed_participants = {n["participant_id"] for n in delegated["notes"]}
            notes_doc["notes"] = [n for n in notes_doc["notes"] if n["participant_id"] not in reviewed_participants] + delegated["notes"]
        content_review = materialize_content_review(
            stories, notes_doc, audio_dir)
        write_json(output / "content_comparison.json", {"schema_version": 1, "comparisons": content_review,
                   "reference_truth_provenance": reference_truth,
                   "revision_provenance": notes_doc.get("revision_provenance"),
                   "method": "Exact reference and transcript quotations with adjacent question/answer context"})
        pd.DataFrame([display_story_titles(r) for r in content_review]).to_csv(output / "content_comparison.csv", index=False, encoding="utf-8-sig")
    content_report = content_review_report(stories, content_review)
    content_report["topic_summaries"] = [display_story_titles(r) for r in content_report["topic_summaries"]]
    for participant_summary in content_report["participant_summaries"]:
        participant_summary["content_summary"] = participant_narratives.get(participant_summary["participant_id"])
    write_json(output / "content_comparison_report.json", content_report)
    pd.DataFrame(content_report["topic_summaries"]).to_csv(
        output / "topic_coverage.csv", index=False, encoding="utf-8-sig")
    comparison_plot(output, stories, matches)
    # Never overwrite human annotations on reruns.
    annotations = private / "reviewed_trials.csv"
    if not annotations.exists():
        pd.DataFrame(columns=["participant_id", "recording_id", "story_id", "start_s", "end_s",
                              "condition", "label_source", "reviewed", "temperament", "temperament_source"]).to_csv(
                                  annotations, index=False, encoding="utf-8-sig")
    features_path = private / "labelled_features.csv"
    if features_path.exists():
        trials = pd.read_csv(features_path)
        feature_spec = private / "research_features.json"
        features = json.loads(feature_spec.read_text(encoding="utf-8")) if feature_spec.exists() else []
        if not features:
            raise ValueError("labelled_features.csv requires prespecified feature names in research_features.json")
        validation = {"baseline": grouped_validation(trials, features),
                      "with_temperament": grouped_validation(trials, features, True)}
        hypotheses = temperament_hypotheses(trials, features)
    else:
        reason = "Written references are user-confirmed truth; video answer labels and validated classical temperament mapping are absent"
        validation = {"status": "blocked", "reason": reason}
        hypotheses = {"status": "blocked", "reason": reason}
    write_json(output / "validation.json", validation)
    write_json(output / "hypotheses.json", hypotheses)
    summary = {
        "source_documents": len(source["documents"]), "stories": len(stories),
        "participants": len(roster), "profile_workbook_rows": len(all_profiles),
        "profiles_linked_by_explicit_tab_number": int(roster.profile_link_status.eq("explicit_tab_number_match").sum()),
        "profiles_requested_id_not_found": int(roster.profile_link_status.eq("requested_id_not_found").sum()),
        "classical_temperaments_known": int(roster.temperament.notna().sum()),
        "source_declared_truth_labels": 0,
        "reference_truth_confirmed_by_user": sum(s["declared_truth_label"] == "truth" for s in stories),
        "transcripts_compared": len(files), "story_match_candidates": len(matches),
        "comparison_status": "review_required" if matches else "awaiting_transcripts",
        "content_comparison_rows": len(content_review),
        "content_reviewed_participants": len({r["participant_id"] for r in content_review}),
        "content_reviewed_stories": content_report["stories_reviewed"],
        "content_relation_counts": content_report["relation_counts"],
        "stories_with_contradiction_candidates": content_report["stories_with_contradiction_candidates"],
        "content_comparison_report": "outputs/research/content_comparison_report.json",
        "comparison_method": "Written reference and transcript quotations with question-and-answer context",
        "hypotheses_status": hypotheses["status"],
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--workbook", type=Path)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--notes", type=Path, help="Complete reviewed notes; disables legacy first-four merge")
    args = parser.parse_args()
    run(args.root, args.workbook, args.audio_dir, args.notes)
