"""Reference-text comparison and participant-level research with explicit data gates.

Text similarity is a retrieval aid. It is never a truth, deception, or emotion score.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_catalog(documents: list[dict], specification: list[dict], reference_truth: dict | None = None) -> list[dict]:
    """Use inspected headings or individual story files, retaining source spans."""
    source_by_id = {d["id"]: d for d in documents}
    stories = []
    for spec in specification:
        units = []
        files = spec.get("source_files") or [spec]
        for file_spec in files:
            doc = source_by_id[file_spec["source_file_id"]]
            text = doc["text"].replace("\r\n", "\n").replace("\xa0", " ").lstrip("\ufeff")
            headings = [file_spec["heading"]] if "heading" in file_spec else file_spec["headings"]
            locations = []
            for heading in headings:
                matches = list(re.finditer(re.escape(heading), text))
                if len(matches) != 1:
                    raise ValueError(f"Heading ambiguous/missing: {doc['id']}: {heading!r}")
                locations.append((matches[0].start(), matches[0].end(), heading))
            if locations != sorted(locations):
                raise ValueError("Headings must follow source order")
            for local_number, (start, body_start, heading) in enumerate(locations, 1):
                end = locations[local_number][0] if local_number < len(locations) else len(text)
                marker = file_spec.get("body_end_marker")
                if marker:
                    found = list(re.finditer(re.escape(marker), text))
                    if len(found) != 1 or not body_start <= found[0].start() <= end:
                        raise ValueError("Body end marker ambiguous or outside story")
                    end = found[0].start()
                number = file_spec.get("story_number", local_number)
                units.append((number, start, body_start, end, heading, text, doc))
        for n, start, body_start, end, heading, text, doc in units:
            body = text[body_start:end].strip()
            stories.append({
                "story_id": f"{spec['participant_id']}_story_{n:02d}",
                "participant_id": spec["participant_id"],
                "title": " ".join(heading.split()),
                "text": body,
                "text_sha256": sha256_text(body),
                "source_file_id": doc["id"],
                "source_title": doc["title"],
                "source_url": doc["url"],
                "source_modified_time": doc["modified_time"],
                "source_text_sha256": sha256_text(doc["text"]),
                "normalized_source_span": [start, end],
                "extraction_method": doc["extraction_method"],
                "declared_truth_label": reference_truth["label"] if reference_truth else None,
                "label_provenance": reference_truth["provenance"] if reference_truth else None,
                "label_status": "reference_truth_confirmed_by_user" if reference_truth else "not_declared_in_source",
                "participant_mapping_evidence": spec["participant_mapping_evidence"],
            })
    ids = [s["story_id"] for s in stories]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate story numbers in participant catalog")
    return stories


def profile_roster(workbook: Path, specification: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join attested ISU identifiers to Таб.; retain source values and locations."""
    def identifier(value):
        if value is None or pd.isna(value):
            return None
        value = str(value).strip()
        if re.fullmatch(r"\d+\.0+", value):
            value = value.split(".")[0]
        if not re.fullmatch(r"\d+", value):
            raise ValueError(f"Invalid ISU identifier: {value!r}")
        return value

    with pd.ExcelFile(workbook) as excel:
        sheet = excel.sheet_names[0]
        source = pd.read_excel(excel, sheet_name=sheet)
    source["profile_source_row"] = source.index + 2
    source = source.dropna(subset=["Таб."]).copy()
    source["tab_number"] = source["Таб."].map(identifier)
    if source["tab_number"].duplicated().any():
        raise ValueError("Duplicate tab numbers prevent unambiguous profile joins")
    requested = [identifier(s.get("tab_number")) for s in specification]
    known = [key for key in requested if key is not None]
    if len(known) != len(set(known)):
        raise ValueError("Duplicate participant ISU identifiers")
    indexed = source.set_index("tab_number")
    workbook_hash = hashlib.sha256(workbook.read_bytes()).hexdigest()
    rows = []
    for spec, tab in zip(specification, requested):
        linked = tab in indexed.index if tab else False
        profile = indexed.loc[tab] if linked else {}
        rows.append({
            "participant_id": spec["participant_id"], "tab_number": tab,
            "profile_link_status": ("explicit_tab_number_match" if linked else
                                    "requested_id_not_found" if tab else "unlinked"),
            "profile_link_evidence": spec.get("tab_number_evidence"),
            "profile_source_file": workbook.name,
            "profile_source_sha256": workbook_hash,
            "profile_source_sheet": sheet,
            "profile_source_row": profile.get("profile_source_row"),
            "profile_source_number": profile.get("№"),
            "l34_r": profile.get("Л34(р)"), "l34_p": profile.get("Л34(п)"),
            "profile_natural_raw": profile.get("Естественный"),
            "profile_demonstrated_raw": profile.get("Демонстрируемый"),
            "expression": profile.get("Выраженность"),
            "emotion_positive": profile.get("Эмоц+"), "emotion_negative": profile.get("Эмоц-"),
            "emotion_balance": profile.get("Эмоц_Баланс"),
            "temperament": None, "temperament_source": None,
            "taxonomy_note": "Source profiles are not a validated mapping to the four classical temperaments",
        })
    roster = pd.DataFrame(rows)
    roster["profile_source_row"] = roster["profile_source_row"].astype("Int64")
    return roster, source


def transcript_windows(segments: list[dict], window_s: float = 45, step_s: float = 10) -> list[dict]:
    """Use word timestamps where available; otherwise retain complete ASR segments."""
    units = []
    for i, segment in enumerate(segments):
        words = segment.get("words") or []
        stale_alignment = segment.get("word_alignment_status") in {
            "original_ASR_word_times_not_realigned_after_editorial_text_change",
            "repeat_ASR_word_times_not_realigned_after_editorial_text_change",
        }
        if words and not stale_alignment:
            units.extend({"start": w["start"], "end": w["end"], "text": w.get("word", ""),
                          "segment_id": segment.get("id", i), "alignment_unit": "word"}
                         for w in words)
        else:
            units.append({**segment, "segment_id": segment.get("id", i), "alignment_unit": "segment"})
    valid = [s for s in units if s.get("text", "").strip()
             and np.isfinite(s["start"]) and np.isfinite(s["end"]) and s["end"] > s["start"]]
    if not valid:
        return []
    windows = []
    for start in np.arange(0, max(s["end"] for s in valid), step_s):
        selected = [s for s in valid if start <= (s["start"] + s["end"]) / 2 < start + window_s]
        if selected:
            text = " ".join(s["text"].strip() for s in selected)
            if len(re.findall(r"\w+", text)) >= 5:
                windows.append({"start_s": min(s["start"] for s in selected),
                                "end_s": max(s["end"] for s in selected), "text": text,
                                "segment_ids": sorted(set(s["segment_id"] for s in selected)),
                                "alignment_unit": "word" if all(s["alignment_unit"] == "word" for s in selected) else "segment"})
    return windows


def compare_stories(stories: list[dict], transcript: dict, recording_id: str) -> tuple[list[dict], list[dict]]:
    """Return best window per story, and best story per window for manual review.

    IDF uses the complete reference catalog, with no truth or temperament labels.
    Candidate stories are restricted to the participant specified by the user filenames.
    """
    participant = recording_id.split("_", 1)[-1] if recording_id[:5].isdigit() else recording_id
    own = [s for s in stories if s["participant_id"] == participant]
    windows = transcript_windows(transcript.get("segments", []))
    if not own or not windows:
        return [], []
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=False, sublinear_tf=True)
    reference = vectorizer.fit_transform([normalized_text(s["text"]) for s in stories])
    subset = [i for i, s in enumerate(stories) if s["participant_id"] == participant]
    query = vectorizer.transform([normalized_text(w["text"]) for w in windows])
    scores = (query @ reference[subset].T).toarray()
    provenance = {
        "recording_id": recording_id, "participant_id": participant,
        "method": "char_wb_tfidf_3_5_cosine", "window_s": 45, "step_s": 10,
        "asr_model": transcript.get("model"), "transcript_sha256": sha256_text(json.dumps(transcript, ensure_ascii=False, sort_keys=True)),
        "review_required": True, "truth_label": None,
        "interpretation": "Lexical overlap only; differences may reflect paraphrase or ASR errors",
    }
    per_story, per_window = [], []
    for j, story in enumerate(own):
        i = int(scores[:, j].argmax())
        w = windows[i]
        alternatives = np.delete(scores[i], j)
        per_story.append({
            **provenance, "candidate_story_id": story["story_id"], "story_title": story["title"],
            "source_url": story["source_url"], "source_text_sha256": story["text_sha256"],
            "start_s": w["start_s"], "end_s": w["end_s"],
            "alignment_unit": w["alignment_unit"],
            "similarity": float(scores[i, j]),
            "margin_over_other_stories": float(scores[i, j] - alternatives.max()) if len(alternatives) else None,
            "transcript_excerpt": w["text"], "source_excerpt": story["text"],
        })
    for i, w in enumerate(windows):
        order = np.argsort(scores[i])[::-1]
        j = int(order[0])
        per_window.append({
            **provenance, "start_s": w["start_s"], "end_s": w["end_s"],
            "alignment_unit": w["alignment_unit"],
            "candidate_story_id": own[j]["story_id"], "story_title": own[j]["title"],
            "similarity": float(scores[i, j]),
            "runner_up_story_id": own[int(order[1])]["story_id"] if len(order) > 1 else None,
            "margin": float(scores[i, j] - scores[i, order[1]]) if len(order) > 1 else None,
            "transcript_excerpt": w["text"],
        })
    return per_story, per_window


def materialize_content_review(stories: list[dict], notes_document: dict, audio_dir: Path) -> list[dict]:
    """Check manually assessed text comparisons against exact reference and ASR evidence."""
    catalog = {s["story_id"]: s for s in stories}
    transcripts = {}
    for path in audio_dir.glob("*.segments.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        participant = doc.get("participant_id") or path.name.split("_", 1)[-1].removesuffix(".segments.json")
        transcripts[participant] = (path, doc)
    rows = []
    for number, note in enumerate(notes_document["notes"], 1):
        story_id = f"{note['participant_id']}_story_{note['story_number']:02d}"
        story = catalog[story_id]
        if normalized_text(note["source_excerpt"]) not in normalized_text(story["text"]):
            raise ValueError(f"Source quote not found for {story_id}: {note['source_excerpt']!r}")
        path, transcript = transcripts[note["participant_id"]]
        segments = transcript["segments"]
        full_text = " ".join(s["text"] for s in segments)
        evidence = note.get("spoken_excerpt")
        context = []
        if evidence:
            if normalized_text(evidence) not in normalized_text(full_text):
                raise ValueError(f"ASR quote not found for {story_id}: {evidence!r}")
            timed = " ".join(s["text"] for s in segments
                             if s["end"] >= note["start_s"] and s["start"] <= note["end_s"])
            if normalized_text(evidence) not in normalized_text(timed):
                raise ValueError(f"ASR quote outside declared interval for {story_id}")
            context = [s for s in segments
                       if s["end"] >= note["start_s"] - 10 and s["start"] <= note["end_s"] + 5]
        for extra in note.get("additional_spoken_evidence", []):
            timed = " ".join(s["text"] for s in segments
                             if s["end"] >= extra["start_s"] and s["start"] <= extra["end_s"])
            if normalized_text(extra["spoken_excerpt"]) not in normalized_text(timed):
                raise ValueError(f"Additional ASR quote outside declared interval for {story_id}")
            if extra.get("question_context") and normalized_text(extra["question_context"]) not in normalized_text(timed):
                raise ValueError(f"Additional question outside declared interval for {story_id}")
        context_excerpt = " ".join(s["text"] for s in context) if context else None
        if note.get("question_context") and normalized_text(note["question_context"]) not in normalized_text(context_excerpt or ""):
            raise ValueError(f"Question outside declared context for {story_id}")
        rows.append({
            "comparison_id": f"comparison_{number:03d}", **note,
            "story_id": story_id, "story_title": story["title"],
            "recording_id": path.name.removesuffix(".segments.json"),
            "source_url": story["source_url"], "source_truth_label": story["declared_truth_label"],
            "source_text_sha256": story["text_sha256"], "asr_text_sha256": sha256_text(full_text),
            "asr_model": transcript.get("model"), "reviewer": notes_document["reviewer"],
            "comparison_basis": "Known-true written reference versus automatic transcript; assistant textual review",
            "context_excerpt": context_excerpt,
            "context_start_s": min(s["start"] for s in context) if context else None,
            "context_end_s": max(s["end"] for s in context) if context else None,
            "speaker_role_status": "editorial_context_review" if transcript.get("editorial_review") else "not_verified",
            "audio_verified": False, "review_required": True, "video_truth_label": None,
        })
    return rows


def content_review_report(stories: list[dict], rows: list[dict]) -> dict:
    """Aggregate evidence coverage without converting textual relations into lie labels."""
    relations = ["consistent", "different", "omitted", "uncertain", "contradiction_candidate"]
    unknown = {r["relation"] for r in rows} - set(relations)
    if unknown:
        raise ValueError(f"Unknown comparison relation: {unknown}")
    story_ids = {s["story_id"] for s in stories}
    if {r["story_id"] for r in rows} - story_ids:
        raise ValueError("Comparison refers to unknown source story")
    topics = []
    for story in stories:
        evidence = [r for r in rows if r["story_id"] == story["story_id"]]
        counts = Counter(r["relation"] for r in evidence)
        candidates = [r["comparison_id"] for r in evidence if r["relation"] == "contradiction_candidate"]
        topics.append({
            "participant_id": story["participant_id"], "story_id": story["story_id"],
            "story_title": story["title"], "source_url": story["source_url"],
            "source_truth_label": story["declared_truth_label"],
            "coverage": "reviewed" if evidence else "not_reviewed",
            "relation_counts": {key: counts[key] for key in relations},
            "comparison_ids": [r["comparison_id"] for r in evidence],
            "contradiction_candidate_ids": candidates,
            "summary": " ".join(r["interpretation"] for r in evidence) if evidence else "Тема ещё не сопоставлена.",
            "video_truth_label": None,
        })
    participants = []
    for participant in sorted({s["participant_id"] for s in stories}):
        evidence = [r for r in rows if r["participant_id"] == participant]
        owned = [s for s in topics if s["participant_id"] == participant]
        counts = Counter(r["relation"] for r in evidence)
        candidates = [r for r in evidence if r["relation"] == "contradiction_candidate"]
        participants.append({
            "participant_id": participant,
            "recording_id": evidence[0]["recording_id"] if evidence else None,
            "stories_total": len(owned),
            "stories_reviewed": sum(s["coverage"] == "reviewed" for s in owned),
            "evidence_rows": len(evidence),
            "relation_counts": {key: counts[key] for key in relations},
            "candidate_story_ids": sorted({r["story_id"] for r in candidates}),
            "candidate_intervals": [{"comparison_id": r["comparison_id"], "story_id": r["story_id"],
                                    "start_s": r["start_s"], "end_s": r["end_s"],
                                    "spoken_excerpt": r["spoken_excerpt"],
                                    "interpretation": r["interpretation"]} for r in candidates],
            "summary": (f"Сопоставлено тем: {sum(s['coverage'] == 'reviewed' for s in owned)} из {len(owned)}. "
                        f"Фрагментов с несоответствием эталону: {len(candidates)}; "
                        f"затронуто историй: {len({r['story_id'] for r in candidates})}."),
            "video_truth_label": None,
        })
    counts = Counter(r["relation"] for r in rows)
    return {
        "schema_version": 1, "analysis_unit": "source_story_with_multiple_evidence_rows",
        "stories_total": len(stories), "stories_reviewed": sum(t["coverage"] == "reviewed" for t in topics),
        "participants_reviewed": sum(p["stories_reviewed"] > 0 for p in participants),
        "evidence_rows": len(rows), "relation_counts": {key: counts[key] for key in relations},
        "stories_with_contradiction_candidates": sum(bool(t["contradiction_candidate_ids"]) for t in topics),
        "participant_summaries": participants, "topic_summaries": topics,
        "limits": [
            "Письменные истории используются как истинные эталоны.",
            "Число заметок не равно числу независимых ответов: один ответ может сопоставляться с несколькими историями.",
            "Сверка выполнена по автоматической стенограмме с сохранением соседних вопросов и ответов.",
        ],
        "audio_verified": False, "review_required": True,
    }


def validated_trials(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict]:
    """Require externally reviewed labels. Report exclusions instead of silently dropping."""
    required = {"participant_id", "condition", "label_source", "reviewed", *features}
    absent = sorted(required - set(frame.columns))
    if absent:
        return pd.DataFrame(), {"status": "blocked", "reasons": ["missing_columns"], "missing_columns": absent}
    labelled = frame["condition"].isin(["truth", "lie"])
    provenance = frame["label_source"].fillna("").astype(str).str.strip().ne("")
    reviewed = frame["reviewed"].map(lambda x: x is True or str(x).lower() == "true")
    participant = frame["participant_id"].notna() & frame["participant_id"].astype(str).str.strip().ne("")
    numeric = frame[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    usable = numeric.notna().any(axis=1)
    keep = labelled & provenance & reviewed & participant & usable
    result = frame.loc[keep].copy()
    result[features] = numeric.loc[keep]
    return result, {
        "status": "ready" if len(result) else "blocked", "input_rows": len(frame),
        "retained_rows": len(result), "excluded_rows": int((~keep).sum()),
        "unknown_condition_rows": int((~labelled).sum()),
        "unreviewed_rows": int((~reviewed).sum()), "missing_provenance_rows": int((~provenance).sum()),
        "missing_participant_rows": int((~participant).sum()),
        "no_finite_feature_rows": int((~usable).sum()),
        "missing_features_retained": numeric.loc[keep].isna().sum().to_dict(),
    }


def group_splits(participants: list[str] | np.ndarray):
    groups = np.asarray(participants)
    if pd.isna(groups).any() or any(not str(x).strip() for x in groups) or len(set(groups)) < 2:
        raise ValueError("At least two non-missing participant groups are required")
    return list(LeaveOneGroupOut().split(np.zeros(len(groups)), groups=groups))


def grouped_validation(frame: pd.DataFrame, features: list[str], include_temperament: bool = False) -> dict:
    """Fixed logistic baseline, leave-one-participant-out; all fitting inside each fold."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    data, audit = validated_trials(frame, features)
    if data.empty:
        return {"status": "blocked", "audit": audit, "reason": "No verified labelled trials"}
    if data.participant_id.nunique() < 5 or data.condition.nunique() < 2:
        return {"status": "blocked", "audit": audit, "reason": "Require >=5 participants and two classes (pragmatic feasibility gate)"}
    if include_temperament:
        accepted = {"choleric", "sanguine", "phlegmatic", "melancholic"}
        if not {"temperament", "temperament_source"} <= set(data):
            return {"status": "blocked", "reason": "Temperament measurement/provenance absent", "audit": audit}
        if not data.temperament.isin(accepted).all() or data.temperament_source.fillna("").eq("").any():
            return {"status": "blocked", "reason": "Unverified or incompatible temperament taxonomy", "audit": audit}
        if data.groupby("participant_id").temperament.nunique().gt(1).any():
            return {"status": "blocked", "reason": "Conflicting temperament within participant", "audit": audit}
    data = data.reset_index(drop=True)
    y = (data.condition == "lie").astype(int).to_numpy()
    predictions = np.full(len(data), np.nan)
    folds = []
    for train, test in group_splits(data.participant_id.to_numpy()):
        if len(np.unique(y[train])) != 2:
            return {"status": "blocked", "audit": audit, "reason": "A training fold has only one class"}
        numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                            ("scale", StandardScaler())])
        transforms = [("numeric", numeric, features)]
        if include_temperament:
            transforms.append(("temperament", OneHotEncoder(handle_unknown="ignore"), ["temperament"]))
        model = Pipeline([("preprocess", ColumnTransformer(transforms)),
                          ("classifier", LogisticRegression(C=1, class_weight="balanced", max_iter=1000, random_state=42))])
        model.fit(data.iloc[train], y[train])
        predictions[test] = model.predict_proba(data.iloc[test])[:, 1]
        folds.append({"train_participants": sorted(set(data.iloc[train].participant_id)),
                      "test_participants": sorted(set(data.iloc[test].participant_id))})
    prediction = predictions >= 0.5
    # Equal weight per participant, rather than more weight for longer recordings.
    weights = 1 / data.participant_id.map(data.participant_id.value_counts()).to_numpy()
    return {
        "status": "exploratory", "audit": audit, "method": "leave_one_participant_out_logistic_C1",
        "include_temperament": include_temperament, "n_participants": data.participant_id.nunique(),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction, sample_weight=weights)),
        "f1": float(f1_score(y, prediction, sample_weight=weights, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, predictions, sample_weight=weights)),
        "folds": folds, "out_of_fold_probability": predictions.tolist(),
        "row_index": data.index.tolist(), "warning": "Small-sample internal validation; no individual lie verdict or deployment claim",
    }


def temperament_hypotheses(frame: pd.DataFrame, features: list[str], permutations: int = 4999) -> dict:
    """Exploratory participant-level lie-minus-truth contrasts, permutation omnibus + Holm.

    Group-label permutation assumes exchangeable participants under the null; this is
    not a causal test. Repeated windows are averaged within participant/condition.
    """
    from scipy.stats import f_oneway

    data, audit = validated_trials(frame, features)
    needed = {"temperament", "temperament_source"}
    accepted = {"choleric", "sanguine", "phlegmatic", "melancholic"}
    if data.empty or not needed <= set(data):
        return {"status": "blocked", "audit": audit, "reason": "Reviewed conditions and measured classical temperament are required"}
    if not data.temperament.isin(accepted).all() or data.temperament_source.fillna("").eq("").any():
        return {"status": "blocked", "audit": audit, "reason": "Unknown temperament or incompatible source taxonomy"}
    if data.groupby("participant_id").temperament.nunique().gt(1).any():
        return {"status": "blocked", "audit": audit, "reason": "Conflicting temperament within participant"}
    temperament = data.groupby("participant_id").temperament.first()
    results = []
    rng = np.random.default_rng(42)
    for feature in features:
        wide = data.groupby(["participant_id", "condition"])[feature].mean().unstack()
        if not {"truth", "lie"} <= set(wide):
            results.append({"feature": feature, "status": "blocked", "reason": "Both conditions needed"}); continue
        delta = (wide["lie"] - wide["truth"]).dropna()
        labels = temperament.reindex(delta.index)
        counts = labels.value_counts()
        if len(counts) != 4 or (counts < 3).any():
            results.append({"feature": feature, "status": "blocked", "reason": "Require >=3 paired participants in each of four groups",
                            "paired_n_by_group": counts.to_dict()}); continue
        values = delta.to_numpy()
        groups = sorted(counts.index)
        labels = labels.to_numpy()
        arrays = [values[labels == group] for group in groups]
        # Degenerate groups do not support an F statistic.
        if sum(np.sum((a - a.mean()) ** 2) for a in arrays) == 0:
            results.append({"feature": feature, "status": "blocked", "reason": "Zero within-group variation"}); continue
        observed = float(f_oneway(*arrays).statistic)
        null = []
        for _ in range(permutations):
            permuted = rng.permutation(labels)
            null.append(float(f_oneway(*(values[permuted == group] for group in groups)).statistic))
        p = (1 + np.count_nonzero(np.asarray(null) >= observed)) / (permutations + 1)
        means, intervals = {}, {}
        for group, a in zip(groups, arrays):
            means[group] = float(a.mean())
            boot = rng.choice(a, (2000, len(a)), replace=True).mean(axis=1)
            intervals[group] = np.quantile(boot, [0.025, 0.975]).tolist()
        between = sum(len(a) * (a.mean() - values.mean()) ** 2 for a in arrays)
        total = np.sum((values - values.mean()) ** 2)
        results.append({"feature": feature, "status": "exploratory", "F": observed, "p_raw": float(p),
                        "eta_squared": float(between / total), "paired_n_by_group": counts.to_dict(),
                        "mean_lie_minus_truth": means, "bootstrap_95ci": intervals,
                        "missing_paired_participants": int(data.participant_id.nunique() - len(delta))})
    successful = sorted([r for r in results if "p_raw" in r], key=lambda r: r["p_raw"])
    adjusted = 0.
    # Count all planned features, including blocked ones, in the correction family.
    for i, result in enumerate(successful):
        adjusted = max(adjusted, min(1., result["p_raw"] * (len(features) - i)))
        result["p_holm"] = adjusted
    return {"status": "exploratory" if successful else "blocked", "audit": audit,
            "sampling_unit": "participant", "permutations": permutations, "seed": 42, "results": results}
