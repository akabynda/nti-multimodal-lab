"""Estimate explicitly unverified acquisition-constrained ECG/rPPG alignment."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nti_lab.alignment import estimate_alignment, compare_modalities
from nti_lab.biosignals import _safe


def media_info(source, asr, video):
    result = {"recording_duration_s": max(float(asr["duration_s"]),
                                          float(video["duration_s"])),
              "duration_basis": "maximum of decoded audio duration and last analyzed video timestamp",
              "duration_resolution_uncertainty_s": 1 / float(video["analysis_fps"])}
    if source.exists():
        import av
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            result.update(sample_aspect_ratio=float(stream.sample_aspect_ratio or 1),
                          video_stream_start_pts_s=float(stream.start_time * stream.time_base),
                          video_first_decoded_pts_s=float(next(container.decode(video=0)).time))
        with av.open(str(source)) as container:
            result["audio_first_decoded_pts_s"] = float(next(container.decode(audio=0)).time)
    else:
        result.update(sample_aspect_ratio=4 / 3, media_origin_status="source_unavailable_assumed_equal_decoded_origins",
                      audio_first_decoded_pts_s=0., video_first_decoded_pts_s=0.)
    return result


def plot_alignment(path, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curve = result.get("score_curve")
    if not curve:
        return
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    x = np.asarray(curve["lead_s"])
    ax.plot(x, curve["beginning_abs_correlation"], label="Beginning baseline", color="#0072B2")
    ax.plot(x, curve["ending_abs_correlation"], label="Ending baseline", color="#D55E00", linestyle="--")
    ax.plot(x, curve["joint_score"], label="Joint score", color="#222222", lw=1.3)
    lead = result["bio_lead_s"]
    ax.axvline(lead, label=f"Unverified candidate: {lead:.2f} s", color="#009E73", linestyle=":")
    ax.set(xlabel="BioRadio lead before video (s)", ylabel="Absolute waveform correlation",
           ylim=(0, 1), title=f"{result['participant_id']} | {result['status']}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_aggregate(output, results):
    """Summarize every available recording without implying verified clocks."""
    rows = []
    for result in results:
        baselines = {b["name"]: b for b in result.get("baselines", [])}
        interval = result.get("lead_uncertainty_interval_s") or [None, None]
        row = {
            "recording_id": result["recording_id"],
            "participant_id": result["participant_id"],
            "status": result["status"],
            "verified": result["verified"],
            "bio_lead_s": result.get("bio_lead_s"),
            "bio_to_video_offset_s": result.get("bio_to_video_offset_s"),
            "acquisition_lead_min_s": result["lead_bounds_s"][0],
            "acquisition_lead_max_s": result["lead_bounds_s"][1],
            "lead_sensitivity_min_s": interval[0],
            "lead_sensitivity_max_s": interval[1],
            "scale": result["scale"],
            "joint_score": result.get("joint_support", {}).get("best_score"),
            "joint_margin": result.get("joint_support", {}).get("margin"),
            "baseline_best_lag_difference_s": result.get("baseline_best_lag_difference_s"),
        }
        for name in ("beginning", "ending"):
            baseline = baselines.get(name, {})
            for field in ("start_s", "end_s", "evidence", "eye_state",
                          "low_eye_opening_fraction", "finite_pulse_coverage",
                          "best_lead_s", "best_score", "margin"):
                row[f"{name}_{field}"] = baseline.get(field)
            row[f"{name}_score_at_candidate"] = result.get(f"{name}_score_at_candidate")
            for field, value in baseline.get("ecg_quality", {}).items():
                row[f"{name}_ecg_{field}"] = value
        row.update(result.get("crossmodal_summary", {}))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output / "summary.csv", index=False)
    if not len(table):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = {"supported_unverified": "#0072B2", "ambiguous_unverified": "#D55E00",
              "insufficient_signal": "#777777", "acquisition_order_inconsistent": "#777777"}
    fig, axes = plt.subplots(1, 3, figsize=(14, max(4.2, .48 * len(table) + 1.5)),
                             sharey=True, layout="constrained")
    y = np.arange(len(table))
    labels = [f"P{int(r['recording_id'].split('_')[0]):02d}" for r in rows]
    for i, row in enumerate(rows):
        color = colors[row["status"]]
        axes[0].hlines(i, row["acquisition_lead_min_s"], row["acquisition_lead_max_s"],
                       color="#BBBBBB", linewidth=1.2, zorder=1)
        if row["bio_lead_s"] is not None:
            axes[0].hlines(i, row["lead_sensitivity_min_s"], row["lead_sensitivity_max_s"],
                           color=color, linewidth=4, alpha=.65)
            axes[0].plot(row["bio_lead_s"], i, "o", color=color, markersize=5)
        for name, marker, offset in (("beginning", "o", -.12), ("ending", "s", .12)):
            score = row.get(f"{name}_score_at_candidate")
            if score is not None:
                axes[1].plot(score, i + offset, marker=marker, color=color,
                             markerfacecolor=color if name == "beginning" else "white",
                             markersize=6)
        mae = row.get("hr_mae_bpm")
        n = row.get("hr_paired_windows", 0)
        if mae is not None:
            axes[2].barh(i, mae, color=color, height=.5, alpha=.75)
            axes[2].annotate(f"n={n}", (mae, i), xytext=(4, 0),
                             textcoords="offset points", va="center", fontsize=8)
        else:
            axes[2].text(0, i, "No quality-passing pairs", va="center", fontsize=8)
    axes[0].set(yticks=y, yticklabels=labels, xlabel="BioRadio lead before video (s)",
                title="Candidate and sensitivity bounds", xlim=(0, None))
    axes[0].invert_yaxis()
    axes[1].set(xlabel="Absolute pulse correlation at candidate", title="Independent baseline support",
                xlim=(0, 1))
    axes[1].legend(handles=[
        Line2D([], [], marker="o", color="#333333", linestyle="", label="Beginning"),
        Line2D([], [], marker="s", color="#333333", markerfacecolor="white",
               linestyle="", label="Ending")], loc="lower right", fontsize=8)
    axes[2].set(xlabel="rPPG versus ECG HR MAE (bpm)", title="Exploratory matched-window agreement")
    axes[2].set_xlim(0, max(5., float(table["hr_mae_bpm"].max() or 0) * 1.35))
    for ax in axes:
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("All clock offsets remain unverified; scale = 1", fontsize=13)
    fig.supxlabel("Blue: supported candidate; orange: ambiguous. Gray lines: acquisition bounds. "
                  "Colored intervals: sensitivity bounds, not confidence intervals.\n"
                  "n counts overlapping 30 s windows, not independent observations.", fontsize=9)
    fig.savefig(output / "aggregate_summary.png", dpi=160)
    fig.savefig(output / "aggregate_summary.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--recording", help="Optional exact recording_id")
    args = parser.parse_args()
    output = args.root / "outputs/sync"
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted((args.root / "outputs/video").glob("*.json"))
    for path in paths:
        if args.recording and path.stem != args.recording:
            continue
        try:
            video = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"WAIT video file still being written: {path.stem}", flush=True)
            continue
        # Skip known obsolete full-frame detector products, rather than publishing
        # new sync results from them during a concurrent video rebuild.
        if "source_width" not in video:
            print(f"WAIT updated video: {path.stem}", flush=True)
            continue
        subject = video["participant_id"]
        biodir = args.root / "outputs/bio" / subject
        asrpath = args.root / "outputs/audio" / f"{path.stem}.segments.json"
        if not asrpath.exists() or not (biodir / "signals.npz").exists():
            print(f"WAIT required modalities: {path.stem}", flush=True)
            continue
        bio = dict(np.load(biodir / "signals.npz"))
        metadata = json.loads((biodir / "metadata.json").read_text(encoding="utf-8"))
        asr = json.loads(asrpath.read_text(encoding="utf-8"))
        info = media_info(args.data / "1_30" / video["source_file"], asr, video)
        result = estimate_alignment(video, bio, metadata, asr, info)
        bio_windows = pd.read_csv(biodir / "metrics.csv").to_dict("records")
        rows = compare_modalities(video, bio, result, bio_windows)
        pairs = pd.DataFrame(rows)
        result["crossmodal_summary"] = {
            "interpretation": "descriptive signal agreement under unverified candidate offset",
            "hr_paired_windows": 0, "hr_mae_bpm": None,
            "hr_bias_rppg_minus_ecg_bpm": None, "resp_paired_windows": 0,
            "resp_mae_bpm": None, "overlapping_windows_independent": False,
        }
        if len(pairs):
            pairs.to_csv(output / f"{path.stem}.comparisons.csv", index=False)
            hr = pairs["hr_difference_bpm"].dropna()
            resp = pairs["resp_difference_bpm"].dropna()
            result["crossmodal_summary"] = {
                "interpretation": "descriptive signal agreement under unverified candidate offset",
                "hr_paired_windows": len(hr), "hr_mae_bpm": float(hr.abs().mean()) if len(hr) else None,
                "hr_bias_rppg_minus_ecg_bpm": float(hr.mean()) if len(hr) else None,
                "resp_paired_windows": len(resp), "resp_mae_bpm": float(resp.abs().mean()) if len(resp) else None,
                "overlapping_windows_independent": False,
            }
        # Suggested values prefill a UI; they do not overwrite expert annotations.
        result["suggested_offset_s"] = result.get("bio_to_video_offset_s")
        result["suggested_scale"] = result["scale"]
        interval = result.get("lead_uncertainty_interval_s")
        result["offset_uncertainty_interval_s"] = [-interval[1], -interval[0]] if interval else None
        result["uncertainty_interpretation"] = "Sensitivity interval, not a statistical confidence interval; unmeasured pulse transit can lie outside the assumed range."
        result["requires_confirmation"] = True
        result = _safe(result)
        destination = output / f"{path.stem}.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(destination)
        plot_alignment(output / f"{path.stem}.png", result)
        print(f"{path.stem}: {result['status']}, lead={result.get('bio_lead_s')}, score={result.get('joint_support',{}).get('best_score')}", flush=True)
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(output.glob("000*.json"))]
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps({"recordings": results}, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output / "summary.json")
    tables = [pd.read_csv(p) for p in sorted(output.glob("*.comparisons.csv"))]
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(output / "comparisons.csv", index=False)
    write_aggregate(output, results)


if __name__ == "__main__":
    main()
