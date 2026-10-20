"""Process all supplied BioRadio recordings."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from nti_lab.biosignals import analyze_recording, read_session


def aggregate_plot(output, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    video_files = sorted((output.resolve().parents[2] / "1_30").glob("*.MTS"))
    order = [p.stem.split("_", 1)[-1] for p in video_files] or sorted(s["subject_id"] for s in summaries)
    lookup = {s["subject_id"]: s for s in summaries}
    records = []
    for i, subject in enumerate(order):
        if subject in lookup:
            meta = lookup[subject]
            records.append({"participant": f"P{i + 1:02}", "subject_id": subject,
                            "median_hr_bpm": meta["median_hr_bpm"],
                            "usable_windows": meta["usable_ecg_windows"],
                            "total_windows": meta["window_count"],
                            "usable_pct": 100 * meta["usable_ecg_windows"] / meta["window_count"],
                            "rejected_rr_pct": 100 * meta["rr_rejected_fraction"]})
    data = pd.DataFrame(records)
    data.to_csv(output / "aggregate_summary.csv", index=False)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), layout="constrained")
        axes[0].scatter(data["median_hr_bpm"], data["participant"], color="#0072B2", marker="o")
        axes[0].set(xlabel="Median HR in usable windows (bpm)", ylabel="Participant")
        axes[1].barh(data["participant"], data["usable_pct"], color="#0072B2")
        axes[1].set(xlabel="ECG windows passing automatic QC (%)", xlim=(0, 100))
        axes[2].barh(data["participant"], data["rejected_rr_pct"], color="#D55E00")
        axes[2].set(xlabel="Rejected RR intervals (%)", xlim=(0, max(20, data["rejected_rr_pct"].max() * 1.1)))
        for ax in axes:
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=.2)
        fig.suptitle("BioRadio: 8 participants, 120 s windows / 5 s step\nOverlapping windows are not independent; automatic QC requires expert review", fontsize=11)
        fig.savefig(output / "aggregate_summary.png", dpi=180, facecolor="white")
        fig.savefig(output / "aggregate_summary.svg", facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "bio")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in sorted(args.data.glob("*.bcrx")):
        print(f"Analyzing {path.name}", flush=True)
        summaries.append(analyze_recording(path, args.output))
    if not summaries:
        raise SystemExit(f"No .bcrx files in {args.data}")
    result = {"recordings": summaries, "session": read_session(args.data / "Session1.bcsx") if (args.data / "Session1.bcsx").exists() else None}
    (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pd.concat([pd.read_csv(args.output / s["subject_id"] / "metrics.csv") for s in summaries], ignore_index=True).to_csv(args.output / "windows.csv", index=False)
    aggregate_plot(args.output, summaries)
    print(f"Done: {len(summaries)} recordings, {sum(s['window_count'] for s in summaries)} windows", flush=True)


if __name__ == "__main__":
    main()
