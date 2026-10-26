"""Audit completed local outputs; no data are sent outside this computer."""
from pathlib import Path
import json
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))
from nti_lab.palette import BLUE, PLOT_STYLE, RED, decimal_comma


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def plot_video_quality(folder, rows):
    """Render saved availability rows without recomputing audit metrics."""
    if not rows:
        return
    with plt.rc_context({**PLOT_STYLE, "font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(9,4.8), layout="constrained")
        y = np.arange(len(rows))
        ax.barh(y-.17,[r["face_coverage"]*100 for r in rows],height=.32,color=BLUE,label="Лицевые ROI доступны")
        ax.barh(y+.17,[r["hr_window_fraction"]*100 for r in rows],height=.32,color=RED,label="ЧСС rPPG прошла автоматический фильтр")
        ax.set(yticks=y,yticklabels=[r["display_id"] for r in rows],
               xlim=(0,105),xlabel="Доля кадров / 30-секундных окон, %")
        ax.invert_yaxis()
        ax.legend(loc="lower left",bbox_to_anchor=(0,1.01),frameon=False,ncol=1)
        ax.grid(axis="x",alpha=1)
        ax.set_axisbelow(True)
        decimal_comma(fig)
        fig.savefig(folder/"video_quality.png",dpi=180)
        fig.savefig(folder/"video_quality.svg")
        plt.close(fig)


def main():
    rows, errors = [], []
    for i, source in enumerate(sorted((OUT / "audio").glob("*.segments.json")), 1):
        audio = load(source)
        rid = audio["recording_id"]
        pid = audio["participant_id"]
        paths = {
            "video": OUT / "video" / f"{rid}.json",
            "sync": OUT / "sync" / f"{rid}.json",
            "bio": OUT / "bio" / pid / "metadata.json",
            "media": OUT / "media" / f"{rid}.mp4",
        }
        missing = [k for k, p in paths.items() if not p.exists()]
        if missing:
            errors.append(f"{rid}: missing {missing}")
            continue
        video, sync, bio = (load(paths[k]) for k in ("video", "sync", "bio"))
        t = np.array(video["times"])
        pulse = np.array([np.nan if x is None else x for x in video["rppg"]])
        if not video.get("source_width") or len(t) != len(pulse) or np.any(np.diff(t) <= 0):
            errors.append(f"{rid}: invalid video timebase or legacy output")
        if abs(video["duration_s"] - audio["duration_s"]) > .3:
            errors.append(f"{rid}: incomplete video or mismatched media duration")
        if sync.get("verified") is not False:
            errors.append(f"{rid}: automatic clock must remain unverified")
        w = video["windows"]
        rows.append({
            "display_id": f"P{i:02d}", "recording_id": rid,
            "video_duration_s": video["duration_s"],
            "audio_duration_s": audio["duration_s"],
            "bio_duration_s": bio["analysis_duration_s"],
            "face_coverage": video["face_coverage"],
            "pulse_finite_fraction": float(np.isfinite(pulse).mean()),
            "rppg_hr_windows": sum(x["rppg_hr_bpm"] is not None for x in w),
            "video_windows": len(w),
            "hr_window_fraction": sum(x["rppg_hr_bpm"] is not None for x in w)/max(len(w),1),
            "sync_status": sync["status"],
            "sync_offset_s": sync.get("suggested_offset_s"),
            "asr_segments": len(audio["segments"]),
        })
    result = {"status": "passed" if rows and not errors else "incomplete",
              "recordings": rows, "errors": errors,
              "interpretation": "Integrity checks and automatic signal availability; not accuracy validation."}
    folder = OUT / "audit"
    folder.mkdir(exist_ok=True)
    (folder / "project.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        with (folder / "video_quality.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plot_video_quality(folder, rows)
    print(json.dumps({"status": result["status"], "recordings": len(rows), "errors": errors}, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
