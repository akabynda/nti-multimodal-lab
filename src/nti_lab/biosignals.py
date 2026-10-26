"""BioCapture v2 reader and transparent research ECG/PPG/RIP analysis.

The BCRX byte layout is inferred from the supplied files, not a vendor SDK.
No diagnostic interpretation or truth/lie label is produced.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile
import zlib

import numpy as np
import pandas as pd
from scipy import ndimage, signal


@dataclass
class Recording:
    channels: dict[str, np.ndarray]
    sample_rates: dict[str, float]
    metadata: dict


def read_bcrx(path: str | Path) -> Recording:
    """Read raw-deflated little-endian float32 channel chunks in numeric order."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        header = ET.fromstring(archive.read("header.xml"))
        if header.tag != "BioCaptureRecordingHeader" or header.findtext("Version") != "2.0":
            raise ValueError("Only BioCaptureRecordingHeader version 2.0 is supported")
        definitions = header.findall("SessionInfo/CustomSignalDefinitions/CustomSignalDefinition")
        channels, rates, channel_meta = {}, {}, []
        for index, definition in enumerate(definitions):
            name = definition.findtext("Signal/Name")
            fs = float(definition.findtext("Signal/SamplesPerSecond", "0"))
            if not name or name in channels or fs <= 0:
                raise ValueError("Missing/duplicate channel name or invalid sample rate")
            entries = []
            for item in archive.namelist():
                match = re.fullmatch(rf"{index}_(\d+)\.rec", item)
                if match:
                    entries.append((int(match[1]), item))
            entries.sort()
            if not entries or [v[0] for v in entries] != list(range(len(entries))):
                raise ValueError(f"Missing or nonconsecutive chunks in {name}")
            chunks = []
            for _, item in entries:
                decoder = zlib.decompressobj(wbits=-15)
                raw = decoder.decompress(archive.read(item)) + decoder.flush()
                if not decoder.eof or decoder.unused_data or len(raw) % 4:
                    raise ValueError(f"Invalid float32/DEFLATE payload: {item}")
                chunks.append(np.frombuffer(raw, dtype="<f4"))
            samples = np.concatenate(chunks)
            if not len(samples):
                raise ValueError(f"Empty channel {name}")
            channels[name], rates[name] = samples, fs
            channel_meta.append({
                "index": index, "name": name, "sample_rate_hz": fs,
                "sample_count": len(samples), "duration_s": len(samples) / fs,
                "native_unit": definition.findtext("Signal/Units"),
                "display_unit": definition.findtext("Units"),
                "display_gain": float(definition.findtext("Gain", "1")),
                "display_offset": float(definition.findtext("Offset", "0")),
                "device_channel": int(definition.findtext("Signal/Configuration/ChannelIndex", "-1")),
                "nan_count": int(np.isnan(samples).sum()),
                "nonfinite_count": int((~np.isfinite(samples)).sum()),
                "chunks": len(chunks),
                "chunk_sample_counts": [len(c) for c in chunks],
            })
        if not channels:
            raise ValueError("No recorded channels")
        duration = int(header.findtext("RecordingLengthTicks", "0")) / 10_000_000
        if duration <= 0:
            raise ValueError("Missing recording duration")
        for meta in channel_meta:
            if abs(meta["duration_s"] - duration) > 8 / meta["sample_rate_hz"] + 1e-8:
                raise ValueError(f"Sample count and header duration disagree: {meta['name']}")
        metadata = {
            "subject_id": path.stem, "source_file": path.name,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "format_version": "2.0",
            "parser_basis": "inferred raw DEFLATE / little-endian IEEE754 float32",
            "recording_start_time": header.findtext("RecordingStartTime"),
            "header_duration_s": duration,
            "analysis_duration_s": min([duration] + [c["duration_s"] for c in channel_meta]),
            "application_version": header.findtext("ApplicationVersion"),
            "total_packets": int(header.findtext("TotalPackets", "0")),
            "missing_packet_count": int(header.findtext("MissingPacketCount", "0")),
            "events": [ET.tostring(e, encoding="unicode") for e in header.findall("EventList/*")],
            "channels": channel_meta,
            "contact_ppg_available": any("PPG" in n.upper() or "PLETH" in n.upper() for n in channels),
            "sync_status": "absolute BioRadio start available; no video offset established",
        }
    return Recording(channels, rates, metadata)


def read_session(path: str | Path) -> dict:
    """BCSX stores a configuration template, not a synchronized acquisition."""
    root = ET.parse(path).getroot()
    if root.tag != "BioCaptureSession":
        raise ValueError("Expected a BioCaptureSession XML file")
    return {
        "file": Path(path).name, "kind": "configuration_template",
        "channels": [{"name": d.findtext("Signal/Name"),
                      "sample_rate_hz": float(d.findtext("Signal/SamplesPerSecond"))}
                     for d in root.findall("CustomSignalDefinitions/CustomSignalDefinition")],
        "contains_recording_start": root.find("RecordingStartTime") is not None,
        "contains_sync_events": root.find("EventList") is not None,
        "note": "EventMarkerSettings defines available marker keys, not observed events.",
    }


def _interpolate_nonfinite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    good = np.isfinite(x)
    if good.sum() < 3:
        raise ValueError("Fewer than three finite samples")
    return np.interp(np.arange(len(x)), np.flatnonzero(good), x[good])


def _bandpass(x, fs, low, high, order=3):
    return signal.sosfiltfilt(
        signal.butter(order, [low, high], fs=fs, btype="bandpass", output="sos"),
        _interpolate_nonfinite(x),
    )


def detect_r_peaks(ecg: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """QRS-energy detector with 30 s adaptive thresholds; returns indices, ECG.

    Filtering interpolates missing samples solely to avoid propagating NaNs.
    Candidates within 0.25 s of a missing sample are removed.
    """
    if fs < 80 or len(ecg) < fs * 3:
        raise ValueError("ECG needs >=80 Hz and >=3 seconds")
    filtered = _bandpass(ecg, fs, 5, min(25, fs * .4))
    energy = ndimage.uniform_filter1d(filtered ** 2, max(1, round(.06 * fs)))
    threshold = np.zeros(len(energy))
    block = max(1, round(30 * fs))
    for start in range(0, len(energy), block):
        lo, hi = max(0, start - block // 2), min(len(energy), start + 3 * block // 2)
        threshold[start:start + block] = max(np.quantile(energy[lo:hi], .95) * .25, 1e-20)
    candidate, _ = signal.find_peaks(energy, height=threshold, distance=max(1, int(.30 * fs)))
    # The stronger QRS extremum is used consistently; inverted leads are allowed.
    polarity = 1 if np.quantile(filtered, .995) >= -np.quantile(filtered, .005) else -1
    refined = []
    for peak in candidate:
        lo, hi = max(0, peak - round(.09 * fs)), min(len(ecg), peak + round(.09 * fs) + 1)
        refined.append(lo + int(np.argmax(polarity * filtered[lo:hi])))
    peaks = np.unique(refined).astype(int)
    invalid = ndimage.maximum_filter1d((~np.isfinite(ecg)).astype(np.uint8),
                                      2 * round(.25 * fs) + 1)
    peaks = peaks[(peaks > .3 * fs) & (peaks < len(ecg) - .3 * fs) & (invalid[peaks] == 0)]
    return peaks, filtered


def detect_ppg_peaks(ppg: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Generic contact-PPG pulse peaks; unavailable channels are never invented."""
    if fs < 10 or len(ppg) < fs * 3:
        raise ValueError("PPG needs >=10 Hz and >=3 seconds")
    filtered = _bandpass(ppg, fs, .5, min(5, .4 * fs))
    local_scale = ndimage.maximum_filter1d(np.abs(filtered), max(3, round(fs * 2)))
    peaks, _ = signal.find_peaks(filtered, distance=max(1, round(.3 * fs)),
                                prominence=max(float(np.median(local_scale)) * .25, 1e-12))
    invalid = ndimage.maximum_filter1d((~np.isfinite(ppg)).astype(np.uint8), 2 * round(.3 * fs) + 1)
    return peaks[invalid[peaks] == 0], filtered


def analyze_ppg(ppg: np.ndarray, fs: float, window_s=120, step_s=5) -> tuple[pd.DataFrame, dict]:
    """Analyze an explicitly supplied contact PPG; intervals describe PRV, not ECG HRV.

    Also exposes pulse-amplitude respiratory modulation as an unvalidated proxy.
    """
    if window_s != 120 or step_s != 5:
        raise ValueError("This lab protocol requires 120 s windows and 5 s steps")
    peaks, filtered = detect_ppg_peaks(ppg, fs)
    peak_times = peaks / fs
    rr_t, rr_ms, accepted = clean_rr(peak_times)
    rows = []
    for start in np.arange(0, len(ppg) / fs - window_s + 1e-8, step_s):
        row = {"window_start_s": start, "window_end_s": start + window_s,
               "center_s": start + window_s / 2, "source": "contact_PPG_PRV"}
        row.update(hrv_metrics(rr_t, rr_ms, accepted, start, start + window_s))
        segment = np.asarray(ppg)[int(start * fs):int((start + window_s) * fs)]
        finite = np.isfinite(segment)
        span = float(np.ptp(segment[finite])) if finite.any() else 0.
        gap = row["max_interpolation_gap_s"]
        row.update(ppg_nonfinite_fraction=float((~finite).mean()),
                   ppg_quality="usable_automatic" if row["rr_rejected_fraction"] <= .1
                   and row["rr_coverage_fraction"] >= .9 and gap is not None and gap <= 3
                   and finite.mean() >= .99 and span > 1e-9 else "review_required")
        rows.append(row)
    series = {"pulse_peaks_s": peak_times, "interval_t_s": rr_t,
              "interval_ms": rr_ms, "interval_accepted": accepted, "filtered": filtered}
    if len(peaks) >= 4 and peak_times[-1] - peak_times[0] > 10:
        grid = np.arange(peak_times[0], peak_times[-1], .25)
        amplitude = np.interp(grid, peak_times, filtered[peaks])
        respiration_proxy = _bandpass(amplitude, 4, .08, .7)
        for before, after in zip(peak_times[:-1], peak_times[1:]):
            if after - before > 3:
                respiration_proxy[(grid >= before) & (grid <= after)] = np.nan
        series.update(respiration_proxy_t_s=grid,
                      respiration_proxy=respiration_proxy)
    return pd.DataFrame(rows), series


def clean_rr(peaks_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reject RR outside 300–2000 ms or >20% from 11-interval local median.

    This flags artifacts, not ectopic diagnoses. RR timestamps are midpoints.
    """
    peaks_s = np.asarray(peaks_s, dtype=float)
    if len(peaks_s) < 2 or np.any(np.diff(peaks_s) <= 0):
        if len(peaks_s) < 2:
            return np.array([]), np.array([]), np.array([], dtype=bool)
        raise ValueError("Peak times must be strictly increasing")
    rr = np.diff(peaks_s) * 1000
    local = ndimage.median_filter(rr, size=11, mode="nearest")
    valid = (rr >= 300) & (rr <= 2000) & (np.abs(rr - local) <= .20 * local)
    return (peaks_s[:-1] + peaks_s[1:]) / 2, rr, valid


def rr_spectrum(rr_t_s, rr_ms, valid, start_s, end_s) -> dict:
    """Same quality-gated PSD used by hrv_metrics; ready for an API/UI plot.

    Returns lists `frequency_hz`, `power_ms2_hz` and a string `status`.
    """
    rr_t_s, rr_ms, valid = np.asarray(rr_t_s), np.asarray(rr_ms), np.asarray(valid, dtype=bool)
    selected = (rr_t_s >= start_s) & (rr_t_s < end_s)
    good = selected & valid
    times, nn = rr_t_s[good], rr_ms[good]
    result = {"frequency_hz": [], "power_ms2_hz": [], "status": "insufficient_quality"}
    if len(nn) < 30 or not selected.any():
        return result
    if ((times[-1] - times[0]) / (end_s - start_s) < .9
            or 1 - good.sum() / selected.sum() > .2 or np.diff(times).max() > 5):
        return result
    grid = np.arange(times[0], times[-1], .25)
    tachogram = np.interp(grid, times, nn)
    frequencies, psd = signal.welch(tachogram, fs=4, nperseg=len(tachogram),
                                    detrend="linear", window="hann", scaling="density")
    result.update(frequency_hz=frequencies.tolist(), power_ms2_hz=psd.tolist(),
                  status="exploratory_120s")
    return result


def stress_index(nn_ms: np.ndarray, bin_ms: float = 50) -> float | None:
    """Baevsky stress index AMo / (2 × Mo × MxDMn) on accepted RR intervals.

    Mo — centre of the modal 50 ms histogram bin, s; AMo — share of intervals in
    that bin, %; MxDMn — RR range, s. Requires at least 30 intervals and a range > 0.
    """
    nn = np.asarray(nn_ms, dtype=float)
    if len(nn) < 30 or np.ptp(nn) <= 0:
        return None
    bins = np.floor(nn / bin_ms).astype(int)
    values, counts = np.unique(bins, return_counts=True)
    modal = values[np.argmax(counts)]
    mode_s = (modal + .5) * bin_ms / 1000
    amplitude = 100 * counts.max() / len(nn)
    return float(amplitude / (2 * mode_s * np.ptp(nn) / 1000))


def hrv_metrics(rr_t_s: np.ndarray, rr_ms: np.ndarray, valid: np.ndarray,
                start_s: float, end_s: float) -> dict:
    """Time metrics on accepted intervals; PSD on 4 Hz interpolated tachogram."""
    selected = (rr_t_s >= start_s) & (rr_t_s < end_s)
    good = selected & valid
    nn = rr_ms[good]
    count = int(selected.sum())
    result = {
        "rr_count": count, "rr_accepted_count": len(nn),
        "rr_rejected_fraction": float(1 - len(nn) / count) if count else 1.,
        "hr_bpm": None, "mean_rr_ms": None, "sdnn_ms": None, "rmssd_ms": None,
        "pnn50_pct": None, "cv_rr_pct": None, "lf_ms2": None, "hf_ms2": None,
        "lf_hf": None, "total_ms2": None, "vlf_ms2": None, "ulf_ms2": None,
        "lf_nu": None, "hf_nu": None, "stress_index": None, "spectrum_status": "insufficient_data",
        "rr_coverage_fraction": 0., "max_interpolation_gap_s": None,
    }
    if len(nn) < 4:
        return result
    result.update(hr_bpm=float(60000 / nn.mean()), mean_rr_ms=float(nn.mean()),
                  sdnn_ms=float(nn.std(ddof=1)), cv_rr_pct=float(100 * nn.std(ddof=1) / nn.mean()),
                  stress_index=stress_index(nn))
    # Do not bridge a rejected interval when calculating successive differences.
    adjacency = good[:-1] & good[1:]
    differences = np.diff(rr_ms)[adjacency]
    if len(differences):
        result.update(rmssd_ms=float(np.sqrt(np.mean(differences ** 2))),
                      pnn50_pct=float(np.mean(np.abs(differences) > 50) * 100))
    times = rr_t_s[good]
    coverage = float((times[-1] - times[0]) / (end_s - start_s))
    gap = float(np.diff(times).max())
    result.update(rr_coverage_fraction=coverage, max_interpolation_gap_s=gap)
    if len(nn) < 30 or coverage < .9 or result["rr_rejected_fraction"] > .2 or gap > 5:
        result["spectrum_status"] = "insufficient_quality"
        return result
    # One full-window Hann segment retains ~1/120 Hz resolution. No zero padding.
    spectrum = rr_spectrum(rr_t_s, rr_ms, valid, start_s, end_s)
    frequencies, psd = np.asarray(spectrum["frequency_hz"]), np.asarray(spectrum["power_ms2_hz"])
    def band(lo, hi):
        f = np.r_[lo, frequencies[(frequencies > lo) & (frequencies < hi)], hi]
        return float(np.trapezoid(np.interp(f, frequencies, psd), f))
    lf, hf, vlf = band(.04, .15), band(.15, .4), band(.0033, .04)
    result.update(lf_ms2=lf, hf_ms2=hf, vlf_ms2=vlf, total_ms2=vlf + lf + hf,
                  lf_hf=lf / hf if hf > 1e-12 else None,
                  lf_nu=100 * lf / (lf + hf) if lf + hf > 1e-12 else None,
                  hf_nu=100 * hf / (lf + hf) if lf + hf > 1e-12 else None,
                  spectrum_status="exploratory_120s",
                  psd_bin_width_hz=float(frequencies[1] - frequencies[0]))
    return result


def respiratory_metrics(x: np.ndarray, fs: float) -> dict:
    """Dominant RIP frequency with explicit periodicity quality checks."""
    result = {"resp_bpm": None, "resp_candidate_bpm": None,
              "resp_spectral_concentration": None, "resp_autocorrelation": None,
              "resp_status": "insufficient_data"}
    if len(x) < 30 * fs or np.isfinite(x).mean() < .98:
        return result
    filtered = _bandpass(x, fs, .08, .7)
    f, p = signal.welch(filtered, fs=fs, nperseg=len(filtered), detrend="linear")
    mask = (f >= .08) & (f <= .7)
    if not np.any(mask) or p[mask].sum() <= 1e-20:
        return result
    frequency = f[mask][np.argmax(p[mask])]
    concentration = float(p[mask & (np.abs(f - frequency) <= .04)].sum() / p[mask].sum())
    lag = max(1, round(fs / frequency))
    ac = float(np.corrcoef(filtered[:-lag], filtered[lag:])[0, 1])
    accepted = concentration >= .35 and ac >= .25
    result.update(resp_candidate_bpm=float(60 * frequency),
                  resp_bpm=float(60 * frequency) if accepted else None,
                  resp_spectral_concentration=concentration, resp_autocorrelation=ac,
                  resp_status="RIP_periodic_candidate" if accepted else "RIP_low_periodicity")
    return result


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(_safe(value), ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")), encoding="utf-8")


def _contact_ppg_results(recording, duration, ecg_frame):
    """Contact PRV and ECG comparison use the same recorded BioRadio clock."""
    name = next((n for n in recording.channels if "PPG" in n.upper() or "PLETH" in n.upper()), None)
    if name is None:
        return None
    fs = recording.sample_rates[name]
    raw = recording.channels[name][:round(duration * fs)]
    ppg_frame, series = analyze_ppg(raw, fs)
    t = np.arange(len(raw)) / fs
    arrays = {"ppg_t_s": t, "ppg_raw": raw, "ppg_filtered": series["filtered"],
              "ppg_peaks_s": series["pulse_peaks_s"], "ppg_interval_t_s": series["interval_t_s"],
              "ppg_interval_ms": series["interval_ms"], "ppg_interval_valid": series["interval_accepted"]}
    step = max(1, round(fs / 25))
    channel = next(c for c in recording.metadata["channels"] if c["name"] == name)
    overview = {"available": True, "source_channel": name,
                "native_unit": channel["native_unit"] or "a.u.",
                "t_s": t[::step], "raw": raw[::step], "filtered": series["filtered"][::step],
                "fs_hz": fs / step, "pulse_peaks_s": series["pulse_peaks_s"],
                "intervals": {"t_s": series["interval_t_s"], "ms": series["interval_ms"],
                              "accepted": series["interval_accepted"]}}
    if "respiration_proxy" in series:
        arrays.update(ppg_respiration_t_s=series["respiration_proxy_t_s"],
                      ppg_respiration_proxy=series["respiration_proxy"])
        overview["respiration_proxy"] = {"t_s": series["respiration_proxy_t_s"],
                                         "value": series["respiration_proxy"], "kind": "pulse amplitude modulation"}
    measures = ["hr_bpm", "rmssd_ms", "sdnn_ms", "lf_ms2", "hf_ms2", "lf_hf"]
    ecg = ecg_frame[["window_start_s", "window_end_s", "center_s", "ecg_quality"] + measures].rename(columns={k: f"ecg_{k}" for k in measures})
    remote = ppg_frame[["window_start_s", "ppg_quality"] + measures].rename(columns={k: f"ppg_{k}" for k in measures})
    comparison = ecg.merge(remote, on="window_start_s", how="inner", validate="one_to_one")
    comparison["paired_usable"] = comparison.ecg_quality.eq("usable_automatic") & comparison.ppg_quality.eq("usable_automatic")
    comparison["clock_basis"] = "shared_BioRadio_samples"
    for measure in measures:
        difference = comparison[f"ppg_{measure}"] - comparison[f"ecg_{measure}"]
        comparison[f"ppg_minus_ecg_{measure}"] = difference.where(comparison.paired_usable)
    good = ppg_frame.ppg_quality.eq("usable_automatic")
    changes = np.flatnonzero(np.diff(raw) != 0) + 1
    edges = np.r_[0, changes, len(raw)]
    finite_runs = np.isfinite(raw[edges[:-1]])
    analysis = {"source_channel": name, "sample_rate_hz": fs,
                "consecutive_value_changes_hz": float(np.count_nonzero(np.isfinite(raw[1:]) & np.isfinite(raw[:-1]) & (raw[1:] != raw[:-1])) / duration),
                "median_hold_samples": float(np.median(np.diff(edges)[finite_runs])) if finite_runs.any() else None,
                "sample_rate_note": "Stored grid; pulse-oximeter update timestamps are not separately recorded.",
                "pulse_peak_count": len(series["pulse_peaks_s"]),
                "interval_count": len(series["interval_ms"]),
                "interval_rejected_fraction": float(1 - series["interval_accepted"].mean()) if len(series["interval_accepted"]) else None,
                "window_count": len(ppg_frame), "usable_ppg_windows": int(good.sum()),
                "spectral_windows": int(ppg_frame.spectrum_status.eq("exploratory_120s").sum()),
                "median_pulse_rate_bpm": float(ppg_frame.loc[good, "hr_bpm"].median())}
    summary = {"clock_basis": "shared_BioRadio_samples", "paired_windows": int(comparison.paired_usable.sum()),
               "hr_mae_bpm": float(comparison.ppg_minus_ecg_hr_bpm.abs().mean()),
               "hr_bias_ppg_minus_ecg_bpm": float(comparison.ppg_minus_ecg_hr_bpm.mean()),
               "rmssd_mae_ms": float(comparison.ppg_minus_ecg_rmssd_ms.abs().mean()),
               "sdnn_mae_ms": float(comparison.ppg_minus_ecg_sdnn_ms.abs().mean()),
               "window_s": 120, "step_s": 5, "interval_kind": "PPG PRV versus ECG HRV"}
    return arrays, overview, analysis, ppg_frame, comparison, summary


def analyze_recording(path: str | Path, output_dir: str | Path,
                      window_s: float = 120, step_s: float = 5,
                      subject_id: str | None = None) -> dict:
    """Analyze one recording and persist metrics, traces, peak intervals and QC."""
    if window_s != 120 or step_s != 5:
        raise ValueError("This lab protocol requires 120 s windows and 5 s steps")
    recording = read_bcrx(path)
    meta = recording.metadata
    if subject_id is not None:
        if not subject_id or subject_id in (".", "..") or any(c in subject_id for c in '/\\:'):
            raise ValueError("subject_id must be a single directory name")
        meta["subject_id"] = subject_id
    meta["source_path"] = str(Path(path).resolve())
    out = Path(output_dir) / meta["subject_id"]
    out.mkdir(parents=True, exist_ok=True)
    duration = meta["analysis_duration_s"]
    ecg = recording.channels["ECG"]
    fs = recording.sample_rates["ECG"]
    ecg = ecg[:int(round(duration * fs))]
    peaks, filtered = detect_r_peaks(ecg, fs)
    peak_times = peaks / fs
    rr_t, rr_ms, rr_valid = clean_rr(peak_times)
    rip_raw, rip, rip_fs = None, None, 25.
    if "RIP" in recording.channels:
        rip_raw = recording.channels["RIP"][:int(round(duration * recording.sample_rates["RIP"]))]
        rip = signal.resample_poly(_interpolate_nonfinite(rip_raw), 1, round(recording.sample_rates["RIP"] / rip_fs))
        original_missing = ~np.isfinite(rip_raw)
        if original_missing.any():
            invalid = signal.resample_poly(original_missing.astype(float), 1, round(recording.sample_rates["RIP"] / rip_fs)) > .001
            rip[invalid] = np.nan
    rows = []
    for start in np.arange(0, duration - window_s + 1e-8, step_s):
        end = start + window_s
        row = {"subject_id": meta["subject_id"], "window_start_s": start,
               "window_end_s": end, "center_s": (start + end) / 2,
               "source": "ECG"}
        row.update(hrv_metrics(rr_t, rr_ms, rr_valid, start, end))
        segment = ecg[int(start * fs):int(end * fs)]
        filtered_segment = filtered[int(start * fs):int(end * fs)]
        amplitude = float(np.quantile(filtered_segment, .995) - np.quantile(filtered_segment, .005)) * 1000
        row.update(ecg_nonfinite_fraction=float((~np.isfinite(segment)).mean()),
                   ecg_qrs_span_mv=amplitude,
                   ecg_clipping_fraction=float(np.mean(np.abs(segment) >= .18749)),
                   ecg_quality="usable_automatic" if row["rr_rejected_fraction"] <= .1 and amplitude >= .05
                   and np.isfinite(segment).mean() >= .99
                   and row["rr_coverage_fraction"] >= .9
                   and row["max_interpolation_gap_s"] is not None
                   and row["max_interpolation_gap_s"] <= 3
                   and np.mean(np.abs(segment) >= .18749) < .001 else "review_required")
        if rip is not None:
            row.update(respiratory_metrics(rip[int(start * rip_fs):int(end * rip_fs)], rip_fs))
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame({"time_s": rr_t, "rr_ms": rr_ms, "accepted": rr_valid}).to_csv(out / "rr_intervals.csv", index=False)
    # At 250 Hz ECG retains morphology; UI overview uses min/max envelopes below.
    ecg_ds = signal.resample_poly(_interpolate_nonfinite(ecg), 1, round(fs / 250)) * 1000
    ecg_filtered_ds = signal.resample_poly(filtered, 1, round(fs / 250)) * 1000
    ecg_t = np.arange(len(ecg_ds)) / 250
    if np.any(~np.isfinite(ecg)):
        invalid = signal.resample_poly((~np.isfinite(ecg)).astype(float), 1, round(fs / 250)) > .001
        ecg_ds[invalid] = np.nan
        ecg_filtered_ds[invalid] = np.nan
    arrays = {"ecg_t_s": ecg_t, "ecg_mv": ecg_ds, "ecg_filtered_mv": ecg_filtered_ds,
              "r_peaks_s": peak_times, "rr_t_s": rr_t, "rr_ms": rr_ms, "rr_valid": rr_valid}
    if "GSR" in recording.channels:
        gsr_fs = recording.sample_rates["GSR"]
        gsr = recording.channels["GSR"][:int(round(duration * gsr_fs))]
        step = max(1, round(gsr_fs / 5))
        arrays.update(gsr_t_s=np.arange(0, len(gsr), step) / gsr_fs, gsr_us=gsr[::step])
        finite = gsr[np.isfinite(gsr)]
        mode_fraction = float(np.max(np.unique(finite, return_counts=True)[1]) / len(finite)) if len(finite) else 1.
        meta["gsr_quality"] = {
            "most_common_value_fraction": mode_fraction,
            "outside_declared_range_fraction": float(np.mean((gsr < 0) | (gsr > 24000))),
            "status": "flat_or_out_of_range_review_required" if mode_fraction > .95 or np.any(gsr > 24000) else "not_validated",
            "note": "Raw 5 Hz samples retained; no electrodermal stress inference.",
        }
    if rip is not None:
        arrays.update(rip_t_s=np.arange(len(rip)) / rip_fs, rip_mv=rip * 1000)
        respiration = _bandpass(rip, rip_fs, .08, .7) * 1000
        respiration[~np.isfinite(rip)] = np.nan
        arrays["respiration_mv"] = respiration
    # ECG-derived respiration is an amplitude proxy, not measured breathing.
    if len(peaks) >= 4:
        edr_t = np.arange(peak_times[0], peak_times[-1], .25)
        edr = np.interp(edr_t, peak_times, filtered[peaks])
        edr_filtered = _bandpass(edr, 4, .08, .7) * 1000
        for before, after in zip(peak_times[:-1], peak_times[1:]):
            if after - before > 3:
                edr_filtered[(edr_t >= before) & (edr_t <= after)] = np.nan
        arrays.update(edr_t_s=edr_t, edr_mv=edr_filtered)
    ppg = _contact_ppg_results(recording, duration, frame)
    if ppg is not None:
        ppg_arrays, ppg_overview, ppg_analysis, ppg_frame, comparison, ppg_comparison = ppg
        arrays.update(ppg_arrays)
        meta.update(ppg_analysis=ppg_analysis, ecg_ppg_comparison=ppg_comparison)
        ppg_frame.to_csv(out / "ppg_metrics.csv", index=False)
        comparison.insert(0, "subject_id", meta["subject_id"])
        comparison.to_csv(out / "ecg_ppg_comparison.csv", index=False)
        pd.DataFrame({"time_s": ppg_arrays["ppg_interval_t_s"], "interval_ms": ppg_arrays["ppg_interval_ms"],
                      "accepted": ppg_arrays["ppg_interval_valid"]}).to_csv(out / "ppg_intervals.csv", index=False)
    np.savez_compressed(out / "signals.npz", **arrays)
    # ECG min/max per 0.1 s keeps narrow QRS features visible in a compact overview.
    bin_size = max(1, round(fs / 10))
    size = len(ecg) // bin_size
    bins = ecg[:size * bin_size].reshape(size, bin_size) * 1000
    overview = {
        "subject_id": meta["subject_id"], "duration_s": duration,
        "ecg": {"t_s": np.arange(size) / 10, "min_mv": np.min(bins, axis=1),
                "max_mv": np.max(bins, axis=1), "representation": "0.1 s min/max envelope"},
        "r_peaks_s": peak_times, "rr": {"t_s": rr_t, "ms": rr_ms, "accepted": rr_valid},
        "ppg": ppg_overview if ppg is not None else {"available": False, "reason": "No contact PPG channel in source recording"},
    }
    if rip is not None:
        overview["rip"] = {"t_s": arrays["rip_t_s"][::5], "mv": arrays["rip_mv"][::5],
                           "respiration_mv": arrays["respiration_mv"][::5], "fs_hz": 5}
    if "edr_t_s" in arrays:
        overview["edr"] = {"t_s": arrays["edr_t_s"], "mv": arrays["edr_mv"],
                           "kind": "ECG amplitude respiration proxy"}
    if "gsr_t_s" in arrays:
        overview["gsr"] = {"t_s": arrays["gsr_t_s"], "us": arrays["gsr_us"],
                           "fs_hz": 5, "quality": meta["gsr_quality"]}
    _write_json(out / "timeseries.json", overview)
    meta.update({
        "r_peak_count": len(peaks), "rr_count": len(rr_ms),
        "rr_rejected_fraction": float(1 - rr_valid.mean()) if len(rr_valid) else None,
        "window_s": window_s, "step_s": step_s, "window_count": len(rows),
        "usable_ecg_windows": int((frame["ecg_quality"] == "usable_automatic").sum()),
        "spectral_windows": int((frame["spectrum_status"] == "exploratory_120s").sum()),
        "median_hr_bpm": float(frame.loc[frame["ecg_quality"] == "usable_automatic", "hr_bpm"].median()),
        "median_hr_all_windows_bpm": float(frame["hr_bpm"].median()),
        "respiration_source": "RIP belt; dominant-frequency candidates require review",
        "contact_ppg_status": "recorded_processed" if ppg is not None else "not_recorded",
        "limitations": [
            "Automatic R peaks and artifact flags are not expert-annotated ground truth.",
            "ULF is not estimable in 120 seconds; it is null, never zero.",
            "VLF and total power are exploratory at 120 s; standard 5 min HRV differs.",
            "LF/HF is not a validated stress score or lie detector.",
            "RIP frequencies are periodicity candidates, not manually counted breaths.",
            "EDR is an ECG-amplitude proxy and is not reference respiration.",
            "No BioRadio/video offset is inferred from a file name.",
            "Overlapping 120 s windows at 5 s steps are not independent observations.",
        ],
    })
    _write_json(out / "metadata.json", meta)
    _plot_qc(out, meta["subject_id"], arrays, frame)
    if ppg is not None:
        _plot_contact_ppg(out, arrays, comparison)
    return _safe(meta)


def _plot_contact_ppg(out, arrays, comparison):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .palette import BLUE, MUTED, PLOT_STYLE, VIOLET, decimal_comma
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), layout="constrained")
        use = (arrays["ppg_t_s"] >= 30) & (arrays["ppg_t_s"] < 40)
        axes[0].plot(arrays["ppg_t_s"][use], arrays["ppg_filtered"][use], color=VIOLET, label="Контактный ФПГ")
        r_times = arrays["r_peaks_s"][(arrays["r_peaks_s"] >= 30) & (arrays["r_peaks_s"] < 40)]
        for index, peak in enumerate(r_times):
            axes[0].axvline(peak, color=BLUE, lw=.7, label="R-зубцы ЭКГ" if index == 0 else None)
        axes[0].set(xlabel="Время БиоРадио, с", ylabel="ФПГ, усл. ед.",
                    title="Контактный ФПГ и R-зубцы ЭКГ на общей шкале БиоРадио")
        low, high = axes[0].get_ylim()
        axes[0].set_ylim(low, high + .3 * (high - low))
        axes[0].legend(loc="upper right", fontsize=9)
        good = comparison.paired_usable
        axes[1].plot(comparison.center_s, comparison.ecg_hr_bpm, label="ЭКГ", color=BLUE, linestyle="-")
        axes[1].plot(comparison.center_s, comparison.ppg_hr_bpm, label="Контактный ФПГ", color=VIOLET, linestyle="--", alpha=.7)
        axes[1].scatter(comparison.loc[~good, "center_s"], comparison.loc[~good, "ppg_hr_bpm"],
                        marker="x", color=MUTED, s=12, label="Окна, исключённые из сравнения")
        axes[1].set(xlabel="Центр окна, с", ylabel="ЧСС / пульс, уд/мин")
        axes[1].legend(fontsize=8, loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=3)
        axes[2].scatter(comparison.loc[good, "center_s"], comparison.loc[good, "ppg_minus_ecg_hr_bpm"],
                        color=VIOLET, s=12)
        axes[2].axhline(0, color=MUTED, lw=.7)
        axes[2].set(xlabel="Центр окна, с", ylabel="ФПГ − ЭКГ, уд/мин",
                    title=f"Прошедшие контроль окна: {int(good.sum())}; длительность 120 с, шаг 5 с")
        for ax in axes:
            ax.grid(alpha=1)
        decimal_comma(fig)
        fig.savefig(out / "ecg_ppg_comparison.png", dpi=160)
        plt.close(fig)


def _plot_qc(out, subject, arrays, frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), layout="constrained")
        t = arrays["ecg_t_s"]
        selection = (t >= 30) & (t < 40)
        axes[0].plot(t[selection], arrays["ecg_filtered_mv"][selection], color="#0072B2", lw=.8)
        peaks = arrays["r_peaks_s"]
        peaks = peaks[(peaks >= 30) & (peaks < 40)]
        axes[0].scatter(peaks, np.interp(peaks, t, arrays["ecg_filtered_mv"]), s=18, marker="x", color="#D55E00", label="Detected QRS")
        axes[0].set(ylabel="ECG 5–25 Hz (mV)", title=f"{subject} | automated signal quality audit")
        axes[0].legend()
        good = arrays["rr_valid"]
        axes[1].scatter(arrays["rr_t_s"][good], arrays["rr_ms"][good], s=2, color="#0072B2", label="Accepted RR")
        axes[1].scatter(arrays["rr_t_s"][~good], arrays["rr_ms"][~good], s=8, marker="x", color="#D55E00", label="Rejected RR")
        axes[1].set(ylabel="RR (ms)")
        axes[1].legend()
        axes[2].plot(frame["center_s"], frame["hr_bpm"], color="#0072B2", label="120 s mean HR")
        bad = frame["ecg_quality"] != "usable_automatic"
        axes[2].scatter(frame.loc[bad, "center_s"], frame.loc[bad, "hr_bpm"], marker="x", s=10, color="#D55E00", label="Review required")
        axes[2].set(ylabel="Heart rate (bpm)")
        axes[2].legend()
        if "resp_bpm" in frame:
            axes[3].plot(frame["center_s"], frame["resp_candidate_bpm"], color="#777777", lw=.7, label="All RIP frequency candidates")
            axes[3].scatter(frame["center_s"], frame["resp_bpm"], color="#009E73", marker=".", s=12, label="Passed periodicity checks")
            axes[3].set(ylabel="RIP frequency (breaths/min)")
            axes[3].legend()
        for ax in axes:
            ax.set_xlabel("Time since BioRadio recording start (s)")
            ax.grid(alpha=.2)
        fig.savefig(out / "quality_audit.png", dpi=160, facecolor="white")
        plt.close(fig)
