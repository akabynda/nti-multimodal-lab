from pathlib import Path
import zipfile
import zlib

import numpy as np
import pytest

from nti_lab.biosignals import analyze_ppg, clean_rr, detect_ppg_peaks, detect_r_peaks, hrv_metrics, read_bcrx, respiratory_metrics, rr_spectrum


def _archive(path, arrays, duration, indices=None):
    definitions = "".join(f"""<CustomSignalDefinition><Signal><Name>{name}</Name>
        <SamplesPerSecond>1000</SamplesPerSecond><Units>V</Units>
        <Configuration><ChannelIndex>{i}</ChannelIndex></Configuration></Signal>
        <Gain>1000</Gain><Offset>0</Offset><Units>mV</Units></CustomSignalDefinition>"""
        for i, name in enumerate(arrays))
    header = f"""<BioCaptureRecordingHeader><Version>2.0</Version>
        <SessionInfo><CustomSignalDefinitions>{definitions}</CustomSignalDefinitions></SessionInfo>
        <RecordingLengthTicks>{round(duration * 1e7)}</RecordingLengthTicks>
        <RecordingStartTime>2026-09-30T12:00:00+03:00</RecordingStartTime><EventList/>
        </BioCaptureRecordingHeader>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("header.xml", header)
        for ch, chunks in enumerate(arrays.values()):
            for i in reversed(range(len(chunks))):
                encoder = zlib.compressobj(wbits=-15)
                raw = np.asarray(chunks[i], dtype="<f4").tobytes()
                archive.writestr(f"{ch}_{indices[i] if indices else i}.rec", encoder.compress(raw) + encoder.flush())


def test_bcrx_numeric_chunk_order_and_nan_preservation(tmp_path):
    values = [np.array([i, np.nan], dtype=float) for i in range(12)]
    path = tmp_path / "subject.bcrx"
    _archive(path, {"ECG": values}, .024)
    rec = read_bcrx(path)
    np.testing.assert_array_equal(rec.channels["ECG"][::2], np.arange(12))
    assert np.isnan(rec.channels["ECG"][1::2]).all()
    assert rec.metadata["channels"][0]["nan_count"] == 12
    assert rec.metadata["recording_start_time"].endswith("+03:00")


def test_bcrx_rejects_missing_chunk_and_wrong_duration(tmp_path):
    path = tmp_path / "subject.bcrx"
    _archive(path, {"ECG": [[1, 2], [3, 4]]}, .004, indices=[0, 2])
    with pytest.raises(ValueError, match="nonconsecutive"):
        read_bcrx(path)
    _archive(path, {"ECG": [[1, 2]]}, 10)
    with pytest.raises(ValueError, match="duration disagree"):
        read_bcrx(path)


def test_r_peak_detection_handles_polarity_baseline_and_missing_gap():
    fs = 250
    t = np.arange(0, 40, 1 / fs)
    expected = np.arange(1, 39, .8)
    x = .0002 * np.sin(2 * np.pi * .2 * t)
    for peak in expected:
        x -= .001 * np.exp(-((t - peak) / .018) ** 2)
    x += np.random.default_rng(2026).normal(0, .00001, len(x))
    x[(t >= 16) & (t < 16.1)] = np.nan
    observed, _ = detect_r_peaks(x, fs)
    actual = observed / fs
    expected = expected[np.abs(expected - 16.05) > .35]
    assert all(np.min(np.abs(actual - p)) < .03 for p in expected)
    assert len(actual) <= len(expected) + 1
    assert not np.any((actual > 15.75) & (actual < 16.35))


def test_rr_rejection_and_successive_differences_do_not_bridge_bad_intervals():
    times = np.arange(0, 10, 1.)
    times[5] += .4
    rr_t, rr, valid = clean_rr(times)
    assert not valid[4] and not valid[5]
    result = hrv_metrics(rr_t, rr, valid, 0, 10)
    assert result["rmssd_ms"] == 0
    assert result["hr_bpm"] == 60
    assert result["ulf_ms2"] is None


def test_hrv_spectral_separation_and_units():
    t = np.arange(.5, 120, .8)
    rr = 800 + 40 * np.sin(2 * np.pi * .1 * t) + 20 * np.sin(2 * np.pi * .25 * t)
    result = hrv_metrics(t, rr, np.ones(len(t), dtype=bool), 0, 120)
    assert result["spectrum_status"] == "exploratory_120s"
    assert 3 < result["lf_hf"] < 7  # interpolation attenuates HF slightly
    assert 600 < result["lf_ms2"] < 900
    assert result["ulf_ms2"] is None
    assert result["total_ms2"] == pytest.approx(result["vlf_ms2"] + result["lf_ms2"] + result["hf_ms2"])


def test_spectrum_is_withheld_for_long_gaps():
    t = np.arange(.5, 120, .8)
    good = (t < 50) | (t > 60)
    result = hrv_metrics(t, np.full(len(t), 800.), good, 0, 120)
    assert result["spectrum_status"] == "insufficient_quality"
    assert result["lf_ms2"] is None


def test_ppg_and_respiration_synthetic_rates():
    fs = 25
    t = np.arange(0, 120, 1 / fs)
    x = np.sin(2 * np.pi * 1.2 * t) + .2 * np.sin(2 * np.pi * .2 * t)
    peaks, _ = detect_ppg_peaks(x, fs)
    assert 70 < 60 / np.median(np.diff(peaks) / fs) < 75
    result = respiratory_metrics(np.sin(2 * np.pi * .25 * t), fs)
    assert result["resp_bpm"] == pytest.approx(15, abs=.5)


def test_ppg_pipeline_and_ui_spectrum():
    fs = 25
    t = np.arange(0, 130, 1 / fs)
    ppg = np.sin(2 * np.pi * 1.2 * t)
    windows, series = analyze_ppg(ppg, fs)
    assert len(windows) == 3
    assert windows["source"].eq("contact_PPG_PRV").all()
    assert windows["hr_bpm"].median() == pytest.approx(72, abs=1)
    psd = rr_spectrum(series["interval_t_s"], series["interval_ms"], series["interval_accepted"], 0, 120)
    assert psd["status"] == "exploratory_120s"
    assert len(psd["frequency_hz"]) == len(psd["power_ms2_hz"]) > 100
    assert "respiration_proxy" in series


_local_recording = next(Path(__file__).resolve().parents[2].glob("*.bcrx"), None)

@pytest.mark.skipif(_local_recording is None, reason="local private data unavailable")
def test_actual_sample_count_matches_header():
    rec = read_bcrx(_local_recording)
    assert set(rec.channels) == {"ECG", "RIP", "GSR"}
    assert abs(len(rec.channels["ECG"]) / 1000 - rec.metadata["header_duration_s"]) < .01
    assert not rec.metadata["contact_ppg_available"]
