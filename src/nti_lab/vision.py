"""Face ROI POS-rPPG and visible-body kinematics, without emotion inference."""
from __future__ import annotations

import json
import hashlib
import urllib.request
from pathlib import Path
import numpy as np
from scipy import signal

MODEL_URLS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}
ROIS = {
    "forehead": [10, 109, 108, 9, 337, 338],
    "left_cheek": [50, 101, 205, 206, 203, 118],
    "right_cheek": [280, 330, 425, 426, 423, 347],
    "mouth": [61, 0, 291, 17],
    "eyes": [33, 159, 133, 145, 362, 386, 263, 374],
}
POSE_EDGES = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),
              (12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]


def models(folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, url in MODEL_URLS.items():
        p = folder / name
        if not p.exists():
            urllib.request.urlretrieve(url, p)
        result[name] = p
    return result


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def pos_rppg(rgb: np.ndarray, fs: float) -> np.ndarray:
    """Wang et al. POS, overlap-add 1.6-s windows; complete finite RGB required."""
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("RGB must be Nx3")
    if not np.isfinite(rgb).all() or fs <= 8:
        raise ValueError("POS requires finite RGB and sampling frequency >8 Hz")
    n = len(rgb)
    length = max(8, int(round(1.6 * fs)))
    if n < length + 12:
        return np.full(n, np.nan)
    pulse, count = np.zeros(n), np.zeros(n)
    projection = np.array([[0, 1, -1], [-2, 1, 1]])
    for end in range(length, n + 1):
        c = rgb[end-length:end].T
        mean = c.mean(axis=1, keepdims=True)
        if np.any(mean < 1):
            continue
        s = projection @ (c / mean - 1)
        h = s[0] + (s[0].std() / max(s[1].std(), 1e-9)) * s[1]
        pulse[end-length:end] += h - h.mean()
        count[end-length:end] += 1
    pulse /= np.maximum(count, 1)
    sos = signal.butter(3, [0.7, 3.5], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, pulse)


def dominant_rate(x, fs, band=(0.7, 3.5)):
    x = np.asarray(x, float)
    if len(x) < fs * 15 or not np.isfinite(x).all() or np.std(x) < 1e-9:
        return None, None
    f, p = signal.welch(signal.detrend(x), fs, nperseg=min(len(x), int(30*fs)))
    m = (f >= band[0]) & (f <= band[1])
    if not np.any(m) or np.sum(p[m]) == 0:
        return None, None
    peak = float(f[m][np.argmax(p[m])])
    power = np.sum(p[(abs(f-peak) < 0.12) & m]) / np.sum(p[m])
    return peak * 60, float(power)


def _runs(mask):
    padded = np.r_[False, mask, False].astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return zip(edges[::2], edges[1::2])


def process_video(path: Path, out_dir: Path, model_dir: Path, sample_fps=12.5, max_seconds=None):
    import av
    import cv2
    import mediapipe as mp
    cv2.setNumThreads(1)
    assets = models(model_dir)
    base = mp.tasks.BaseOptions
    vision = mp.tasks.vision
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=base(model_asset_path=str(assets["face_landmarker.task"])),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
        output_facial_transformation_matrixes=True))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=base(model_asset_path=str(assets["pose_landmarker_lite.task"])),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))
    container = av.open(str(path))
    stream = container.streams.video[0]
    source_fps = float(stream.average_rate)
    origin = None
    next_t = 0
    records, rgb_values, times, overlays = [], [], [], []
    last_pose = None
    last_pose_t = -1
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame in container.decode(stream):
        if frame.time is None:
            continue
        if origin is None:
            origin = frame.time
        t = float(frame.time - origin)
        if max_seconds is not None and t > max_seconds:
            break
        if t + 0.001 < next_t:
            continue
        next_t += 1/sample_fps
        rgb = frame.to_ndarray(format="rgb24")
        h, w = rgb.shape[:2]
        timestamp = round(t*1000)
        if t - last_pose_t >= 0.19:
            small=cv2.resize(rgb,(640,round(h*640/w)))
            im=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(small))
            last_pose = pose.detect_for_video(im, timestamp)
            last_pose_t = t
        # In room-scale video the face is too small for full-frame face detection.
        # Pose nose/shoulders localize a crop; use original pixels for color sampling.
        x0,y0,cw,ch=0,0,w,h
        if last_pose and last_pose.pose_landmarks:
            p=last_pose.pose_landmarks[0]
            if p[0].visibility>.5:
                center=np.array([p[0].x*w,p[0].y*h])
                shoulder=np.linalg.norm(np.array([(p[11].x-p[12].x)*w,(p[11].y-p[12].y)*h]))
                size=int(np.clip(shoulder*1.35,120,min(w,h)*.6))
                x0,y0=np.maximum(center-size/2,0).astype(int)
                x1,y1=min(x0+size,w),min(y0+size,h)
                cw,ch=x1-x0,y1-y0
        crop=cv2.resize(rgb[y0:y0+ch,x0:x0+cw],(384,384))
        fim=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(crop))
        fr=face.detect_for_video(fim,timestamp)
        row = {"t":round(t,4),"face_visible":bool(fr.face_landmarks)}
        overlay = {"t":round(t,3),"rois":{},"pose":[]}
        values = []
        if fr.face_landmarks:
            lm = np.array([[x0+l.x*cw, y0+l.y*ch] for l in fr.face_landmarks[0]])
            row["head_x"], row["head_y"] = (lm[1]/[w,h]).tolist()
            eye = lm[263] - lm[33]
            row["head_roll_deg"] = float(np.degrees(np.arctan2(eye[1],eye[0])))
            row["head_yaw_proxy"] = float((lm[1,0]-(lm[234,0]+lm[454,0])/2) /
                                         max(np.linalg.norm(lm[234]-lm[454]),1))
            for name, ids in ROIS.items():
                points = lm[ids].astype(np.int32)
                overlay["rois"][name] = (points/[w,h]).round(4).tolist()
                if name in ("mouth","eyes"):
                    continue
                bx,by,bw,bh=cv2.boundingRect(points)
                bx,by=max(bx,0),max(by,0)
                bw,bh=min(bw,w-bx),min(bh,h-by)
                if bw<=0 or bh<=0:
                    continue
                mask = np.zeros((bh,bw), np.uint8)
                cv2.fillConvexPoly(mask, cv2.convexHull(points-[bx,by]), 255)
                pixels = rgb[by:by+bh,bx:bx+bw][mask > 0]
                if len(pixels) > 25:
                    # Saturated, near-black pixels are not useful pulsatile observations.
                    pixels = pixels[(pixels.max(axis=1)<250)&(pixels.min(axis=1)>15)]
                    if len(pixels)>25:
                        values.append(pixels.mean(axis=0))
        rgb_values.append(np.mean(values,axis=0) if values else [np.nan]*3)
        if last_pose and last_pose.pose_landmarks:
            lms = last_pose.pose_landmarks[0]
            pts = np.array([[l.x,l.y] for l in lms])
            visible = np.array([l.visibility>0.6 and 0<=l.x<=1 and 0<=l.y<=1 for l in lms])
            overlay["pose"] = [[round(l.x,4),round(l.y,4),bool(visible[i])] for i,l in enumerate(lms)]
            for label, ids in {"shoulders":[11,12],"hips":[23,24],
                               "hands":[15,16],"feet":[27,28]}.items():
                row[label+"_visible"] = bool(visible[ids].all())
                if visible[ids].all():
                    for axis, index in [("x",0),("y",1)]:
                        row[label+"_"+axis] = float(pts[ids,index].mean())
            if visible[[11,12,23,24]].all():
                axis = pts[[11,12]].mean(axis=0) - pts[[23,24]].mean(axis=0)
                row["torso_tilt_deg"] = float(np.degrees(np.arctan2(axis[0]*w,-axis[1]*h)))
        records.append(row)
        times.append(t)
        if len(records)%3 == 1:
            overlays.append(overlay)
        if len(records)==1:
            cv2.imwrite(str(out_dir/(path.stem+".jpg")), cv2.cvtColor(cv2.resize(rgb,(960,round(h*960/w))),cv2.COLOR_RGB2BGR))
        if len(records)%1000==0:
            print(f"{path.stem}: {t:.0f}s / {len(records)} frames",flush=True)
    face.close()
    pose.close()
    container.close()
    t = np.asarray(times)
    if len(t)<2:
        raise ValueError("No usable decoded video frames")
    fs = 1/np.median(np.diff(t))
    rgb_values = np.asarray(rgb_values)
    valid = np.isfinite(rgb_values).all(axis=1)
    pulse = np.full(len(t),np.nan)
    # Never bridge missing face intervals: each contiguous run is processed separately.
    for a,b in _runs(valid):
        if b-a >= fs*5:
            pulse[a:b] = pos_rppg(rgb_values[a:b],fs)
    windows = []
    for start in np.arange(0,max(t[-1]-30+0.01,0),5):
        m = (t>=start)&(t<start+30)
        p = pulse[m]
        coverage = float(np.isfinite(p).mean())
        hr,quality = dominant_rate(p,fs) if coverage==1 else (None,None)
        shoulders = np.array([r.get("shoulders_y",np.nan) for r in records])[m]
        respiration,rq = dominant_rate(shoulders,fs,(0.1,0.5))
        windows.append({"start_s":float(start),"end_s":float(start+30),
            "rppg_hr_bpm":hr if quality and quality>=0.35 else None,
            "spectral_concentration":quality,"face_coverage":coverage,
            "respiratory_motion_bpm":respiration if rq and rq>=0.5 else None,
            "respiratory_motion_quality":rq,
            "interpretation":"exploratory, motion-sensitive; validate against ECG/RIP"})
    for key in ("head_x","head_y","hands_x","hands_y","feet_x","feet_y","torso_tilt_deg"):
        arr = np.array([r.get(key,np.nan) for r in records])
        if len(arr)>1:
            speed = np.gradient(arr,t)
            for r,v in zip(records,speed):
                r[key+"_velocity"] = float(v)
    result = {"recording_id":path.stem,"participant_id":path.stem.split("_",1)[-1],
        "source_file":path.name,"duration_s":float(t[-1]),"source_fps":source_fps,
        "source_width":w,"source_height":h,
        "analysis_fps":float(fs),"face_coverage":float(valid.mean()),
        "algorithm":"POS + MediaPipe landmarks; visible-body kinematic proxies",
        "model_sha256":{n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in assets.items()},
        "times":t.round(4),"rppg":pulse,"kinematics":records,"windows":windows,
        "overlays":overlays,"pose_edges":POSE_EDGES,
        "limitations":["No emotion or truth inference from appearance",
            "2D torso axis is not anatomical spine curvature or true centre of mass",
            "Occluded limbs remain missing","Respiratory movement is not chest volume"]}
    target=out_dir/(path.stem+".json")
    temporary=target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(finite_json(result),ensure_ascii=False),encoding="utf-8")
    temporary.replace(target)
    import pandas as pd
    pd.DataFrame(records).to_csv(out_dir/(path.stem+".kinematics.csv"),index=False)
    pd.DataFrame(windows).to_csv(out_dir/(path.stem+".windows.csv"),index=False)
    print(f"DONE {path.stem}: {t[-1]:.1f}s face={valid.mean():.1%}",flush=True)
    return result
