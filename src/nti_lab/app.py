"""Local five-zone research interface, saved annotations and explicit clocks."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import re
from functools import lru_cache
import numpy as np
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from nti_lab.story_topics import display_story_titles

ROOT=Path(os.environ.get("NTI_PROJECT_ROOT",Path(__file__).resolve().parents[2])).resolve()
DATA=Path(os.environ.get("NTI_DATA_ROOT",ROOT.parent)).resolve()
OUTPUT=ROOT/"outputs"
STATIC=Path(__file__).parent/"static"
app=FastAPI(title="НТИ · Анализ записей",version="0.1.0")
app.mount("/static",StaticFiles(directory=STATIC),name="static")


def read_json(path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # A legacy producer may be replacing an output during processing.
        return default


def safe_id(s):
    if not re.fullmatch(r"[A-Za-z0-9_-]+",s):
        raise HTTPException(400,"Invalid recording identifier")
    return s


def session_ids():
    found={p.stem for p in (DATA/"1_30").glob("*.MTS")}
    found|={p.stem for p in (OUTPUT/"media").glob("*.mp4") if ".partial" not in p.name}
    found|={p.stem for p in (OUTPUT/"video").glob("*.json")}
    return sorted(found)


def checked(s):
    safe_id(s)
    if s not in session_ids():
        raise HTTPException(404,"Session not found")
    return s


def load_csv(p):
    if not p.exists():
        return []
    with p.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


@app.get("/")
def index():
    return FileResponse(STATIC/"index.html")


@app.get("/api/health")
def health():
    return {"status":"ok","sessions":len(session_ids())}


@app.get("/api/sessions")
def sessions():
    return [{"id":s,"participant_id":s.split("_",1)[-1],
             "video_ready":(OUTPUT/"media"/f"{s}.mp4").exists(),
             "vision_ready":(OUTPUT/"video"/f"{s}.json").exists(),
             "audio_ready":(OUTPUT/"audio"/f"{s}.segments.json").exists()} for s in session_ids()]


@app.get("/api/session/{sid}")
def session(sid):
    checked(sid)
    participant=sid.split("_",1)[-1]
    alignment=read_json(OUTPUT/"sync"/f"{sid}.json",{})
    suggested=alignment.get("suggested_offset_s")
    cfg=read_json(ROOT/"data/private/annotations"/f"{sid}.json",
                  {"bio_offset_s":suggested if suggested is not None else 0,
                   "bio_scale":1,"sync_verified":False,"speaker_roles":{},"events":[]})
    stories=read_json(ROOT/"data/private/stories.json",{"stories":[]})
    if isinstance(stories,dict):
        stories=stories.get("stories",[])
    bio_dir=OUTPUT/"bio"/participant
    comparison_path=OUTPUT/"research/content_comparison.json"
    comparisons=read_json(comparison_path,{}).get("comparisons",[])
    recheck_doc=read_json(OUTPUT/"research/audio_recheck.json",{})
    current_hash=hashlib.sha256(comparison_path.read_bytes()).hexdigest() if comparison_path.exists() else None
    rechecks=(recheck_doc.get("candidate_checks",[])
              if current_hash and recheck_doc.get("source_comparisons_sha256")==current_hash else [])
    profile=next((row for row in load_csv(OUTPUT/"research/profile_link_audit.csv")
                  if row.get("participant_id")==participant
                  and row.get("profile_link_status") in {"explicit_tab_number_match","requested_id_not_found"}),None)
    names=read_json(ROOT/"data/private/participant_names.json",{})
    return {"id":sid,"participant_id":participant,"annotation":cfg,
        "profile":profile,"profile_display_name":names.get(participant),
        "comparisons":[display_story_titles(r) for r in comparisons if r.get("participant_id")==participant],
        "audio_rechecks":[r for r in rechecks if r.get("participant_id")==participant],
        "alignment":alignment,
        "audio":read_json(OUTPUT/"audio"/f"{sid}.segments.json",{}),
        "audio_summary":read_json(OUTPUT/"audio"/f"{sid}.summary.json",{}),
        "audio_features":load_csv(OUTPUT/"audio"/f"{sid}.features.csv"),
        "video":read_json(OUTPUT/"video"/f"{sid}.json",{}),
        "bio":read_json(bio_dir/"metadata.json",{}),
        "biosignals":read_json(bio_dir/"timeseries.json",{}),
        "hrv":load_csv(bio_dir/"metrics.csv"),
        "ppg_metrics":load_csv(bio_dir/"ppg_metrics.csv"),
        "stories":[display_story_titles(s) for s in stories if s.get("participant_id")==participant],
        "matches":[display_story_titles(s) for s in load_csv(OUTPUT/"research/story_matches.csv")
                   if s.get("recording_id")==sid],
        "media_ready":(OUTPUT/"media"/f"{sid}.mp4").exists()}


@lru_cache(maxsize=8)
def signal_arrays(participant,mtime):
    with np.load(OUTPUT/"bio"/participant/"signals.npz") as d:
        return {k:d[k] for k in d.files}


@app.get("/api/session/{sid}/signal-window")
def signal_window(sid,t:float=0):
    checked(sid)
    if not np.isfinite(t):
        raise HTTPException(400,"Finite time required")
    participant=sid.split("_",1)[-1]
    p=OUTPUT/"bio"/participant/"signals.npz"
    if not p.exists():
        return {}
    arr=signal_arrays(participant,p.stat().st_mtime_ns)
    et=arr["ecg_t_s"]
    if not len(et) or t<et[0] or t>et[-1]:
        return {"status":"outside_recording"}
    ix=np.searchsorted(et,[t-10,t+10])
    ecg=arr["ecg_filtered_mv"][ix[0]:ix[1]]
    result={"ecg":{"t_s":et[ix[0]:ix[1]].tolist(),
                   "mv":[float(x) if np.isfinite(x) else None for x in ecg]}}
    from nti_lab.biosignals import rr_spectrum
    window_start=max(0,min(t-60,float(et[-1])-120))
    spectrum=rr_spectrum(arr["rr_t_s"],arr["rr_ms"],arr["rr_valid"],
                         window_start,window_start+120)
    if spectrum["status"]!="insufficient_quality":
        result["spectrum"]={"f":spectrum["frequency_hz"],"power":spectrum["power_ms2_hz"]}
    return result


class Event(BaseModel):
    start:float=Field(ge=0,allow_inf_nan=False)
    end:float=Field(ge=0,allow_inf_nan=False)
    label:Literal["unknown","baseline","truth","lie","question","answer","topic"]="unknown"
    note:str=Field(default="",max_length=4000)
    story_id:str|None=None
    verified:bool=False


class Annotation(BaseModel):
    bio_offset_s:float=Field(default=0,ge=-100000,le=100000,allow_inf_nan=False)
    bio_scale:float=Field(default=1,ge=.9,le=1.1,allow_inf_nan=False)
    sync_verified:bool=False
    speaker_roles:dict[str,Literal["interviewer","respondent","unknown"]]={}
    events:list[Event]=Field(default_factory=list,max_length=5000)


@app.put("/api/session/{sid}/annotation")
def save_annotation(sid,annotation:Annotation):
    checked(sid)
    if any(e.end<=e.start for e in annotation.events):
        raise HTTPException(422,"End must be greater than start")
    folder=ROOT/"data/private/annotations"
    folder.mkdir(parents=True,exist_ok=True)
    p=folder/f"{sid}.json"
    temporary=p.with_suffix(".tmp")
    temporary.write_text(annotation.model_dump_json(indent=2),encoding="utf-8")
    temporary.replace(p)
    return {"saved":True}


@app.get("/media/{sid}")
def media(sid):
    checked(sid)
    p=OUTPUT/"media"/f"{sid}.mp4"
    if not p.exists():
        raise HTTPException(404,"Preview is not yet processed")
    return FileResponse(p,media_type="video/mp4")


@app.get("/api/research")
def research():
    return read_json(OUTPUT/"research/summary.json",{})
