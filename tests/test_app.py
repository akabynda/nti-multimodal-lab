import json
import numpy as np
from fastapi.testclient import TestClient
from nti_lab import app as module

def test_health_and_annotations(tmp_path,monkeypatch):
    data=tmp_path/"data";(data/"1_30").mkdir(parents=True)
    (data/"1_30/00001_Test.MTS").touch()
    monkeypatch.setattr(module,"ROOT",tmp_path)
    monkeypatch.setattr(module,"DATA",data)
    monkeypatch.setattr(module,"OUTPUT",tmp_path/"outputs")
    c=TestClient(module.app)
    assert c.get("/api/health").json()["sessions"]==1
    assert c.get("/api/session/00001_Test").status_code==200
    a={"bio_offset_s":3,"events":[{"start":1,"end":2,"label":"baseline"}]}
    assert c.put("/api/session/00001_Test/annotation",json=a).status_code==200
    assert c.get("/api/session/00001_Test").json()["annotation"]["bio_offset_s"]==3
    a["events"][0]["end"]=.5
    assert c.put("/api/session/00001_Test/annotation",json=a).status_code==422
    assert c.get("/api/session/no_such_session").status_code==404


def test_signal_outside_recording_has_no_spectrum(tmp_path,monkeypatch):
    output=tmp_path/"outputs"
    folder=output/"bio"/"Test";folder.mkdir(parents=True)
    (output/"media").mkdir()
    (output/"media"/"00001_Test.mp4").touch()
    np.savez(folder/"signals.npz",ecg_t_s=np.arange(0,10,.004))
    monkeypatch.setattr(module,"DATA",tmp_path/"raw")
    monkeypatch.setattr(module,"OUTPUT",output)
    module.signal_arrays.cache_clear()
    client=TestClient(module.app)
    for t in (-1,10,100):
        result=client.get(f"/api/session/00001_Test/signal-window?t={t}").json()
        assert result=={"status":"outside_recording"}
    module.signal_arrays.cache_clear()


def test_recheck_must_match_current_comparison_version(tmp_path,monkeypatch):
    import hashlib
    output=tmp_path/"outputs"
    (output/"media").mkdir(parents=True)
    (output/"media"/"00001_Test.mp4").touch()
    research=output/"research";research.mkdir()
    comparison=research/"content_comparison.json"
    comparison.write_text(json.dumps({"comparisons":[]}),encoding="utf-8")
    result={"candidate_checks":[{"participant_id":"Test","comparison_id":"comparison_001"}],
            "source_comparisons_sha256":"stale"}
    path=research/"audio_recheck.json"
    path.write_text(json.dumps(result),encoding="utf-8")
    monkeypatch.setattr(module,"DATA",tmp_path/"raw")
    monkeypatch.setattr(module,"OUTPUT",output)
    client=TestClient(module.app)
    assert client.get("/api/session/00001_Test").json()["audio_rechecks"]==[]
    result["source_comparisons_sha256"]=hashlib.sha256(comparison.read_bytes()).hexdigest()
    path.write_text(json.dumps(result),encoding="utf-8")
    assert len(client.get("/api/session/00001_Test").json()["audio_rechecks"])==1


def test_session_profile_uses_only_the_linked_participant_row(tmp_path,monkeypatch):
    import csv
    output=tmp_path/"outputs"
    (output/"media").mkdir(parents=True)
    for sid in ("00001_Test","00002_Other","00003_Unlinked","00004_Missing","00005_NotFound"):
        (output/"media"/f"{sid}.mp4").touch()
    research=output/"research";research.mkdir()
    profile={"participant_id":"Test","tab_number":"123456","profile_link_status":"explicit_tab_number_match",
             "profile_natural_raw":"Профиль из Excel","profile_demonstrated_raw":"Другой профиль",
             "expression":"0","emotion_positive":"0","emotion_negative":"2.50","emotion_balance":"-2.50",
             "profile_source_sheet":"Лист 1","profile_source_row":"8","profile_source_file":"profiles.xlsx"}
    other={**profile,"participant_id":"Other","tab_number":"654321","profile_natural_raw":"Чужой профиль"}
    unlinked={**profile,"participant_id":"Unlinked","profile_link_status":"unlinked"}
    not_found={**dict.fromkeys(profile,""),"participant_id":"NotFound","tab_number":"339013",
               "profile_link_status":"requested_id_not_found"}
    with (research/"profile_link_audit.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(profile))
        writer.writeheader();writer.writerows([other,profile,unlinked,not_found])
    private=tmp_path/"data/private";private.mkdir(parents=True)
    (private/"participant_names.json").write_text(json.dumps({"Test":"Тестовый участник","Other":"Другой участник"}),encoding="utf-8")
    monkeypatch.setattr(module,"ROOT",tmp_path)
    monkeypatch.setattr(module,"DATA",tmp_path/"raw")
    monkeypatch.setattr(module,"OUTPUT",output)
    client=TestClient(module.app)
    result=client.get("/api/session/00001_Test").json()
    assert result["profile"]==profile
    assert result["profile_display_name"]=="Тестовый участник"
    assert client.get("/api/session/00002_Other").json()["profile"]==other
    assert client.get("/api/session/00003_Unlinked").json()["profile"] is None
    missing=client.get("/api/session/00004_Missing").json()
    assert missing["profile"] is None and missing["profile_display_name"] is None
    assert client.get("/api/session/00005_NotFound").json()["profile"]==not_found
