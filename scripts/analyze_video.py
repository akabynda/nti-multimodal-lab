from pathlib import Path
import argparse
import concurrent.futures
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from nti_lab.vision import process_video, models

def run(args):
    return process_video(*args)["recording_id"]

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="../1_30")
    p.add_argument("--output",default="outputs/video")
    p.add_argument("--models",default="models")
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--max-seconds",type=float)
    p.add_argument("--start-index",type=int,default=0)
    p.add_argument("--end-index",type=int)
    a=p.parse_args()
    models(Path(a.models))
    source=Path(a.input)
    files=([source] if source.is_file() else sorted(f for f in source.iterdir()
            if f.suffix.lower() in {".mts",".mp4",".mov",".mkv",".m2ts"}))
    files=files[a.start_index:a.end_index]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        for name in pool.map(run,[(f,Path(a.output),Path(a.models),12.5,a.max_seconds) for f in files]):
            print(name,flush=True)
