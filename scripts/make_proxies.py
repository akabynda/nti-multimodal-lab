"""Make local browser-compatible H264 previews without altering source media."""
from pathlib import Path
import subprocess
import concurrent.futures
import argparse
import imageio_ffmpeg

def convert(pair):
    src,out=pair
    if out.exists():
        return str(out)
    temp=out.with_name(out.stem+".partial.mp4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-hide_banner","-loglevel","error","-y",
        "-i",str(src),"-map","0:v:0","-map","0:a:0?","-vf","yadif,scale=960:-2",
        "-c:v","libx264","-preset","veryfast","-crf","24","-threads","3",
        "-c:a","aac","-b:a","96k","-movflags","+faststart",str(temp)],check=True)
    temp.replace(out)
    print("PROXY "+src.name,flush=True)
    return str(out)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="../1_30")
    p.add_argument("--output",default="outputs/media")
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        source=Path(a.input)
        files=([source] if source.is_file() else sorted(f for f in source.iterdir()
                if f.suffix.lower() in {".mts",".mp4",".mov",".mkv",".m2ts"}))
        list(pool.map(convert,[(f,out/(f.stem+".mp4")) for f in files]))
