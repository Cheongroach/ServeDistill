"""Independent frozen baseline evaluation and reference serving measurement."""
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--student",required=True)
    p.add_argument("--data",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--teacher-mode",action="store_true")
    a=p.parse_args()
    args=[sys.executable,"-m","servedistill.evaluate","--student",a.model,
          "--data",a.data,"--output",str(a.output/"final"),"--phase","final"]
    if a.teacher_mode:
        args=[sys.executable,"-m","servedistill.evaluate","--student",a.student,
              "--teacher",a.model,"--evaluate-teacher","--data",a.data,
              "--output",str(a.output/"final"),"--phase","final"]
    subprocess.run(args,check=True)
    subprocess.run([sys.executable,"-m","servedistill.benchmark","--model",a.model,
                    "--data",str(Path(a.data)/"test.jsonl"),"--output",str(a.output/"benchmark"),
                    "--backend","hf","--per-domain","12"],check=True)

if __name__ == "__main__":main()
