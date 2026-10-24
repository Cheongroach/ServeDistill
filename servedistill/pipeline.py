"""Independent experiment execution, immutable dev selection, and final scoring."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
from .runtime import write_json

def call(args):
    subprocess.run([sys.executable, *args], check=True)

def selected(output):
    value=json.loads((output / "selected.json").read_text())
    path=output / value["path"]
    if not (path / "adapter_model.safetensors").exists():
        raise RuntimeError("Selected checkpoint has no immutable adapter")
    return str(path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--student",required=True)
    p.add_argument("--teacher",required=True)
    p.add_argument("--data",required=True)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--teacher-labels")
    p.add_argument("--business-config",type=Path)
    p.add_argument("--benchmark",action="store_true")
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    common=["--student",a.student,"--teacher",a.teacher,"--data",a.data]
    adapter=None
    if a.business_config:
        before=a.output / "business_stage"
        call(["-m","servedistill.train",*common,"--config",str(a.business_config),"--output",str(before)])
        adapter=selected(before)
        call(["-m","servedistill.evaluate",*common,"--adapter",adapter,"--output",str(a.output / "business_final"),"--phase","final"])
    train=a.output / "train"
    argv=["-m","servedistill.train",*common,"--config",str(a.config),"--output",str(train)]
    if a.teacher_labels:argv += ["--teacher-labels",a.teacher_labels]
    if adapter:argv += ["--adapter",adapter]
    call(argv)
    chosen=selected(train)
    write_json(a.output / "frozen-selection.json",json.loads((train / "selected.json").read_text()))
    # Final phase is a fresh process, with the adapter merged into BF16 deployment weights.
    call(["-m","servedistill.evaluate","--student",a.student,"--data",a.data,"--adapter",chosen,
          "--output",str(a.output / "final"),"--phase","final"])
    if a.benchmark:
        call(["-m","servedistill.benchmark","--model",a.student,"--adapter",chosen,
              "--data",str(Path(a.data)/"test.jsonl"),"--output",str(a.output / "benchmark"),
              "--per-domain","12","--backend","hf"])
    write_json(a.output / "pipeline-completed.json",{"completed":True,"selection":chosen,
               "final_precision":"BF16 merged adapter","test_used_for_selection":False})

if __name__ == "__main__":main()
