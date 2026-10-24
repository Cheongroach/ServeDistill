"""Export a selected LoRA checkpoint into auditable BF16 language weights."""
import argparse
import hashlib
import json
from pathlib import Path
from .native_export import repackage_native
from .runtime import Runtime, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",required=True)
    parser.add_argument("--adapter",required=True)
    parser.add_argument("--selection",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--vllm-native",action="store_true")
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError("Refusing to replace a deployment export")
    selection=json.loads(Path(a.selection).read_text())
    if Path(a.adapter).name != selection["path"]:
        raise ValueError("Adapter directory does not match the frozen dev selection")
    runtime=Runtime(a.model,adapter_path=a.adapter)
    a.output.mkdir(parents=True)
    runtime.student.save_pretrained(a.output,safe_serialization=True,max_shard_size="4GB")
    runtime.tokenizer.save_pretrained(a.output)
    packaging = repackage_native(Path(a.model), a.output) if a.vllm_native else {
        "native_wrapper": False, "architecture": runtime.student.config.model_type}
    files={p.name:{"sha256":hashlib.file_digest(p.open("rb"),"sha256").hexdigest(),
                   "bytes":p.stat().st_size}
           for p in sorted(a.output.iterdir()) if p.is_file()}
    write_json(a.output/"export-manifest.json", {
        "selection":selection,"precision":"BF16","architecture":runtime.student.config.model_type,
        "adapter_merged":True,"test_used_for_selection":False,"files":files,
        "loading":runtime.contract["loading"],
        "packaging":packaging,
    })
    print(json.dumps({"event":"export_completed","files":len(files),"selection_step":selection["step"]}),flush=True)

if __name__ == "__main__":main()
