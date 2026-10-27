"""Export research summaries and task scores with private platform metadata removed."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil

from scripts.export_public_evidence import FIELDS


PRIVATE_KEYS = {
    "sid", "job_run_sid", "job_run_id", "trial_sid", "trial_id", "trial_pod_sid",
    "namespace", "created_by", "username", "user_name", "source_username", "pod_name", "host",
}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def scrub(value, key=""):
    if isinstance(value, dict):
        return {scrub(k): scrub(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(item, key) for item in value]
    if (key in PRIVATE_KEYS or key.endswith("_sid")) and value is not None:
        return "redacted-" + sha(str(value).encode())[:12]
    if not isinstance(value, str):
        return value
    if re.fullmatch(r"[0-9a-f]{16}", value) or re.fullmatch(r"\d{9,}", value):
        return "redacted-" + sha(value.encode())[:12]
    value = re.sub(r"https?://[^\s\"'<>]*?(?:bytedance\.net|byted\.org|byteintl\.net)[^\s\"'<>]*",
                   "<private-platform-url>", value)
    value = re.sub(r"hdfs://[^\s\"'<>]+", "<private-evidence-location>", value)
    value = re.sub(r"/mnt/hdfs/[^\s\"'<>]+", "<private-evidence-mount>", value)
    value = re.sub(r"/mlx_devbox/[^\s\"'<>]+", "<private-workspace-path>", value)
    value = re.sub(r"/home/[^/\s]+/\.merlin-cli/bin/merlin-cli", "merlin-cli", value)
    value = re.sub(r"/home/byte_[^\s\"'<>]+", "<private-evidence-location>", value)
    value = re.sub(r"\b[A-Za-z0-9_-]+\.exp\b", "<private-owner>", value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Original project root")
    parser.add_argument("--output", type=Path, required=True, help="Public project root")
    args = parser.parse_args()
    source, output = args.source / "reports/advantage", args.output / "reports/advantage"
    if source.resolve() == output.resolve():
        raise ValueError("Never scrub original evidence in place")
    output.mkdir(parents=True, exist_ok=True)
    audit = []
    for path in sorted(source.glob("*")):
        if path.suffix not in (".json", ".md") or not path.is_file():
            continue
        content = path.read_bytes()
        if path.suffix == ".json":
            public = (json.dumps(scrub(json.loads(content)), ensure_ascii=False, indent=2) + "\n").encode()
        else:
            text = scrub(content.decode())
            text = text.replace("../../.ops/router-stability-dev-v1/contract.json",
                                "../../configs/router-stability-dev-v1.json")
            text = text.replace("../../.ops/aopd-router-confirmation-v1/protocol.json",
                                "../../configs/router-confirmation-v1.json")
            text = text.replace("../../.ops/aopd-router-confirmation-v1/analysis-contract.json",
                                "../../configs/router-confirmation-analysis-v1.json")
            text = text.replace("raw/router-stability-dev-v1/collection-audit.json",
                                "router-stability-collection-audit.json")
            text = text.replace("raw/aopd-router-confirmation-v1/collection-audit.json",
                                "router-confirmation-collection-audit.json")
            if path.name == "PROTOCOL.md":
                text = ("历史协议及迭代记录（2026-10-25 起）。其中旧并发预算不适用于 "
                        "2026-10-27 新轮；新轮最多 8 H20，见 anchor-development-v1.json。\n\n" + text)
            public = text.encode()
        (output / path.name).write_bytes(public)
        audit.append({"path": str((output / path.name).relative_to(args.output)),
                      "original_sha256": sha(content), "public_sha256": sha(public),
                      "original_bytes": len(content), "public_bytes": len(public)})
    for relative, target in (
        (".ops/router-stability-dev-v1/contract.json", "configs/router-stability-dev-v1.json"),
        (".ops/aopd-router-confirmation-v1/protocol.json", "configs/router-confirmation-v1.json"),
        (".ops/aopd-router-confirmation-v1/analysis-contract.json", "configs/router-confirmation-analysis-v1.json"),
        ("reports/advantage/raw/router-stability-dev-v1/collection-audit.json",
         "reports/advantage/router-stability-collection-audit.json"),
        ("reports/advantage/raw/aopd-router-confirmation-v1/collection-audit.json",
         "reports/advantage/router-confirmation-collection-audit.json"),
    ):
        path = args.source / relative
        content = path.read_bytes()
        public = (json.dumps(scrub(json.loads(content)), ensure_ascii=False, indent=2) + "\n").encode()
        destination = args.output / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(public)
        audit.append({"path": target, "original_sha256": sha(content), "public_sha256": sha(public)})
    figures = source / "figures"
    if figures.exists():
        for path in figures.iterdir():
            if path.suffix in (".png", ".pdf", ".svg"):
                (output / "figures").mkdir(exist_ok=True)
                destination = output / "figures" / path.name
                if path.suffix == ".svg":
                    destination.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
                else:
                    shutil.copyfile(path, destination)
    for study in ("router-stability-dev-v1", "aopd-router-confirmation-v1"):
        destination = output / (study + "-scores.jsonl.gz")
        count, sources = 0, []
        with gzip.open(destination, "wt", encoding="utf-8") as stream:
            for path in sorted((source / "raw" / study).rglob("*.jsonl")):
                content = path.read_bytes()
                sources.append({"path": str(path.relative_to(source / "raw" / study)),
                                "sha256": sha(content)})
                for line in content.decode().splitlines():
                    row = json.loads(line)
                    if "target_class" not in row and "success" not in row:
                        continue
                    record = {key: row[key] for key in FIELDS if key in row}
                    scope = "training_dev" if "training" in path.relative_to(source / "raw" / study).parts else (
                        "confirmation" if study == "aopd-router-confirmation-v1" else "development")
                    stream.write(json.dumps({"scope": scope, "split": path.stem,
                                             "source": str(path.relative_to(source / "raw" / study)),
                                             **record}, ensure_ascii=False) + "\n")
                    count += 1
        audit.append({"path": str(destination.relative_to(args.output)), "records": count,
                      "public_sha256": sha(destination.read_bytes()), "original_sources": sources,
                      "boundary": "Scope/split distinguish training DEV from frozen confirmation. Repeated seeds/checkpoints are not new independent queries; raw responses remain separate."})
    (output / "PUBLIC_EXPORT_MANIFEST.json").write_text(json.dumps({
        "records": audit, "boundary": "Private platform metadata removed; original and public byte hashes are distinct. Numeric research results unchanged.",
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary_exports": len(audit), "output": str(output)}))


if __name__ == "__main__":
    main()
