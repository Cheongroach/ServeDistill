"""Verify original capture hashes and export sanitized diagnostics for publication."""
import argparse
import gzip
import json
from pathlib import Path

from scripts.export_research_evidence import scrub
from servedistill.profiling import digest, report_validity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+",
                        help="Relative capture directories; default is the four original suite backends")
    args = parser.parse_args()
    source = args.evidence.resolve()
    if args.output.resolve() == source:
        raise ValueError("Do not sanitize original evidence in place")
    args.output.mkdir(parents=True, exist_ok=False)
    verified, audits, exports = {}, {}, []
    cases = ({name: Path(name) for name in args.cases} if args.cases else {
        backend: Path("profiles") / backend for backend in ("none", "ncu", "nsys", "torch")})
    for backend, relative in cases.items():
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Capture case must stay inside the evidence root")
        case = source / relative
        index = json.loads((case / "outputs.json").read_text())
        for relative, record in index.items():
            path = case / relative
            if not path.is_file() or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                raise ValueError("Collected capture differs from original output index: " + relative)
        verified[backend] = {"files": len(index), "all_hashes_match": True}
        config = json.loads((case / "manifest.json").read_text())["configuration"]
        audits[backend] = report_validity(case, config, json.loads((case / "tool-audit.json").read_text()))
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        original_hash = digest(path)
        original_bytes = path.stat().st_size
        if path.suffix not in (".json", ".jsonl", ".txt", ".csv", ".log"):
            exports.append({"source": str(relative), "original_sha256": original_hash,
                            "original_bytes": original_bytes, "public_file": None,
                            "boundary": "Original binary/model state retained in the local evidence directory."})
            continue
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "torch-trace.json":
            # Trace contains timing/shape metadata. Prompts and model weights are not present.
            text = scrub(path.read_text())
            destination = destination.with_suffix(".json.gz")
            with gzip.GzipFile(filename=str(destination), mode="wb", mtime=0) as stream:
                stream.write(text.encode())
        elif path.suffix == ".json":
            destination.write_text(json.dumps(scrub(json.loads(path.read_text())), indent=2) + "\n")
        elif path.suffix == ".jsonl":
            with destination.open("w") as stream:
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    # Rollout contracts include responses; expose only existing numeric/contract metadata.
                    if path.name == "rollouts.jsonl":
                        row.pop("text", None)
                        row.pop("texts", None)
                    stream.write(json.dumps(scrub(row)) + "\n")
        else:
            text = scrub(path.read_text(errors="replace"))
            destination.write_text("\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n")
        exports.append({"source": str(relative), "original_sha256": original_hash,
                        "original_bytes": original_bytes,
                        "public_file": str(destination.relative_to(args.output)),
                        "public_sha256": digest(destination), "public_bytes": destination.stat().st_size})
    (args.output / "PUBLIC_EXPORT_MANIFEST.json").write_text(json.dumps({
        "original_hash_verification": verified, "independent_report_validation": audits,
        "records": exports, "boundary": (
            "Sanitized public bytes have distinct hashes; plain-text trailing whitespace is normalized. "
            "Original binary reports and training checkpoints "
            "remain separate. Profiled timings cannot support speed claims.")
    }, indent=2) + "\n")
    print(json.dumps({"exports": len(exports), "original_hash_verification": verified}))


if __name__ == "__main__":
    main()
