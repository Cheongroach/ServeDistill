"""Use the committed serving configuration with the audited request benchmark."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/monitoring/serving.json"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["concurrency_sweeps"] != [1, 4, 16]:
        raise ValueError("The current benchmark implements concurrency 1/4/16")
    argv = [sys.executable, "-m", "servedistill.http_benchmark",
            "--data", str(args.data), "--output", str(args.output)]
    for key in ("endpoint", "model", "per_domain", "max_new_tokens", "request_timeout",
                "sweep_timeout", "gpus"):
        argv += ["--" + key.replace("_", "-"), str(config[key])]
    if args.dry_run:
        print(json.dumps(argv, indent=2))
        return
    subprocess.run(argv, check=True)


if __name__ == "__main__":
    main()
