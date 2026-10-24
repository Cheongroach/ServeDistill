"""Observe a short continuation of an existing, validated training checkpoint."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from servedistill.observation import TrainingObservation
from servedistill.train import train


def main():
    parser = argparse.ArgumentParser()
    for name in ("student", "teacher", "data", "config", "resume", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--backend", choices=("none", "nsys", "torch"), required=True)
    parser.add_argument("--first-update", type=int, required=True)
    parser.add_argument("--last-update", type=int, required=True)
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--torch-first-microbatch", action="store_true")
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    args = parser.parse_args()
    checkpoint = json.loads((Path(args.resume) / "state.json").read_text())
    if args.first_update <= checkpoint["step"] + 1:
        parser.error("Leave at least one full training update for warmup")
    root = Path(args.output)
    observation = TrainingObservation(
        root / "observation", args.backend, args.first_update, args.last_update,
        microbatch=args.microbatch, record_shapes=args.record_shapes,
        profile_memory=args.profile_memory, with_stack=args.with_stack,
        torch_first_microbatch=args.torch_first_microbatch)
    options = SimpleNamespace(
        student=args.student, teacher=args.teacher, teacher_device="cuda:0",
        data=args.data, config=args.config, output=str(root / "train"),
        teacher_labels=None, adapter=None, resume=args.resume,
        stop_after=args.last_update, observation=observation)
    try:
        result = train(options)
        (root / "continuation.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        observation.close()


if __name__ == "__main__":
    main()
