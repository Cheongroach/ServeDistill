"""A measured method budget that survives cold process restoration."""
from __future__ import annotations

import math
import time


class TrainingBudget:
    def __init__(self, seconds=None, prior_seconds=0.0, *, clock=time.monotonic,
                 started=None, restored=None):
        if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
            raise ValueError("Time budget must be finite and positive")
        if not math.isfinite(prior_seconds) or prior_seconds < 0:
            raise ValueError("Prior method costs must be finite and nonnegative")
        if seconds is None and prior_seconds:
            raise ValueError("Prior cost credit requires a method time budget")
        if seconds is not None and prior_seconds >= seconds:
            raise ValueError("Prior method costs exhaust the time budget")
        self.seconds = seconds
        self.prior_seconds = prior_seconds
        self.clock = clock
        self.started = clock() if started is None else started
        self.restored_seconds = 0.0
        if restored is not None:
            if (restored["limit_seconds"] != seconds
                    or restored["prior_method_seconds"] != prior_seconds):
                raise ValueError("Restored time budget differs from the frozen method budget")
            self.restored_seconds = restored["training_wall_seconds"]
            if not math.isfinite(self.restored_seconds) or self.restored_seconds < 0:
                raise ValueError("Invalid restored elapsed time")

    def snapshot(self):
        training = self.restored_seconds + self.clock() - self.started
        total = self.prior_seconds + training
        return {
            "limit_seconds": self.seconds, "prior_method_seconds": self.prior_seconds,
            "training_wall_seconds": training, "charged_method_seconds": total,
            "exhausted": self.seconds is not None and total >= self.seconds,
            "overrun_seconds": max(0.0, total - self.seconds) if self.seconds else 0.0,
            "boundary": "Includes model initialization, training dev, updates and checkpoints. "
                        "Stops between atomic updates; final evaluation and platform costs "
                        "are reported separately and do not disappear from total GPU cost.",
        }

    def exhausted(self):
        return self.snapshot()["exhausted"]


def completed_checkpoint(config, completed):
    """Reject a step cap presented as an equal-time experiment."""
    if not completed.get("completed") or completed["steps"] < 1:
        raise ValueError("Training has no completed update")
    if config.get("method_time_budget_seconds") is None:
        if completed["steps"] != config["steps"]:
            raise ValueError("Training did not reach the frozen update budget")
    else:
        budget = completed["time_budget"]
        if (budget["limit_seconds"] != config["method_time_budget_seconds"]
                or budget["prior_method_seconds"] != config.get("prior_method_seconds", 0.0)
                or not budget["exhausted"]
                or completed["stop_reason"] != "method_time_budget"):
            raise ValueError("Training did not exhaust the frozen method time budget")
    return f"checkpoint-{completed['steps']:04d}"
