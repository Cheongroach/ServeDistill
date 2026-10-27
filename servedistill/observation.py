"""Bounded training observation without changing the probability or loss contract."""
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import time

import torch


class TrainingObservation:
    def __init__(self, output, backend, first_update, last_update, *,
                 microbatch=None, rollout_batch=None, record_shapes=False, profile_memory=False,
                 with_stack=False, torch_first_microbatch=False):
        if backend not in ("none", "nsys", "ncu", "torch"):
            raise ValueError("Unknown observation backend")
        if first_update < 1 or last_update < first_update:
            raise ValueError("Invalid observation window")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.backend = backend
        self.first_update, self.last_update = first_update, last_update
        self.microbatch = microbatch
        if rollout_batch is not None and (type(rollout_batch) is not int or rollout_batch < 1):
            raise ValueError("rollout_batch must be a positive integer")
        self.rollout_batch = rollout_batch
        self.options = dict(record_shapes=record_shapes, profile_memory=profile_memory,
                            with_stack=with_stack)
        self.active = False
        self.profiler = None
        self.torch_first_microbatch = torch_first_microbatch
        self.trace_finished = False
        self.events = []
        self.update = None
        self.origin = time.perf_counter_ns()
        self.started = None
        self.configuration = None

    def configure(self, config):
        # Called after the immutable checkpoint configuration has been validated.
        self.configuration = vars(config).copy()
        if self.microbatch is not None:
            if self.microbatch < 1:
                raise ValueError("microbatch must be positive")
            config.microbatch = self.microbatch
        if self.rollout_batch is not None:
            config.rollout_batch = self.rollout_batch
        (self.output / "configuration.json").write_text(json.dumps({
            "checkpoint_configuration": self.configuration,
            "observed_configuration": vars(config),
            "backend": self.backend,
            "first_update": self.first_update, "last_update": self.last_update,
            "profiler_options": self.options,
            "torch_first_microbatch": self.torch_first_microbatch,
            "boundary": "Diagnostic clone after strict checkpoint restoration. Rollout batching "
                        "may change sampled trajectories; prompt/group/token budgets are retained. "
                        "Original checkpoints and final selection remain unchanged.",
        }, indent=2) + "\n")

    def begin_update(self, update):
        self.update = update
        if update == self.first_update:
            if self.backend in ("nsys", "ncu"):
                torch.cuda.synchronize()
                torch.cuda.profiler.start()
            elif self.backend == "torch":
                self.profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA],
                    **self.options)
                if not self.torch_first_microbatch:
                    self.profiler.__enter__()
            self.active = True
        if self.active:
            self.started = time.perf_counter_ns()
            if self.backend in ("nsys", "ncu"):
                torch.cuda.nvtx.range_push(f"update_{update}")

    @contextmanager
    def span(self, name):
        if not self.active:
            yield
            return
        if (self.backend == "torch" and self.torch_first_microbatch
                and not self.trace_finished and name == "learner_forward"):
            self.profiler.__enter__()
        start = time.perf_counter_ns()
        if self.backend in ("nsys", "ncu"):
            torch.cuda.nvtx.range_push(name)
        context = (torch.profiler.record_function(name)
                   if self.backend == "torch" and self.profiler is not None
                   and (not self.torch_first_microbatch or name != "rollout")
                   else nullcontext())
        try:
            with context:
                yield
        finally:
            end = time.perf_counter_ns()
            if self.backend in ("nsys", "ncu"):
                torch.cuda.nvtx.range_pop()
            self.events.append({
                "name": name, "update": self.update,
                "start_ns": start - self.origin, "end_ns": end - self.origin,
                "host_wall_seconds": (end - start) / 1e9,
                "boundary": "host scope; CUDA work may finish outside the range",
            })
            if (self.backend == "torch" and self.torch_first_microbatch
                    and not self.trace_finished and name == "backward"):
                self.finish_torch_trace()

    def end_update(self, metrics):
        if not self.active:
            return
        end = time.perf_counter_ns()
        if self.backend in ("nsys", "ncu"):
            torch.cuda.nvtx.range_pop()
        self.events.append({
            "name": "update", "update": self.update,
            "start_ns": self.started - self.origin, "end_ns": end - self.origin,
            "host_wall_seconds": (end - self.started) / 1e9,
            "response_tokens": metrics["response_tokens"],
            "behavior_max_error": metrics["behavior_max_error"],
            "gradient_norm": metrics["gradient_norm"],
            "parameter_delta_l2": metrics["parameter_delta_l2"],
        })
        if self.update == self.last_update:
            self.close()

    def finish_torch_trace(self):
        torch.cuda.synchronize()
        self.profiler.__exit__(None, None, None)
        self.profiler.export_chrome_trace(str(self.output / "torch-trace.json"))
        table = self.profiler.key_averages(group_by_input_shape=self.options["record_shapes"])
        (self.output / "torch-operators.txt").write_text(
            table.table(sort_by="self_cuda_time_total", row_limit=50))
        self.profiler = None
        self.trace_finished = True

    def close(self):
        if not self.active:
            return
        if self.backend in ("nsys", "ncu"):
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
        elif self.profiler is not None:
            self.finish_torch_trace()
        self.active = False
        (self.output / "host-spans.json").write_text(
            json.dumps(self.events, indent=2) + "\n")
