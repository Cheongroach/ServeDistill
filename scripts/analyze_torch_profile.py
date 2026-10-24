"""Summarize real Chrome-trace categories and recorded input shapes."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from analyze_profile import union_ns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.input.open() as stream:
        trace = json.load(stream)
    events = trace['traceEvents']
    categories = Counter(event.get('cat', '') for event in events)
    kernels = [event for event in events if event.get('cat') == 'kernel']
    operations = defaultdict(lambda: {'calls': 0, 'inclusive_host_us': 0.0})
    shapes = defaultdict(lambda: {'calls': 0, 'inclusive_host_us': 0.0})
    for event in events:
        if event.get('cat') != 'cpu_op':
            continue
        name = event['name']
        shape = event.get('args', {}).get('Input Dims', [])
        for table, key in ((operations, name), (shapes, (name, json.dumps(shape)))):
            table[key]['calls'] += 1
            table[key]['inclusive_host_us'] += event.get('dur', 0.0)
    spans = [{key: event.get(key) for key in ('name', 'cat', 'ts', 'dur')}
             for event in events if event.get('name') in (
                 'learner_forward', 'teacher_forward', 'full_vocabulary_kl',
                 'reference_forward', 'backward')]
    summary = {
        'trace_sha256': hashlib.file_digest(args.input.open('rb'), 'sha256').hexdigest(),
        'bytes': args.input.stat().st_size,
        'event_count': len(events), 'categories': dict(categories),
        'actual_kernel_count': len(kernels),
        'actual_summed_kernel_seconds': sum(event['dur'] for event in kernels) / 1e6,
        'actual_kernel_union_seconds': union_ns([
            (event['ts'], event['ts'] + event['dur']) for event in kernels]) / 1e6,
        'recorded_scopes': spans,
        'top_cpu_operations_by_count': [dict(name=name, **value) for name, value in
            sorted(operations.items(), key=lambda pair: pair[1]['calls'], reverse=True)[:30]],
        'top_shapes_by_count': [dict(name=name, input_dims=json.loads(shape), **value)
            for (name, shape), value in sorted(
                shapes.items(), key=lambda pair: pair[1]['calls'], reverse=True)[:30]],
        'matrix_shapes': [dict(name=name, input_dims=json.loads(shape), **value)
            for (name, shape), value in sorted(shapes.items(),
                key=lambda pair: pair[1]['calls'], reverse=True) if name in ('aten::mm', 'aten::bmm')][:30],
        'boundary': ('Only cat=kernel counted as CUDA kernels. GPU annotations and CPU '
                     'operation durations are nonadditive. Window covers first microbatch, '
                     'not rollout, optimizer, or the complete update.'),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
