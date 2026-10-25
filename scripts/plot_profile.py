"""Plot actual CUDA kernel intervals and NVTX ranges from analyze_profile outputs."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--case', default='nsys_m1')
    args = parser.parse_args()
    data = json.loads((args.input / (args.case + '-cuda-summary.json')).read_text())
    window = data['representative_window']
    begin, end = window['start_seconds'], window['end_seconds']
    phases = ['learner_forward', 'teacher_forward', 'full_vocabulary_kl', 'reference_forward', 'backward']
    labels = ['Student forward', 'Teacher forward', 'Full vocabulary KL', 'Reference forward', 'Backward']
    palette = dict(zip(phases, ['#166534', '#0d9488', '#475569', '#2563eb', '#7c3aed']))
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2.2, 1, 1.4]})
    ax = axes[0]
    for scope in data['host_scopes']:
        if begin <= scope['start_seconds'] < end and scope['name'] in phases:
            name = scope['name']; x = scope['start_seconds'] - begin
            width = scope['end_seconds'] - scope['start_seconds']
            ax.broken_barh([(x, width)], (phases.index(name)-.22, .44), facecolors=palette[name])
            ax.text(x+width/2, phases.index(name)-.32, f'{width*1000:.1f} ms', ha='center', fontsize=9)
    ax.set_yticks(range(5), labels); ax.set_xlim(0, end-begin)
    update_name = data.get('update_name', 'update_42')
    ax.set_title(f'{args.case}: {update_name}, first microbatch ({begin:.3f}-{end:.3f} s)', loc='left')
    ax.set_xlabel('Seconds from first student forward (NVTX host ranges)')
    ax.invert_yaxis()
    kernels = list(csv.DictReader((args.input / (args.case+'-representative-kernels.csv')).open()))
    segments = {}
    for row in kernels:
        phase = row['phase'].removesuffix('_cross_thread_temporal')
        phase = phase if phase in phases else 'Other'
        start = float(row['start_ms']) / 1000
        duration = float(row['duration_us']) / 1e6
        segments.setdefault(phase, []).append([(start, .5), (start+duration, .5)])
    gpu = axes[1]
    for phase, lines in segments.items():
        gpu.add_collection(LineCollection(lines, colors=palette.get(phase, '#94a3b8'), linewidths=12))
    gpu.set_xlim(0, end-begin); gpu.set_ylim(0, 1)
    gpu.set_yticks([.5], ['GPU kernels']); gpu.set_xlabel('Seconds from first student forward (actual CUDA intervals)')
    density = list(csv.DictReader((args.input / (args.case+'-launch-density.csv')).open()))
    axes[2].plot([float(r['start_seconds']) for r in density],
                 [int(r['kernels_started_in_10ms']) for r in density], color='#166534', lw=.75)
    axes[2].axvspan(begin, end, facecolor='#94a3b8', alpha=.2)
    axes[2].set_xlabel('Seconds from update start'); axes[2].set_ylabel('Kernels started / 10 ms')
    coverage = data['kernel_copy_memset_union_seconds_in_update'] / data['host_update_seconds'] * 100
    fig.text(.02, .015, f"Full update: {data['host_update_seconds']:.2f} s; "
             f"{data['cuda_kernel_count']:,} kernels; CUDA activity union {coverage:.1f}%. "
             "Activity coverage is not SM occupancy.\n"
             "Backward phase attribution includes cross-thread temporal association. "
             "Colors indicate launch phase; host range and device completion may differ.", fontsize=9)
    for axis in axes:
        axis.grid(axis='x', alpha=.15); axis.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(rect=[0, .07, 1, 1])
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(args.input / (args.case+'-timeline.'+extension), dpi=180)
        if extension == 'svg':
            path = args.input / (args.case+'-timeline.svg')
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    print(args.input / (args.case+'-timeline.png'))


if __name__ == '__main__':
    main()
