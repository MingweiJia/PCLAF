"""Evaluate each trajectory and summarize runs and repeated fits separately."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .data import trajectory_batches


METHOD_NAMES = {'raw': 'Raw', 'pclaf': 'PCLAF', 'vae': 'VAE', 'gan': 'GAN'}
MODEL_NAMES = {'cnn': 'CNN', 'lstm': 'LSTM', 'transformer': 'Transformer', 'pilstm': 'PI-LSTM'}
UNITS = ('range_normalized', 'mole_percentage_points')


def evaluate(model, test, scaler, device='cpu', batch_size=512):
    """Report physical errors and errors divided by the full target range."""
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    model.eval()
    rows = []
    # Form windows within each trajectory, then batch the completed windows
    # in run/condition order. Batching never changes a window's observations.
    windows = np.concatenate([next(trajectory_batches(process, scaler, len(process)))[1]
                              for process in test['process']])
    x = torch.as_tensor(windows, device=device)
    with torch.no_grad():
        parts = [model(x[start:start+batch_size]).detach().cpu().numpy()
                 for start in range(0, len(x), batch_size)]
        predictions = scaler.inverse_y(np.concatenate(parts)).reshape(len(test['process']), -1)
        for prediction, quality, condition, run in zip(
                predictions, test['quality'], test['condition'], test['run']):
            error = prediction.astype(np.float64) - quality[19:, 0]
            if not np.isfinite(error).all():
                raise RuntimeError('Nonfinite test predictions')
            target_range = float(np.ptp(quality[:, 0]))
            if target_range <= 0:
                raise ValueError('Range-normalized errors require a positive target range')
            mae = float(np.abs(error).mean())
            rmse = float(np.sqrt(np.mean(error**2)))
            for unit, divisor in zip(UNITS, (target_range, 1.)):
                rows.append(dict(run=int(run), condition='D%02d' % condition,
                                 unit=unit, endpoints=len(error), target_range=target_range,
                                 mae=mae/divisor, rmse=rmse/divisor))
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped(rows, keys):
    values = defaultdict(list)
    for row in rows:
        values[tuple(row[k] for k in keys)].append(row)
    return values


def add_condition_averages(rows):
    """Average per-condition metrics equally within each testing run."""
    result = list(rows)
    keys = ('scenario', 'seed', 'method', 'model', 'unit', 'run')
    for group, cells in grouped(rows, keys).items():
        scenario = group[0]
        chosen = cells if scenario == 's1' else [r for r in cells if r['condition'] != 'D00']
        expected = {'D00', 'D01', 'D02', 'D03'} if scenario == 's1' else {'D01', 'D02', 'D03'}
        if {r['condition'] for r in chosen} != expected:
            raise ValueError('Incomplete conditions when computing run averages')
        row = dict(zip(keys, group))
        row.update(condition='all_conditions' if scenario == 's1' else 'faults',
                   endpoints=sum(r['endpoints'] for r in chosen), target_range='',
                   mae=float(np.mean([r['mae'] for r in chosen])),
                   rmse=float(np.mean([r['rmse'] for r in chosen])))
        result.append(row)
    return result


def summarize(rows, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = add_condition_averages(rows)
    # Fix field order for both measured conditions and condition averages.
    fields = ('scenario', 'seed', 'method', 'model', 'condition', 'run',
              'unit', 'endpoints', 'target_range', 'mae', 'rmse')
    rows = [{key: row[key] for key in fields} for row in rows]
    write_csv(output / 'metrics_by_run.csv', rows)
    keys = ('scenario', 'seed', 'method', 'model', 'condition', 'unit')
    seeds = []
    for group, cells in grouped(rows, keys).items():
        row = dict(zip(keys, group))
        row['testing_runs'] = len(cells)
        for metric in ('mae', 'rmse'):
            a = np.asarray([c[metric] for c in cells])
            row[metric] = float(a.mean())
            row[metric + '_sd_across_runs'] = float(a.std(ddof=1)) if len(a) > 1 else ''
        seeds.append(row)
    write_csv(output / 'metrics_by_seed.csv', seeds)
    keys = ('scenario', 'method', 'model', 'condition', 'unit')
    summary = []
    for group, cells in grouped(seeds, keys).items():
        row = dict(zip(keys, group))
        row['seeds'] = len(cells)
        for metric in ('mae', 'rmse'):
            a = np.asarray([c[metric] for c in cells])
            row[metric + '_mean'] = float(a.mean())
            row[metric + '_sd_across_seeds'] = float(a.std(ddof=1)) if len(a) > 1 else ''
        summary.append(row)
    write_csv(output / 'summary.csv', summary)
    make_boxplots(rows, output)
    make_default_preview(rows, output)
    return summary


def make_default_preview(rows, output):
    """Preview Scenario 1 / PI-LSTM / D01 for the first requested seed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cells = [r for r in rows if r['scenario'] == 's1' and r['model'] == 'pilstm'
             and r['condition'] == 'D01' and r['unit'] == UNITS[0]]
    if not cells:
        return
    seed = cells[0]['seed']
    cells = [r for r in cells if r['seed'] == seed]
    methods = [m for m in METHOD_NAMES if any(r['method'] == m for r in cells)]
    values = [[r['rmse'] for r in cells if r['method'] == m] for m in methods]
    for method in methods:
        runs = [int(r['run']) for r in cells if r['method'] == method]
        if sorted(runs) != list(range(31, 61)):
            raise ValueError('Preview requires the same 30 testing runs for each method')
    colors = {'raw': '#8A8A8A', 'pclaf': '#0072B2', 'vae': '#E69F00', 'gan': '#009E73'}
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        fig, ax = plt.subplots(figsize=(5.0, 4.2))
        bp = ax.boxplot(values, patch_artist=True, showmeans=True, widths=.58,
                        medianprops={'color': 'black'},
                        meanprops={'marker': 'D', 'markerfacecolor': '#A82424',
                                   'markeredgecolor': '#A82424', 'markersize': 4},
                        flierprops={'marker': '.', 'markersize': 3})
        for patch, method in zip(bp['boxes'], methods):
            patch.set_facecolor(colors[method])
            patch.set_alpha(.65)
        ax.set_xticks(np.arange(1, len(methods) + 1))
        ax.set_xticklabels([METHOD_NAMES[m] for m in methods])
        ax.set_title('PI-LSTM / D01')
        ax.set_ylabel('RMSE (normalized)')
        ax.grid(axis='y', alpha=.18)
        fig.text(.5, .015, 'Scenario 1 | seed %s | 30 testing runs' % seed,
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .055, 1, 1))
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        for suffix in ('png', 'pdf', 'svg'):
            fig.savefig(str(output / ('preview.' + suffix)), dpi=300, bbox_inches='tight')
        plt.close(fig)


def make_boxplots(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    colors = {'raw': '#8A8A8A', 'pclaf': '#0072B2', 'vae': '#E69F00', 'gan': '#009E73'}
    # Each seed has its own plot: 30 runs, without pooling repeat/run variation.
    for (scenario, seed, unit), cells in grouped(rows, ('scenario', 'seed', 'unit')).items():
        models = [m for m in MODEL_NAMES if any(r['model'] == m for r in cells)]
        methods = [m for m in METHOD_NAMES if any(r['method'] == m for r in cells)]
        conditions = ['D00', 'D01', 'D02', 'D03'] if scenario == 's1' else ['faults']
        for metric in ('rmse', 'mae'):
            fig, axes = plt.subplots(len(conditions), len(models), squeeze=False,
                                     figsize=(3.3*len(models), 2.7*len(conditions)),
                                     sharey='row')
            for i, condition in enumerate(conditions):
                for j, model in enumerate(models):
                    ax = axes[i, j]
                    values = [[r[metric] for r in cells if r['condition'] == condition
                               and r['model'] == model and r['method'] == method]
                              for method in methods]
                    if any(len(a) != 30 for a in values):
                        raise ValueError('Each box must contain the same 30 testing runs')
                    bp = ax.boxplot(values,
                                    patch_artist=True, showmeans=True, widths=.58,
                                    medianprops={'color': 'black'},
                                    meanprops={'marker': 'D', 'markerfacecolor': '#A82424',
                                               'markeredgecolor': '#A82424', 'markersize': 4},
                                    flierprops={'marker': '.', 'markersize': 3})
                    for patch, method in zip(bp['boxes'], methods):
                        patch.set_facecolor(colors[method])
                        patch.set_alpha(.65)
                    ax.set_xticks(np.arange(1, len(methods) + 1))
                    ax.set_xticklabels([METHOD_NAMES[m] for m in methods])
                    ax.set_title(MODEL_NAMES[model] + ' / ' + condition)
                    ax.tick_params(axis='x', labelrotation=25)
                    ax.grid(axis='y', alpha=.18)
                    if j == 0:
                        ax.set_ylabel(metric.upper() + (' (normalized)' if unit == UNITS[0]
                                                       else ' (mole percentage points)'))
            fig.tight_layout()
            name = '%s_seed%d_%s_%s' % (scenario, seed, metric, unit)
            for suffix in ('png', 'pdf', 'svg'):
                fig.savefig(str(output / (name + '.' + suffix)), dpi=300, bbox_inches='tight')
            plt.close(fig)
