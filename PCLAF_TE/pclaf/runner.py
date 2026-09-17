import argparse
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .config import DEFAULT_SEEDS, default_config, merge_config
from .data import load_data, training_data
from .generators import fit_generate, resolve_config
from .physics import ObservableTEPhysics
from .reporting import evaluate, summarize, write_csv
from .training import fit_predictor


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config, scenarios):
    methods = config['methods']
    if not isinstance(methods, list) or not methods or len(set(methods)) != len(methods):
        raise ValueError('methods must be a nonempty list without duplicates')
    if set(methods) - {'raw', 'pclaf', 'vae', 'gan'}:
        raise ValueError('Unknown augmentation method')
    for key in ('labels', 'synthetic_ratio'):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    for scenario in scenarios:
        values = config['scenarios'][scenario]
        models = values['models']
        if not models or len(set(models)) != len(models) or set(models) - set(config['predictors']):
            raise ValueError('Unknown or duplicate predictor in ' + scenario)
        for model in models:
            updates = values['updates'].get(model)
            if isinstance(updates, bool) or not isinstance(updates, int) or updates < 1:
                raise ValueError('Positive update count required for ' + scenario + '/' + model)
    for method in methods:
        if method != 'raw':
            resolve_config(method, config['generators'][method])


def run(config, scenarios, seeds, data_dir, output, device, eval_batch_size,
        predictor_device='cpu'):
    validate_config(config, scenarios)
    train, test = load_data(data_dir)
    # Resolve subset validity before creating a run directory or starting a fit.
    for scenario in scenarios:
        training_data(train, scenario, config['labels'], seeds[0])
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty output directory: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda:0')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    write_json(output / 'configuration.json', dict(
        config=config, scenarios=scenarios, seeds=seeds, device=str(device),
        predictor_device=str(predictor_device),
        numpy_version=np.__version__, torch_version=torch.__version__,
        evaluation_batch_size=eval_batch_size,
        backend=dict(cpu_threads=1, cudnn_benchmark=False, cudnn_deterministic=True,
                     matmul_allow_tf32=False, cudnn_allow_tf32=True),
        data_sha256={name: file_hash(Path(data_dir) / name) for name in ('train.npz', 'test.npz')},
    ))
    all_rows = []
    for scenario in scenarios:
        for seed in seeds:
            directory = output / scenario / ('seed%d' % seed)
            directory.mkdir(parents=True)
            real, scaler, indices = training_data(train, scenario, config['labels'], seed)
            write_json(directory / 'training_subset.json', dict(
                indices=indices.tolist(), run=train['run'][indices].tolist(),
                condition=real['condition'].tolist(),
                end_index=train['end_index'][indices].tolist(), scaler=scaler.to_dict()))
            banks = {'raw': None}
            for method in config['methods']:
                if method == 'raw':
                    continue
                print('%s seed%d: training %s generator' % (scenario, seed, method), flush=True)
                counts = {int(c): config['synthetic_ratio'] * int((real['condition'] == c).sum())
                          for c in np.unique(real['condition'])}
                gx, gy, gc, _ = fit_generate(
                    method, real['x'], real['y'], real['condition'], counts,
                    ObservableTEPhysics(scaler), config['generators'][method], seed,
                    str(device), directory / 'generators' / method, scaler=scaler)
                if len(gy) != config['synthetic_ratio'] * len(real['y']):
                    raise RuntimeError('Generated bank size mismatch')
                banks[method] = dict(x=gx, y=gy, condition=gc)
            for kind in config['scenarios'][scenario]['models']:
                reference = None
                for method in config['methods']:
                    print('%s seed%d: %s + %s' % (scenario, seed, method, kind), flush=True)
                    model, metadata = fit_predictor(
                        real, banks[method], scaler, ObservableTEPhysics(scaler), kind,
                        copy.deepcopy(config['predictors'][kind]), seed,
                        config['scenarios'][scenario]['updates'][kind], predictor_device)
                    destination = directory / 'predictors' / kind / method
                    destination.mkdir(parents=True)
                    write_json(destination / 'training.json', metadata)
                    # Check the paired controls directly, rather than relying on filenames.
                    paired = dict(initial_state_hash=metadata['initial_state_sha256'],
                                  real_random_stream=[(r['real_indices_sha256'], r['real_dropout_seed'])
                                                      for r in metadata['rng_trace']])
                    if reference is None:
                        reference = paired
                    elif paired != reference:
                        raise RuntimeError('Raw/augmentation controls differ within a predictor')
                    torch.save(dict(state_dict=model.state_dict(), kind=kind,
                                    config=config['predictors'][kind], scaler=scaler.to_dict()),
                               str(destination / 'predictor.pt'))
                    model.to(device)
                    cells = evaluate(model, test, scaler, str(device), eval_batch_size)
                    for row in cells:
                        row.update(scenario=scenario, seed=seed, method=method, model=kind)
                    write_csv(destination / 'metrics.csv', cells)
                    all_rows.extend(cells)
                    del model
            del banks
    result = summarize(all_rows, output / 'reports')
    write_json(output / 'completion.json', dict(status='complete',
               scenarios=scenarios, seeds=seeds, summary_rows=len(result)))
    print('Completed: ' + str(output / 'reports'), flush=True)
    if (output / 'reports' / 'preview.png').exists():
        print('Preview: ' + str(output / 'reports' / 'preview.png'), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train TE comparisons and export metrics and boxplots.')
    parser.add_argument('--scenario', choices=('s1', 's2', 'both'), default='both')
    parser.add_argument('--seeds', type=int, nargs='+',
                        help='Shared random seeds for all methods; defaults to the 10 configured seeds.')
    parser.add_argument('--config', type=Path, help='JSON overrides of the shared default configuration.')
    parser.add_argument('--write-config', type=Path, help='Write default configuration and exit.')
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data' / 'te')
    parser.add_argument('--output', type=Path, help='Empty directory; defaults to outputs/<timestamp>.')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Generator training and evaluation device.')
    parser.add_argument('--predictor-device', default='cpu',
                        help='Predictor training device; defaults to cpu.')
    parser.add_argument('--eval-batch-size', type=int, default=512)
    args = parser.parse_args(argv)
    config = default_config()
    if args.config:
        config = merge_config(config, json.loads(args.config.read_text(encoding='utf-8')))
    if args.write_config:
        write_json(args.write_config, config)
        print('Configuration: ' + str(args.write_config.resolve()))
        return
    seeds = args.seeds if args.seeds is not None else DEFAULT_SEEDS
    if not seeds or any(s < 0 or s >= 2**32 for s in seeds) or len(set(seeds)) != len(seeds):
        parser.error('seeds must be distinct integers between 0 and 2**32-1')
    if args.eval_batch_size < 1:
        parser.error('eval-batch-size must be positive')
    scenarios = ['s1', 's2'] if args.scenario == 'both' else [args.scenario]
    output = args.output or Path('outputs') / datetime.now().strftime('%Y%m%d_%H%M%S')
    run(config, scenarios, seeds, args.data_dir, output, args.device, args.eval_batch_size,
        args.predictor_device)
