"""Fixed-update training with matched real-sample and dropout random streams.

Input pairs are already normalized by the supplied training scaler. Each batch
contains 32 samples drawn with replacement, equally across observed conditions.
Generated pairs use an independent sampling and dropout stream. The trainer
uses only fitting pairs and returns the final model after the requested updates.
"""
import copy
from contextlib import contextmanager
import hashlib
import math
import numbers
import time

import numpy as np
import torch
from torch import nn

from .models import build_model
from .physics import ObservableTEPhysics


# Stable namespace for reproducible, purpose-specific random streams.
RNG_VERSION = 'legacy_endpoint_training:v1'


def _seed(seed, purpose, update=0):
    value = '{}:{}:{}:{}'.format(RNG_VERSION, seed, purpose, update)
    return int.from_bytes(hashlib.sha256(value.encode('ascii')).digest()[:4], 'little')


@contextmanager
def _torch_stream(seed, device):
    """Use a local forward-pass RNG and restore the caller's random state."""
    devices = [] if device.type == 'cpu' else [device.index]
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        if device.type == 'cuda':
            generator = torch.Generator(device=device).manual_seed(seed)
            torch.cuda.set_rng_state(generator.get_state(), device)
        yield


def _state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update((name + ':' + str(value.dtype) + ':' + str(tuple(value.shape))).encode('ascii'))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _arrays(value, name):
    if isinstance(value, dict):
        x, y, condition = (np.asarray(value[key]) for key in ('x', 'y', 'condition'))
    else:
        x, y, condition = np.asarray(value.x), np.asarray(value.y), np.asarray(value.condition)
    if (x.ndim != 3 or x.shape[1:] != (20, 23) or len(x) == 0
            or y.shape != (len(x), 1) or condition.shape != (len(x),)
            or not np.issubdtype(condition.dtype, np.integer)
            or not set(np.unique(condition)).issubset({0, 1, 2, 3})
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError(name + ' requires finite normalized (N,20,23)/(N,1) pairs and TE condition labels 0-3')
    x, y = x.astype(np.float32), y.astype(np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(name + ' overflows float32 storage')
    return x, y, condition.copy()


def _draw(condition, conditions, rng):
    per_condition = 32 // len(conditions)
    return np.concatenate([
        rng.choice(np.flatnonzero(condition == c), per_condition, replace=True)
        for c in conditions]).astype(np.int64)


def _fixed_physics(physics, x):
    """Cache input-dependent physical terms, retaining prediction gradients."""
    with torch.no_grad():
        result = physics._evaluate(x)
    return (result['proxy_molpercent'].detach(),
            (physics.domain_weight * result['domain_penalty']).detach())


def _physical_loss(cached, indices, prediction, physics):
    target = (prediction.to(torch.float64) * physics.y_std.double()
              + physics.y_mean.double()).reshape(-1)
    error = (cached[0][indices] - target) / physics.y_std.double()[0]
    return (error.square() + cached[1][indices]).mean()


def _validate_scaler(scaler, physics):
    if not isinstance(physics, ObservableTEPhysics):
        raise TypeError('physics must be an ObservableTEPhysics instance')
    description = {}
    for name, size in (('x_mean', 23), ('x_std', 23), ('y_mean', 1), ('y_std', 1)):
        value = getattr(scaler, name)
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float64)
        if (value.shape != (size,) or not np.isfinite(value).all()
                or (name.endswith('std') and np.any(value <= 0))):
            raise ValueError('Invalid training scaler ' + name)
        actual = getattr(physics, name).detach().cpu().numpy()
        if not np.array_equal(value, actual):
            raise ValueError('Physics and training scaler disagree on ' + name)
        description[name] = value.tolist()
    return description


def _config(config):
    supplied = copy.deepcopy(dict(config))
    defaults = dict(model={}, rho=0., batch_size=32, synthetic_weight=.25,
                    condition_balanced=True, weight_decay=0., gradient_clip=5.,
                    eval_interval=10)
    if 'learning_rate' not in supplied:
        raise ValueError('learning_rate must be declared explicitly')
    if set(supplied) - set(defaults) - {'learning_rate'}:
        raise ValueError('Unknown training configuration field: ' + str(
            sorted(set(supplied) - set(defaults) - {'learning_rate'})))
    defaults.update(supplied)
    for name in ('batch_size', 'eval_interval'):
        value = defaults[name]
        if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value < 1:
            raise ValueError(name + ' must be a positive integer')
    for name in ('learning_rate', 'rho', 'synthetic_weight', 'weight_decay', 'gradient_clip'):
        value = defaults[name]
        if (not isinstance(value, numbers.Real) or isinstance(value, bool)
                or not math.isfinite(value) or value < 0):
            raise ValueError(name + ' must be finite and nonnegative')
    if defaults['learning_rate'] == 0 or defaults['gradient_clip'] == 0:
        raise ValueError('learning_rate and gradient_clip must be positive')
    if defaults['batch_size'] != 32 or defaults['condition_balanced'] is not True:
        raise ValueError('Training uses condition-balanced batches of 32')
    return defaults


def fit_predictor(real, generated, scaler, physics, kind, config, seed,
                  fixed_updates, device='cpu'):
    """Return ``(model, metadata)`` after exactly ``fixed_updates``.

    ``real`` and optional ``generated`` are dictionaries containing normalized
    ``x`` (N,20,23), normalized ``y`` (N,1), and integer ``condition`` (N,).
    ``config['rho']`` weights physical residuals on both real and synthetic
    pairs, independently of the generator. LSTM and PI-LSTM share a backbone.
    The returned model is in evaluation mode and predicts normalized targets.
    """
    start = time.perf_counter()
    kind = str(kind).lower()
    if kind not in ('pilstm', 'lstm', 'cnn', 'transformer'):
        raise ValueError('Unknown predictor kind')
    if not isinstance(seed, numbers.Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if (not isinstance(fixed_updates, numbers.Integral)
            or isinstance(fixed_updates, bool) or fixed_updates < 1):
        raise ValueError('fixed_updates must be a positive integer')
    config = _config(config)
    rx, ry, rc = _arrays(real, 'Real fitting data')
    conditions = tuple(int(c) for c in np.unique(rc))
    if 32 % len(conditions):
        raise ValueError('Observed condition count must divide the batch size of 32')
    if generated is not None:
        sx, sy, sc = _arrays(generated, 'Generated bank')
        if tuple(int(c) for c in np.unique(sc)) != conditions:
            raise ValueError('Real and generated pairs must cover the same conditions')
    scaler_description = _validate_scaler(scaler, physics)
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only CPU and CUDA execution are supported')
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    torch.set_num_threads(1)
    real_rng = np.random.RandomState(_seed(seed, 'real_indices'))
    synthetic_rng = np.random.RandomState(_seed(seed, 'synthetic_indices'))
    with _torch_stream(_seed(seed, 'initialization'), device):
        model = build_model(kind, config=config['model']).to(device)
    config['model'] = copy.deepcopy(model.predictor_config)
    physics = copy.deepcopy(physics).to(device)
    convert = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    rx, ry = convert(rx), convert(ry)
    if generated is not None:
        sx, sy = convert(sx), convert(sy)
    rho = float(config['rho'])
    weight = 0. if generated is None else float(config['synthetic_weight'])
    real_fixed_physics = _fixed_physics(physics, rx) if rho else None
    synthetic_fixed_physics = _fixed_physics(physics, sx) if rho and generated is not None else None
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    initial_state_hash = _state_hash(model.state_dict())
    history, rng_trace = [], []
    accum, since_log = np.zeros(5, dtype=np.float64), 0
    for update in range(1, int(fixed_updates) + 1):
        model.train()
        ri = _draw(rc, conditions, real_rng)
        real_seed = _seed(seed, 'real_dropout', update)
        synthetic_seed = _seed(seed, 'synthetic_dropout', update)
        ri_device = torch.as_tensor(ri, dtype=torch.long, device=device)
        with _torch_stream(real_seed, device):
            rp = model(rx[ri_device])
        real_mse = (rp - ry[ri_device]).square().mean()
        real_physics = _physical_loss(real_fixed_physics, ri_device, rp, physics) if rho else rp.new_zeros(())
        synthetic_mse, synthetic_physics = rp.new_zeros(()), rp.new_zeros(())
        trace = dict(update=update, real_indices_sha256=hashlib.sha256(ri.tobytes()).hexdigest(),
                     real_dropout_seed=real_seed, synthetic_indices_sha256=None,
                     synthetic_dropout_seed=None)
        if generated is not None:
            si = _draw(sc, conditions, synthetic_rng)
            si_device = torch.as_tensor(si, dtype=torch.long, device=device)
            with _torch_stream(synthetic_seed, device):
                sp = model(sx[si_device])
            synthetic_mse = (sp - sy[si_device]).square().mean()
            synthetic_physics = _physical_loss(synthetic_fixed_physics, si_device, sp, physics) if rho else sp.new_zeros(())
            trace.update(synthetic_indices_sha256=hashlib.sha256(si.tobytes()).hexdigest(),
                         synthetic_dropout_seed=synthetic_seed)
        loss = real_mse + rho * real_physics + weight * (synthetic_mse + rho * synthetic_physics)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite objective at update {}'.format(update))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError('Nonfinite gradient norm at update {}'.format(update))
        optimizer.step()
        rng_trace.append(trace)
        accum += [real_mse.item(), real_physics.item(), synthetic_mse.item(),
                  synthetic_physics.item(), loss.item()]
        since_log += 1
        if update % config['eval_interval'] and update != fixed_updates:
            continue
        row = dict(zip(('real_mse', 'real_physics', 'synthetic_mse', 'synthetic_physics', 'loss'),
                       (accum / since_log).tolist()))
        row.update(update=update, n_updates_averaged=since_log,
                   effective_synthetic_weight=weight)
        history.append(row)
        accum.fill(0.)
        since_log = 0
    model.eval()
    metadata = dict(
        config=config, kind=kind, seed=int(seed), device=str(device), cpu_threads=1,
        history=history, completed_updates=update,
        checkpoint_selection='fixed_updates_last_state',
        parameter_count=sum(p.numel() for p in model.parameters()),
        elapsed_seconds=time.perf_counter() - start, real_count=len(rx),
        synthetic_count=0 if generated is None else len(sx),
        effective_synthetic_weight=weight, scaler=scaler_description,
        conditions=list(conditions), samples_per_condition=32 // len(conditions),
        initial_state_sha256=initial_state_hash, returned_state_sha256=_state_hash(model.state_dict()),
        rng=dict(version=RNG_VERSION,
                 initialization_seed=_seed(seed, 'initialization'),
                 real_index_seed=_seed(seed, 'real_indices'),
                 synthetic_index_seed=_seed(seed, 'synthetic_indices')),
        rng_trace=rng_trace)
    return model, metadata
