"""Train joint TE generators from fitting pairs and sample new endpoint pairs.

Inputs and returned samples use the caller's training standardization. PCLAF
shares one conditional VAE/DDPM across observed conditions; the convolutional
VAE and BCE-GAN fit one model per condition. No data files are read here.
"""
import json
import math
from numbers import Integral
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import generation_core as core
from .baseline_networks import VAE19Channel, Generator, Discriminator
from .generation_core import _finite
from .guidance import DEFAULT_GUIDANCE_CONFIG, guide_standard_latent, resolve_guidance_config


QUALITY = 12
PROCESS_CHANNELS = tuple(range(12)) + tuple(range(13, 24))
MIN_RANGE = 1e-6
SAMPLE_SEED_OFFSET = 810000
GENERATION_CHUNK = 512
DEFAULT_PCLAF_CONFIG = dict(
    latent_dim=16, hidden_dim=128, batch_size=64, vae_epochs=250,
    vae_lr=.001, lambda_y=1., beta_kl=.01, lambda_phys=.1,
    diffusion_steps=100, ddpm_updates=1500, ddpm_lr=.001,
    ema_decay=.995, log_every=100, **DEFAULT_GUIDANCE_CONFIG)
BASELINE_CONFIGS = {
    'vae': dict(value_range='zero_one', epochs=300, latent_dim=128, batch_size=32,
                learning_rate=.005, beta_final=.1),
    'gan': dict(value_range='zero_one', epochs=300, latent_dim=100, batch_size=16,
                generator_lr=.0001, discriminator_lr=.0002,
                betas=[.5, .999], real_label=.9, fake_label=.1),
}


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError('{} must be an integer >= {}'.format(name, minimum))
    return int(value)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def resolve_config(method, config=None):
    """Resolve explicit generator settings before creating output artifacts."""
    method = str(method).lower()
    values = dict(config or {})
    sampling = dict(sample_seed_offset=SAMPLE_SEED_OFFSET, generation_chunk=GENERATION_CHUNK)
    for key in sampling:
        if key in values:
            sampling[key] = values.pop(key)
    sampling['sample_seed_offset'] = _integer(sampling['sample_seed_offset'], 'sample_seed_offset', 0)
    sampling['generation_chunk'] = _integer(sampling['generation_chunk'], 'generation_chunk')
    if method == 'pclaf':
        unknown = set(values) - set(DEFAULT_PCLAF_CONFIG)
        if unknown:
            raise ValueError('Unknown PCLAF options: ' + str(sorted(unknown)))
        result = dict(DEFAULT_PCLAF_CONFIG, **values)
        for key in ('latent_dim', 'hidden_dim', 'batch_size', 'vae_epochs',
                    'diffusion_steps', 'ddpm_updates', 'log_every'):
            result[key] = _integer(result[key], key)
        for key in ('vae_lr', 'lambda_y', 'ddpm_lr'):
            result[key] = float(result[key])
            if not math.isfinite(result[key]) or result[key] <= 0:
                raise ValueError(key + ' must be positive and finite')
        for key in ('beta_kl', 'lambda_phys'):
            result[key] = float(result[key])
            if not math.isfinite(result[key]) or result[key] < 0:
                raise ValueError(key + ' must be nonnegative and finite')
        result['ema_decay'] = float(result['ema_decay'])
        if not 0 <= result['ema_decay'] < 1:
            raise ValueError('ema_decay must be in [0,1)')
        result.update(resolve_guidance_config(result))
    elif method in BASELINE_CONFIGS:
        allowed = set(BASELINE_CONFIGS[method])
        if set(values) - allowed:
            raise ValueError('Unknown baseline options: ' + str(sorted(set(values) - allowed)))
        result = dict(BASELINE_CONFIGS[method], **values)
        if result['value_range'] not in ('zero_one', 'minus_one_one'):
            raise ValueError('value_range must be zero_one or minus_one_one')
        for key in ('epochs', 'latent_dim', 'batch_size'):
            result[key] = _integer(result[key], key)
        rates = ('learning_rate',) if method == 'vae' else ('generator_lr', 'discriminator_lr')
        for key in rates:
            result[key] = float(result[key])
            if not math.isfinite(result[key]) or result[key] <= 0:
                raise ValueError(key + ' must be positive and finite')
        if method == 'vae':
            result['beta_final'] = float(result['beta_final'])
            if not math.isfinite(result['beta_final']) or result['beta_final'] < 0:
                raise ValueError('beta_final must be nonnegative and finite')
        if method == 'gan':
            result['betas'] = [float(v) for v in result['betas']]
            if len(result['betas']) != 2 or any(not 0 <= v < 1 for v in result['betas']):
                raise ValueError('betas must contain two values in [0,1)')
            for key in ('real_label', 'fake_label'):
                result[key] = float(result[key])
                if not 0 <= result[key] <= 1:
                    raise ValueError(key + ' must be in [0,1]')
    else:
        raise ValueError('method must be pclaf, vae, or gan')
    result.update(sampling)
    return result


def _validate_inputs(x, y, condition, n_per_condition, seed):
    x, y, condition = np.asarray(x), np.asarray(y), np.asarray(condition)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if (x.ndim != 3 or x.shape[1:] != (20, 23) or len(x) < 2
            or y.shape != (len(x), 1) or condition.shape != (len(x),)
            or not np.issubdtype(x.dtype, np.floating)
            or not np.issubdtype(y.dtype, np.floating)
            or not np.issubdtype(condition.dtype, np.integer)
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or (condition < 0).any()):
        raise ValueError('Expected finite X=(N,20,23), y=(N,1), integer condition=(N,), N>=2')
    seed = _integer(seed, 'seed', 0)
    if seed >= 2 ** 32:
        raise ValueError('seed must be less than 2**32')
    ids = sorted(int(c) for c in np.unique(condition))
    counts = {_integer(k, 'condition ID', 0): _integer(v, 'generated count', 0)
              for k, v in n_per_condition.items()}
    if set(counts) - set(ids) or sum(counts.values()) < 1:
        raise ValueError('Request a positive total count only for observed fitting conditions')
    return x, y, condition.astype(np.int64, copy=False), ids, counts, seed


def mask_quality_history(pair):
    """Apply the same differentiable mask to real/fake critic inputs."""
    if pair.ndim != 3 or tuple(pair.shape[1:]) != (24, 20):
        raise ValueError('Expected a joint (N,24,20) tensor')
    mask = torch.ones((1, 24, 20), dtype=pair.dtype, device=pair.device)
    mask[:, QUALITY, :-1] = 0
    return pair * mask


def encode_condition(x, y, scaler, value_range):
    """Fit 24 min/max coordinates from observed processes and endpoint y only."""
    raw_x = np.asarray(scaler.inverse_x(x), dtype=np.float64)
    raw_y = np.asarray(scaler.inverse_y(y), dtype=np.float64).reshape(-1, 1)
    minimum = np.empty(24, dtype=np.float64)
    maximum = np.empty(24, dtype=np.float64)
    minimum[list(PROCESS_CHANNELS)] = raw_x.min(axis=(0, 1))
    maximum[list(PROCESS_CHANNELS)] = raw_x.max(axis=(0, 1))
    minimum[QUALITY], maximum[QUALITY] = raw_y.min(), raw_y.max()
    span = np.maximum(maximum - minimum, MIN_RANGE)
    pair = np.zeros((len(x), 24, 20), dtype=np.float32)
    pair[:, list(PROCESS_CHANNELS), :] = (
        (raw_x - minimum[list(PROCESS_CHANNELS)]) / span[list(PROCESS_CHANNELS)]
    ).transpose(0, 2, 1)
    pair[:, QUALITY, -1] = ((raw_y[:, 0] - minimum[QUALITY]) / span[QUALITY]).astype(np.float32)
    if value_range == 'minus_one_one':
        pair[:, list(PROCESS_CHANNELS), :] = 2 * pair[:, list(PROCESS_CHANNELS), :] - 1
        pair[:, QUALITY, -1] = 2 * pair[:, QUALITY, -1] - 1
    transform = dict(minimum=minimum.tolist(), maximum=maximum.tolist(), span=span.tolist(),
                     value_range=value_range,
                     range_floor=MIN_RANGE, quality_channel=QUALITY,
                     quality_scope='observed final endpoint only; history is masked zero')
    return pair, transform


def decode_condition(pair, transform, scaler):
    pair = np.asarray(pair, dtype=np.float64)
    if pair.ndim != 3 or pair.shape[1:] != (24, 20) or not np.isfinite(pair).all():
        raise ValueError('Invalid generated joint samples')
    if transform['value_range'] == 'minus_one_one':
        pair = (pair + 1) / 2
    minimum, span = np.asarray(transform['minimum']), np.asarray(transform['span'])
    raw_x = (pair[:, list(PROCESS_CHANNELS), :].transpose(0, 2, 1)
             * span[list(PROCESS_CHANNELS)] + minimum[list(PROCESS_CHANNELS)])
    raw_y = pair[:, QUALITY, -1:] * span[QUALITY] + minimum[QUALITY]
    return scaler.transform_x(raw_x), scaler.transform_y(raw_y)


def _networks(method, device, config=None):
    config = BASELINE_CONFIGS[method] if config is None else config
    if method == 'vae':
        model = VAE19Channel(input_channels=24, time_steps=20,
                            latent_dim=config['latent_dim'], beta=0.)
        model.device = torch.device(device)
        return model.to(device), None
    return (Generator(noise_dim=config['latent_dim'], channels=24, time_steps=20).to(device),
            Discriminator(channels=24, time_steps=20).to(device))


def _train_condition(method, pair, transform, config, device, condition):
    model, critic = _networks(method, device, config)
    epochs = config['epochs']
    loader = DataLoader(TensorDataset(torch.from_numpy(pair), torch.zeros(len(pair))),
                        batch_size=config['batch_size'], shuffle=True, drop_last=False,
                        num_workers=0)
    history = []
    if method == 'vae':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['generator_lr'], betas=tuple(config['betas']))
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config['discriminator_lr'],
                                             betas=tuple(config['betas']))
        criterion = nn.BCELoss()
    for epoch in range(1, epochs + 1):
        model.train()
        if critic is not None:
            critic.train()
        totals = {}
        steps = 0
        for data, _ in loader:
            data = data.to(device)
            if method == 'vae':
                model.beta = config['beta_final'] * epoch / epochs
                reconstruction, mu, logvar = model(data)
                reconstruction_loss = mask_quality_history(reconstruction - data).square().sum() / len(data)
                kl_loss = -.5 * (1 + logvar - mu.square() - logvar.exp()).sum() / len(data)
                objective = reconstruction_loss + model.beta * kl_loss
                _finite(objective, 'VAE objective')
                optimizer.zero_grad()
                objective.backward()
                optimizer.step()
                logged = dict(objective=float(objective.detach()),
                              reconstruction=float(reconstruction_loss.detach()),
                              kl=float(kl_loss.detach()))
            else:
                critic_optimizer.zero_grad()
                real_output = critic(mask_quality_history(data))
                real_loss = criterion(real_output, torch.full_like(real_output, config['real_label']))
                # The baseline training noise is created on CPU, then moved.
                noise = torch.randn(len(data), config['latent_dim']).to(device)
                fake = model(noise)
                fake_output = critic(mask_quality_history(fake.detach()))
                fake_loss = criterion(fake_output, torch.full_like(fake_output, config['fake_label']))
                discriminator_loss = (real_loss + fake_loss) / 2.
                _finite(discriminator_loss, 'discriminator loss')
                discriminator_loss.backward()
                critic_optimizer.step()
                optimizer.zero_grad()
                output = critic(mask_quality_history(fake))
                adversarial_loss = criterion(output, torch.full_like(output, config['real_label']))
                objective = adversarial_loss
                _finite(objective, 'GAN generator objective')
                objective.backward()
                optimizer.step()
                logged = dict(objective=float(objective.detach()),
                              adversarial=float(adversarial_loss.detach()),
                              discriminator=float(discriminator_loss.detach()))
            for name, value in logged.items():
                totals[name] = totals.get(name, 0.) + value
            steps += 1
        record = dict(epoch=epoch, batches=steps,
                      mean_losses={name: value / steps for name, value in totals.items()})
        if method == 'vae':
            record['beta'] = model.beta
        history.append(record)
        if epoch == 1 or epoch == epochs or epoch % 50 == 0:
            print('{} condition {} epoch {}/{} objective {:.6f}'.format(
                method, condition, epoch, epochs, record['mean_losses']['objective']), flush=True)
    for name, value in model.state_dict().items():
        _finite(value, 'model state ' + name)
    if critic is not None:
        for name, value in critic.state_dict().items():
            _finite(value, 'critic state ' + name)
    return model, critic, history


def sampling_latents(method, count, latent_dim=None):
    """Retain the source's sample-start-index parity, not batch-index parity."""
    latent_dim = BASELINE_CONFIGS[method]['latent_dim'] if latent_dim is None else latent_dim
    if method == 'vae':
        return torch.randn(count, latent_dim), [0], [count], [False]
    if method != 'gan' or count < 1:
        raise ValueError('Unknown generator or invalid sample count')
    batch_size = min(100, count)
    parts, starts, sizes, clipped = [], [], [], []
    for start in range(0, count, batch_size):
        size = min(batch_size, count - start)
        noise = torch.randn(size, latent_dim)
        clip = start % 2 != 0
        if clip:
            noise = torch.clamp(noise, -2, 2)
        parts.append(noise)
        starts.append(start)
        sizes.append(size)
        clipped.append(clip)
    return torch.cat(parts), starts, sizes, clipped


def _generate(method, model, latent, starts, sizes, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for start, size in zip(starts, sizes):
            noise = latent[start:start + size].to(device)
            value = model.decoder(noise) if method == 'vae' else model(noise)
            _finite(value, 'generated pair')
            parts.append(value.detach().cpu().numpy())
    model.train()  # Both baseline generation routines restore training mode.
    return np.concatenate(parts)


def _fit_pclaf(x, y, condition, ids, counts, physics, config, seed, device, target):
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Training arrays must remain finite in float32')
    core._seed(seed)
    tx, ty = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
    tc = torch.as_tensor([ids.index(int(c)) for c in condition], dtype=torch.long, device=device)
    transform = core.JointTransform(tx, ty)
    pair = transform.encode(tx, ty)
    vae, vae_history = core._fit_vae(pair, tc, len(ids), transform, physics, config)
    denoiser, online, latent_mean, latent_scale, ddpm_history = core._fit_ddpm(
        core._ConditionalPosterior(vae, tc), pair, tc, len(ids), config)
    vae.eval()
    checkpoint = dict(method='pclaf', config=config, condition_ids=ids,
                      window=20, features=23, vae=core._state(vae),
                      denoiser=core._state(denoiser), denoiser_online=core._state(online),
                      latent_mean=latent_mean.detach().cpu(), latent_scale=latent_scale.detach().cpu(),
                      transform=core._state(transform))
    for parameter in vae.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    # Sampling has its own seed and fixed per-condition chunk order.
    sample_seed = (seed + config['sample_seed_offset']) % (2 ** 32)
    core._seed(sample_seed)
    parts, traces, latents = [], [], []
    for cid in ids:
        for start in range(0, counts.get(cid, 0), config['generation_chunk']):
            size = min(config['generation_chunk'], counts[cid] - start)
            generated_condition = torch.full((size,), ids.index(cid), dtype=torch.long, device=device)
            before = core.sample_latent_ddpm(
                denoiser, generated_condition, config['latent_dim'], config['diffusion_steps'])

            def decode(value):
                return transform.decode(vae.decode(value * latent_scale + latent_mean,
                                                    generated_condition))

            after, trace = guide_standard_latent(before, decode, physics, config)
            with torch.no_grad():
                gx, gy = decode(after)
            core._finite(gx, 'generated inputs')
            core._finite(gy, 'generated endpoints')
            gc = np.full(size, cid, dtype=np.int64)
            parts.append((gx.cpu().numpy(), gy.cpu().numpy(), gc))
            latents.append((after * latent_scale + latent_mean).detach().cpu().numpy())
            traces.append(dict(trace, standard_before=before.detach().cpu().numpy(),
                               standard_after=after.detach().cpu().numpy(), condition=gc))
    if any(not torch.equal(value.cpu(), checkpoint['vae'][name])
           for name, value in vae.state_dict().items()):
        raise RuntimeError('Latent guidance changed the fixed decoder')
    torch.save(checkpoint, str(target / 'generator.pt'))
    _write_json(target / 'history.json', dict(vae=vae_history, ddpm=ddpm_history))
    trace = {key: np.concatenate([item[key] for item in traces]) for key in traces[0]}
    np.savez_compressed(str(target / 'latent_guidance.npz'), **trace)
    np.savez_compressed(str(target / 'sample_latents.npz'),
                        latent=np.concatenate(latents), condition=trace['condition'])
    diagnostics = dict(
        residual_before_mean=float(trace['before'].mean()),
        residual_after_mean=float(trace['after'].mean()),
        maximum_residual_increase=float((trace['after'] - trace['before']).max()),
        decoder_parameters_unchanged=True)
    return parts, dict(sample_seed=sample_seed, guidance=diagnostics,
                       fitting_scope='pooled conditional joint VAE and latent DDPM')


def _fit_baselines(method, x, y, condition, ids, counts, scaler, config, seed, device, target):
    parts, latents, latent_conditions = [], [], []
    condition_seeds, sample_seeds = {}, {}
    for cid in ids:
        condition_seed = (seed + 100003 * cid) % (2 ** 32)
        sample_seed = (seed + config['sample_seed_offset'] + 100003 * cid) % (2 ** 32)
        condition_seeds[str(cid)], sample_seeds[str(cid)] = condition_seed, sample_seed
        core._seed(condition_seed)
        selected = condition == cid
        pair, transform = encode_condition(x[selected], y[selected], scaler, config['value_range'])
        model, critic, history = _train_condition(method, pair, transform, config, device, cid)
        directory = target / ('condition_%d' % cid)
        directory.mkdir()
        state = dict(method=method, condition=cid, model=core._state(model))
        if critic is not None:
            state['critic'] = core._state(critic)
        torch.save(state, str(directory / 'generator.pt'))
        _write_json(directory / 'transform.json', transform)
        _write_json(directory / 'history.json', history)
        if counts.get(cid, 0):
            core._seed(sample_seed)
            latent, starts, sizes, clipped = sampling_latents(method, counts[cid], config['latent_dim'])
            if method == 'vae':
                starts = list(range(0, counts[cid], config['generation_chunk']))
                sizes = [min(config['generation_chunk'], counts[cid] - start) for start in starts]
                clipped = [False] * len(starts)
            generated = _generate(method, model, latent, starts, sizes, device)
            gx, gy = decode_condition(generated, transform, scaler)
            gc = np.full(len(gx), cid, dtype=np.int64)
            parts.append((gx, gy, gc))
            latents.append(latent.numpy())
            latent_conditions.append(gc)
            _write_json(directory / 'sampling.json', dict(seed=sample_seed, batch_starts=starts,
                                                         batch_sizes=sizes, clipped=clipped))
        del model, critic
    np.savez_compressed(str(target / 'sample_latents.npz'), latent=np.concatenate(latents),
                        condition=np.concatenate(latent_conditions))
    return parts, dict(condition_seeds=condition_seeds, sample_seeds=sample_seeds,
                       fitting_scope='independent convolutional generator per fitting condition')


def fit_generate(method, x, y, condition, n_per_condition, physics, config,
                 seed, device, outdir, scaler=None):
    """Train from scratch and return ``(X, y, condition, metadata)``.

    ``X`` has shape [N,20,23], and ``y`` contains only the final endpoint [N,1].
    ``scaler`` is required for VAE/GAN and must expose inverse_x, inverse_y,
    transform_x, and transform_y. Output files are newly trained run artifacts;
    an occupied directory is rejected. No pretrained model is required.
    """
    started = time.perf_counter()
    method = str(method).lower()
    resolved = resolve_config(method, config)
    x, y, condition, ids, counts, seed = _validate_inputs(x, y, condition, n_per_condition, seed)
    device = torch.device(device)
    if method == 'pclaf' and (physics is None or not hasattr(physics, 'residual')):
        raise ValueError('PCLAF requires a differentiable per-sample physical residual')
    if method in BASELINE_CONFIGS:
        if scaler is None or any(not callable(getattr(scaler, name, None)) for name in
                                 ('inverse_x', 'inverse_y', 'transform_x', 'transform_y')):
            raise ValueError('VAE/GAN require the fitted outer scaler')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; select device=cpu')
    target = Path(outdir)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError('Output directory is not empty: ' + str(target))
    target.mkdir(parents=True, exist_ok=True)
    if isinstance(physics, nn.Module):
        physics = physics.to(device)
    if method == 'pclaf':
        parts, details = _fit_pclaf(x, y, condition, ids, counts, physics,
                                   resolved, seed, device, target)
    else:
        parts, details = _fit_baselines(method, x, y, condition, ids, counts, scaler,
                                       resolved, seed, device, target)
    gx, gy, gc = (np.concatenate([part[index] for part in parts]) for index in range(3))
    if (gx.shape != (sum(counts.values()), 20, 23) or gy.shape != (len(gx), 1)
            or not np.isfinite(gx).all() or not np.isfinite(gy).all()):
        raise FloatingPointError('Generated sample count, shape, or finiteness check failed')
    np.savez_compressed(str(target / 'generated.npz'), x=gx, y=gy, condition=gc)
    metadata = dict(method=method, config=resolved, seed=seed,
                    fitting_counts={str(c): int((condition == c).sum()) for c in ids},
                    generated_counts={str(c): counts.get(c, 0) for c in ids},
                    generated_count=len(gx), window=20, features=23,
                    target='XMEAS(35) final endpoint; quality history is not observed',
                    output_scale='same training-standardized coordinates as input arrays',
                    seconds=time.perf_counter() - started,
                    runtime=dict(numpy=np.__version__, torch=torch.__version__, device=str(device)),
                    **details)
    _write_json(target / 'metadata.json', metadata)
    return gx, gy, gc, metadata
