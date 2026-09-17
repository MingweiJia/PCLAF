"""Joint conditional VAE, posterior diffusion, and ancestral latent sampling."""
import copy
import math
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("Non-finite {}".format(name))


def _physics(physics, x, y, require_samples=False):
    if physics is None:
        return x.new_zeros(len(x))
    value = physics.residual(x, y) if hasattr(physics, "residual") else physics(x, y)
    if value.ndim == 0:
        if require_samples:
            raise ValueError("Physical guidance requires per-sample residuals [B].")
        value = value.reshape(1)
    else:
        value = value.reshape(len(x), -1).mean(dim=1)
    _finite(value, "physical penalty")
    if bool((value < -1e-7).any()):
        raise ValueError("Physical penalty must be nonnegative, not a signed mean.")
    return value


class JointTransform(nn.Module):
    """Standardize using only supplied fit pairs; return their original scale."""

    def __init__(self, x, y):
        super().__init__()
        self.window = x.shape[1]
        self.features = x.shape[2]
        self.register_buffer("x_mean", x.mean(dim=(0, 1), keepdim=True))
        self.register_buffer("x_scale", x.std(dim=(0, 1), unbiased=False, keepdim=True).clamp_min(1e-6))
        self.register_buffer("y_mean", y.mean(dim=0, keepdim=True))
        self.register_buffer("y_scale", y.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6))

    @property
    def joint_dim(self):
        return self.window * self.features + 1

    def encode(self, x, y):
        return torch.cat((((x - self.x_mean) / self.x_scale).reshape(len(x), -1),
                          (y - self.y_mean) / self.y_scale), dim=1)

    def decode(self, pair):
        x = pair[:, :-1].reshape(-1, self.window, self.features) * self.x_scale + self.x_mean
        y = pair[:, -1:] * self.y_scale + self.y_mean
        return x, y

    def description(self):
        return {
            "fit_scope": "training pairs supplied to fit_generate only",
            "x_mean": self.x_mean.flatten().detach().cpu().tolist(),
            "x_scale": self.x_scale.flatten().detach().cpu().tolist(),
            "y_mean": self.y_mean.flatten().detach().cpu().tolist(),
            "y_scale": self.y_scale.flatten().detach().cpu().tolist(),
            "return_scale": "same normalized scale as caller's input arrays",
        }


class LatentDenoiser(nn.Module):
    def __init__(self, latent_dim, hidden_dim, condition_count, time_dim=32, diffusion_steps=100):
        super().__init__()
        self.time_dim = time_dim
        self.context = nn.Embedding(condition_count, 8)
        self.net = nn.Sequential(nn.Linear(latent_dim + time_dim + 8, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, latent_dim))
        # For standardized Gaussian data the exact epsilon predictor is sigma*z.
        # Learn a correction that vanishes at the all-noise endpoint; this avoids
        # amplifying approximation error when the final cosine beta is near 1.
        _, _, retention, _ = cosine_schedule(diffusion_steps, "cpu")
        self.register_buffer("signal", retention.sqrt())
        self.register_buffer("noise", (1.0 - retention).sqrt())
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z, step, condition):
        frequencies = torch.exp(torch.arange(self.time_dim // 2, device=z.device, dtype=z.dtype)
                                * (-math.log(10000.0) / (self.time_dim // 2 - 1)))
        angles = step.to(z.dtype).unsqueeze(1) * frequencies.unsqueeze(0)
        times = torch.cat((angles.sin(), angles.cos()), dim=1)
        correction = self.net(torch.cat((z, times, self.context(condition)), dim=1))
        return self.noise[step - 1, None] * z + self.signal[step - 1, None] * correction


def cosine_schedule(steps, device):
    """Zero-based tensors represent diffusion steps k=1,...,Kd."""
    points = torch.linspace(0, steps, steps + 1, dtype=torch.float64, device=device)
    cumulative = torch.cos(((points / steps + 0.008) / 1.008) * math.pi / 2).square()
    cumulative = cumulative / cumulative[0]
    betas = (1.0 - cumulative[1:] / cumulative[:-1]).clamp(1e-8, 0.999).float()
    alphas = 1.0 - betas
    alpha_bar = alphas.cumprod(0)
    previous = torch.cat((alpha_bar.new_ones(1), alpha_bar[:-1]))
    posterior_var = betas * (1.0 - previous) / (1.0 - alpha_bar)
    return betas, alphas, alpha_bar, posterior_var


def _fit_ddpm(vae, pair, condition, condition_count, config):
    with torch.no_grad():
        means, logvars = vae.encode(pair)
        # Aggregate posterior moments, not validation or test latent statistics.
        latent_mean = means.mean(dim=0, keepdim=True)
        latent_scale = (means.var(dim=0, unbiased=False, keepdim=True)
                        + logvars.exp().mean(dim=0, keepdim=True)).sqrt().clamp_min(1e-6)
    model = LatentDenoiser(config["latent_dim"], config["hidden_dim"], condition_count,
                          diffusion_steps=config["diffusion_steps"]).to(pair.device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["ddpm_lr"])
    _, _, alpha_bar, _ = cosine_schedule(config["diffusion_steps"], pair.device)
    history = []
    running = 0.0
    last_log = 0
    for update in range(int(config["ddpm_updates"])):
        indices = torch.randint(len(pair), (min(config["batch_size"], len(pair)),), device=pair.device)
        # Fresh posterior draws preserve q(z|X,y)'s variance in diffusion training.
        z = means[indices] + (0.5 * logvars[indices]).exp() * torch.randn_like(means[indices])
        z = (z - latent_mean) / latent_scale
        steps = torch.randint(config["diffusion_steps"], (len(indices),), device=pair.device)
        noise = torch.randn_like(z)
        retention = alpha_bar[steps].unsqueeze(1)
        noisy = retention.sqrt() * z + (1.0 - retention).sqrt() * noise
        predicted_noise = model(noisy, steps + 1, condition[indices])
        loss = F.mse_loss(predicted_noise, noise)
        _finite(loss, "DDPM noise loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        decay = min(float(config["ema_decay"]), float(update + 1) / float(update + 10))
        with torch.no_grad():
            for target, source in zip(ema.parameters(), model.parameters()):
                target.mul_(decay).add_(source, alpha=1.0 - decay)
        running += float(loss.item())
        if (update + 1) % config["log_every"] == 0 or update + 1 == config["ddpm_updates"]:
            history.append({"update": update + 1, "noise_mse": running / (update + 1 - last_log)})
            last_log = update + 1
            running = 0.0
    return ema, model, latent_mean, latent_scale, history


@torch.no_grad()
def sample_latent_ddpm(denoiser, condition, latent_dim, diffusion_steps):
    """Ancestral DDPM using the fixed forward posterior covariance."""
    betas, alphas, alpha_bar, posterior_var = cosine_schedule(diffusion_steps, condition.device)
    z = torch.randn(len(condition), latent_dim, device=condition.device)
    for index in range(diffusion_steps - 1, -1, -1):
        steps = torch.full((len(z),), index + 1, device=z.device, dtype=torch.long)
        predicted_noise = denoiser(z, steps, condition)
        mean = (z - betas[index] / (1.0 - alpha_bar[index]).sqrt() * predicted_noise) / alphas[index].sqrt()
        z = mean if index == 0 else mean + posterior_var[index].sqrt() * torch.randn_like(z)
        _finite(z, "reverse diffusion latent")
    return z


class ConditionalJointVAE(nn.Module):
    """One q(z|X,y,c), G(z,c) for every observed fitting condition."""
    def __init__(self, joint_dim, hidden_dim, latent_dim, condition_count):
        super().__init__()
        self.context = nn.Embedding(condition_count, 8)
        self.encoder = nn.Sequential(nn.Linear(joint_dim + 8, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim + 8, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, joint_dim))

    def encode(self, pair, condition):
        hidden = self.encoder(torch.cat((pair, self.context(condition)), dim=1))
        return self.mean(hidden), self.logvar(hidden).clamp(-12.0, 8.0)

    def decode(self, latent, condition):
        return self.decoder(torch.cat((latent, self.context(condition)), dim=1))

    def forward(self, pair, condition):
        mean, logvar = self.encode(pair, condition)
        latent = mean + (0.5 * logvar).exp() * torch.randn_like(mean)
        return self.decode(latent, condition), mean, logvar


def _fit_vae(pair, condition, condition_count, transform, physics, config):
    vae = ConditionalJointVAE(pair.shape[1], config['hidden_dim'], config['latent_dim'],
                              condition_count).to(pair.device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=config['vae_lr'])
    history = []
    for epoch in range(config['vae_epochs']):
        vae.train()
        order = torch.randperm(len(pair), device=pair.device)
        totals = np.zeros(5, dtype=np.float64)
        for start in range(0, len(pair), config['batch_size']):
            indices = order[start:start + config['batch_size']]
            original = pair[indices]
            reconstructed, mean, logvar = vae(original, condition[indices])
            x_loss = F.mse_loss(reconstructed[:, :-1], original[:, :-1])
            y_loss = F.mse_loss(reconstructed[:, -1:], original[:, -1:])
            kl = -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).mean()
            x_hat, y_hat = transform.decode(reconstructed)
            physical = _physics(physics, x_hat, y_hat).mean()
            loss = (x_loss + config['lambda_y'] * y_loss + config['beta_kl'] * kl
                    + config['lambda_phys'] * physical)
            _finite(loss, 'conditional VAE loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), 10.0)
            optimizer.step()
            totals += len(indices) * np.array([loss.item(), x_loss.item(), y_loss.item(),
                                               kl.item(), physical.item()])
        history.append(dict(zip(('loss', 'x_reconstruction', 'y_reconstruction', 'kl',
                                 'physics'), (totals / len(pair)).tolist()), epoch=epoch + 1))
    vae.eval()
    return vae, history


class _ConditionalPosterior:
    """Bind observed context for the core DDPM's one full fitting-set encode."""
    def __init__(self, vae, condition):
        self.vae, self.condition = vae, condition

    def encode(self, pair):
        if len(pair) != len(self.condition):
            raise ValueError('DDPM posterior context and fitting pairs differ')
        return self.vae.encode(pair, self.condition)


def _state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
