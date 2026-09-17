"""Per-sample gradient correction in standardized DDPM latent coordinates."""
import numpy as np
import torch


DEFAULT_GUIDANCE_CONFIG = {
    "guidance_steps": 10,
    "guidance_lr": .05,
    "guidance_trust_radius": .5,
    "guidance_backtracks": 8,
    "guidance_tolerance": 1e-8,
    "guidance_gradient_epsilon": 1e-12,
}


def _rms(value):
    largest = value.abs().amax(dim=1)
    scale = torch.where(largest > 0, largest, torch.ones_like(largest))
    return (value / scale[:, None]).square().mean(dim=1).sqrt() * scale


def resolve_guidance_config(config=None):
    """Validate guidance options while allowing a complete generator config."""
    resolved = dict(DEFAULT_GUIDANCE_CONFIG)
    for key, value in (config or {}).items():
        if key in resolved:
            resolved[key] = value
        elif key.startswith("guidance_"):
            raise ValueError("unknown guidance option: {}".format(key))
    for key in ("guidance_steps", "guidance_backtracks"):
        value = resolved[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("{} must be a nonnegative integer".format(key))
    for key in ("guidance_lr", "guidance_trust_radius", "guidance_gradient_epsilon"):
        if not np.isfinite(resolved[key]) or resolved[key] <= 0:
            raise ValueError("{} must be positive and finite".format(key))
    if not np.isfinite(resolved["guidance_tolerance"]) or resolved["guidance_tolerance"] < 0:
        raise ValueError("guidance_tolerance must be nonnegative and finite")
    return resolved


def guide_standard_latent(z_standard, decode, physics, config=None):
    """Return corrected standard latents and auditable per-sample diagnostics.

    ``decode(v)`` must be the same fixed, evaluation-mode conditional decoder
    throughout the call and return normalized (X_window, y_endpoint) pairs.
    Each sample is decoded independently of other samples in the batch.
    The caller freezes/restores decoder parameters around this function. Only
    v is differentiated here; no optimizer, model update, sample filtering or
    label overwrite occurs. A step follows the negative gradient with initial
    RMS length ``guidance_lr``; backtracking may shorten it. Displacement from
    the initial v stays inside an RMS ball of ``guidance_trust_radius``.
    """
    resolved = resolve_guidance_config(config)
    if (not torch.is_tensor(z_standard) or z_standard.ndim != 2
            or min(z_standard.shape) <= 0 or not z_standard.is_floating_point()
            or not bool(torch.isfinite(z_standard).all())):
        raise ValueError("z_standard must be a finite nonempty floating [N,D] tensor")

    initial = z_standard.detach().clone()
    z = initial.clone()
    n = len(z)

    def evaluate(value, raw=False):
        x, y = decode(value)
        function = physics.raw_residual if raw and hasattr(physics, "raw_residual") else physics.residual
        residual = function(x, y)
        if not torch.is_tensor(residual) or residual.shape != (n,):
            raise ValueError("physics must return one residual per latent sample")
        return residual

    with torch.no_grad():
        before = evaluate(z)
        raw_before = evaluate(z, raw=True)
    if not bool((torch.isfinite(before) & (before >= 0)).all()):
        raise FloatingPointError("initial physical residual must be finite and nonnegative")
    residual = before.clone()
    accepted_steps = torch.zeros(n, dtype=torch.long, device=z.device)
    backtracks = torch.zeros_like(accepted_steps)
    initial_gradient_rms = z.new_zeros(n)
    last_gradient_rms = z.new_zeros(n)
    maximum_step_rms = z.new_zeros(n)
    stopped = torch.zeros(n, dtype=torch.bool, device=z.device)
    status = np.full(n, "step_limit" if resolved["guidance_steps"] else "disabled", dtype="<U24")

    for iteration in range(resolved["guidance_steps"]):
        active = (residual > resolved["guidance_tolerance"]) & ~stopped
        if not bool(active.any()):
            break
        with torch.enable_grad():
            current = z.detach().requires_grad_(True)
            value = evaluate(current)
            if value.requires_grad:
                grad = torch.autograd.grad(value.sum(), current, allow_unused=True)[0]
                if grad is None:
                    grad = torch.zeros_like(current)
            else:
                grad = torch.zeros_like(current)
        finite_gradient = torch.isfinite(grad).all(dim=1)
        grad_rms = _rms(torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad)))
        if iteration == 0:
            initial_gradient_rms = grad_rms.clone()
        last_gradient_rms[active] = grad_rms[active]
        bad = active & ~finite_gradient
        flat = active & finite_gradient & (grad_rms <= resolved["guidance_gradient_epsilon"])
        status[bad.cpu().numpy()] = "nonfinite_gradient"
        status[flat.cpu().numpy()] = "zero_gradient"
        stopped |= bad | flat
        active &= ~stopped
        if not bool(active.any()):
            break
        direction = torch.where(active[:, None], grad, torch.zeros_like(grad))
        direction = direction / grad_rms.clamp_min(resolved["guidance_gradient_epsilon"])[:, None]
        step = z.new_full((n, 1), resolved["guidance_lr"])
        accepted = torch.zeros_like(active)
        new_z, new_residual = z.clone(), residual.clone()
        with torch.no_grad():
            for attempt in range(resolved["guidance_backtracks"] + 1):
                proposal = z - step * direction
                displacement = proposal - initial
                factor = (resolved["guidance_trust_radius"] /
                          _rms(displacement).clamp_min(resolved["guidance_gradient_epsilon"]))
                proposal = initial + displacement * factor.clamp_max(1.)[:, None]
                candidate = evaluate(proposal)
                better = (active & ~accepted & torch.isfinite(candidate)
                          & (candidate >= 0) & (candidate < residual))
                new_z[better] = proposal[better]
                new_residual[better] = candidate[better]
                accepted |= better
                pending = active & ~accepted
                if not bool(pending.any()):
                    break
                if attempt < resolved["guidance_backtracks"]:
                    step[pending] *= .5
                    backtracks[pending] += 1
            failed = active & ~accepted
            stopped |= failed
            status[failed.cpu().numpy()] = "no_strict_decrease"
            maximum_step_rms = torch.maximum(maximum_step_rms, _rms(new_z - z))
            z, residual = new_z, new_residual
            accepted_steps += accepted.long()

    with torch.no_grad():
        after = evaluate(z)
        raw_after = evaluate(z, raw=True)
        displacement = _rms(z - initial)
    if not bool((torch.isfinite(after) & (after >= 0) & (after <= before)).all()):
        raise FloatingPointError("fixed decoder must preserve accepted per-sample residual descent")
    if resolved["guidance_steps"]:
        status[(after <= resolved["guidance_tolerance"]).cpu().numpy()] = "within_tolerance"
        boundary = displacement >= resolved["guidance_trust_radius"] * (1. - 1e-5)
        status[boundary.cpu().numpy() & (status == "step_limit")] = "trust_boundary"
    diagnostics = {
        "before": before,
        "after": after,
        "raw_before": raw_before,
        "raw_after": raw_after,
        "accepted_steps": accepted_steps,
        "backtracks": backtracks,
        "rms_displacement": displacement,
        "initial_gradient_rms": initial_gradient_rms,
        "last_gradient_rms": last_gradient_rms,
        "maximum_step_rms": maximum_step_rms,
    }
    diagnostics = {name: value.detach().cpu().numpy() for name, value in diagnostics.items()}
    diagnostics["status"] = status
    diagnostics["raw_proxy_available"] = np.full(n, hasattr(physics, "raw_residual"), dtype=bool)
    return z.detach(), diagnostics
