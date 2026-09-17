"""Differentiable TE quality proxy and computational-domain penalty.

The proxy uses vapor-liquid equilibrium, a small-accumulation stripper balance,
and lagged analyzer observations. Inputs and targets are normalized with the
scaler fitted on the labeled training pairs. Physical calculations use float64.
"""
import numpy as np
import torch
from torch import nn


def _values(scaler):
    values = {}
    for name, size in (('x_mean', 23), ('x_std', 23), ('y_mean', 1), ('y_std', 1)):
        value = getattr(scaler, name)
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError('{} must be a finite vector of length {}'.format(name, size))
        if name.endswith('std') and np.any(value <= 0):
            raise ValueError('{} must be strictly positive'.format(name))
        values[name] = value.copy()
    return values


class ObservableTEPhysics(nn.Module):
    """Squared G discrepancy plus an explicit generated-domain displacement cost.

    ``x`` is normalized [N,20,23]; ``y`` is normalized [N,1]. Compositions use
    rows 15:20 and the five extra measurements use rows 13:18 (history 5, lag 2).
    All calculations are float64, retaining autograd links to float32 models.
    """

    history = 5
    lag = 2
    domain_weight = 1.
    mixture_floor_molpercent = 1e-6

    def __init__(self, scaler):
        super().__init__()
        values = _values(scaler)
        constants = dict(A=[15.92, 16.35, 16.35, 16.43, 17.21],
            B=[-1444., -2114., -2114., -2748., -3318.],
            C=[259., 265.5, 265.5, 232.9, 249.6],
            K=[8.501, 11.402, 11.795, .048, .0242],
            MW=[32., 46., 48., 62., 76.],
            AD=[23.3, 33.9, 32.8, 49.9, 50.5],
            BD=[-.0700, -.0957, -.0995, -.0191, -.0541],
            CD=[-.0002, -.000152, -.000233, -.000425, -.000150])
        for name, value in dict(values, **constants).items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float64).clone())

    def _check(self, x, y=None):
        if (not torch.is_tensor(x) or not x.is_floating_point()
                or x.ndim != 3 or x.shape[1:] != (20, 23) or len(x) == 0
                or not bool(torch.isfinite(x).all())):
            raise ValueError('x must be finite, nonempty, floating [N,20,23]')
        if x.device != self.x_mean.device:
            raise ValueError('Move the physics module to the input device with .to(device)')
        if y is not None and (not torch.is_tensor(y) or not y.is_floating_point()
                or y.shape != (len(x), 1) or y.device != x.device
                or not bool(torch.isfinite(y).all())):
            raise ValueError('y must be finite floating [N,1] on the input device')

    def _evaluate(self, x, y=None):
        self._check(x, y)
        # Explicit double arithmetic avoids delicate root cancellation in the
        # physical units; the conversion preserves the decoder's gradient path.
        original = x.to(dtype=torch.float64) * self.x_std.double() + self.x_mean.double()
        composition = original[:, 15:20, :18].mean(dim=1)
        extra = original[:, 13:18, 18:].mean(dim=1)
        raw_product = composition[:, 13:18]
        positive_product = raw_product.clamp_min(0.)
        mixture_added = (self.mixture_floor_molpercent
                         - positive_product.sum(dim=1)).clamp_min(0.)
        product = positive_product + mixture_added[:, None] / 5.
        q4 = extra[:, 0].clamp_min(0.)
        temperature = extra[:, 1].clamp(0., 200.)
        gauge_pressure = extra[:, 2].clamp_min(0.)
        liquid_volume = extra[:, 3].clamp_min(.1)
        stripper_temperature = extra[:, 4].clamp(0., 200.)
        projected_extra = torch.stack((q4, temperature, gauge_pressure,
                                        liquid_volume, stripper_temperature), dim=1)
        raw_view = torch.cat((raw_product, extra), dim=1)
        projected_view = torch.cat((product, projected_extra), dim=1)
        projection_delta = projected_view - raw_view
        projection_scaled = projection_delta / self.x_std[13:23].double()
        domain_penalty = projection_scaled.square().mean(dim=1)

        molar_volume = self.MW.double() / (self.AD.double()
            + self.BD.double() * temperature[:, None]
            + self.CD.double() * temperature[:, None].square())
        s0, s1 = product.sum(dim=1), (product * self.K.double()).sum(dim=1)
        u0 = (product * molar_volume).sum(dim=1)
        u1 = (product * molar_volume * self.K.double()).sum(dim=1)
        # All branches are finite before torch.where. In particular, Tc=177
        # must not evaluate an unused division by zero in the middle branch.
        middle = 363.744 / (177. - stripper_temperature).clamp_min(7.) - 2.22579488
        tmpfac = torch.where(stripper_temperature > 170., stripper_temperature - 120.262,
            torch.where(stripper_temperature < 5.292,
                        torch.full_like(stripper_temperature, .1), middle))
        a = q4 * tmpfac / (.359 * liquid_volume)
        b, c = s0 - a * u1, a * u0
        # m>0 because the projected product mixture has positive mass.
        # This scaled discriminant avoids sqrt(a) and its singular derivative
        # at zero feed, while retaining the correct finite dtheta/da there.
        m = s0 + a * u1
        discriminant = m * ((b / m).square() + 4. * (s1 / m) * (c / m)).sqrt()
        positive_b = b >= 0
        rational_denominator = torch.where(positive_b, b + discriminant, torch.ones_like(b))
        rational_root = 2. * c / rational_denominator
        direct_root = (discriminant - b) / (2. * s1)
        theta = torch.where(positive_b, rational_root, direct_root)
        denominator = s0 + s1 * theta
        liquid_fraction = product * (1. + theta[:, None] * self.K.double()) / denominator[:, None]
        density = denominator / (u0 + u1 * theta)
        pressure = 760. * (1. + gauge_pressure / 101.325)
        equilibrium_pressure = (self.A.double()
                                + self.B.double() / (temperature[:, None] + self.C.double())).exp()
        absolute_purge = 100. * equilibrium_pressure * liquid_fraction / pressure[:, None]
        separator_molar_flow = liquid_volume * density * 35.3145
        result = dict(proxy_molpercent=absolute_purge[:, 3],
            raw_view=raw_view, projected_view=projected_view,
            projection_delta=projection_delta, projection_scaled=projection_scaled,
            domain_penalty=domain_penalty, mixture_added_molpercent=mixture_added,
            theta=theta, density=density, separator_molar_flow=separator_molar_flow,
            absolute_pressure=pressure, liquid_fraction=liquid_fraction,
            absolute_purge_D_E_F_G_H=absolute_purge,
            theta_equation_error=theta - a / density)
        if y is not None:
            target = (y.to(dtype=torch.float64) * self.y_std.double()
                      + self.y_mean.double()).reshape(-1)
            error = result['proxy_molpercent'] - target
            standardized = error / self.y_std.double()[0]
            result.update(target_molpercent=target, g_error_molpercent=error,
                standardized_g_error=standardized, standardized_g_mse=standardized.square(),
                combined_residual=standardized.square() + self.domain_weight * domain_penalty)
        if not all(bool(torch.isfinite(value).all()) for value in result.values()):
            raise FloatingPointError('Nonfinite observable physics calculation; no samples were removed')
        return result

    def proxy_molpercent(self, x):
        """Return the proxy in original mol%, including declared domain protection."""
        return self._evaluate(x)['proxy_molpercent']

    def components(self, x, y):
        """Signed normalized G error followed by ten normalized view displacements."""
        result = self._evaluate(x, y)
        return torch.cat((result['standardized_g_error'][:, None], result['projection_scaled']), dim=1)

    def residual(self, x, y):
        """One nonnegative squared loss per sample, with piecewise derivatives."""
        return self._evaluate(x, y)['combined_residual']

    def raw_residual(self, x, y):
        """VLE-proxy G error squared in mol%^2, excluding the domain penalty."""
        return self._evaluate(x, y)['g_error_molpercent'].square()

    def forward(self, x, y):
        return self.residual(x, y).mean()
