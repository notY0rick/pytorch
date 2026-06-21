# mypy: allow-untyped-defs

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["Gamma"]


def _standard_gamma(concentration):
    return torch._standard_gamma(concentration)


class Gamma(ExponentialFamily):
    r"""
    Creates a Gamma distribution parameterized by shape :attr:`concentration` and :attr:`rate`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Gamma(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # Gamma distributed with concentration=1 and rate=1
        tensor([ 0.1046])

    Args:
        concentration (float or Tensor): shape parameter of the distribution
            (often referred to as alpha)
        rate (float or Tensor): rate parameter of the distribution
            (often referred to as beta), rate = 1 / scale
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    support = constraints.nonnegative
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self) -> Tensor:
        return self.concentration / self.rate

    @property
    def mode(self) -> Tensor:
        return ((self.concentration - 1) / self.rate).clamp(min=0)

    @property
    def variance(self) -> Tensor:
        return self.concentration / self.rate.pow(2)

    def __init__(
        self,
        concentration: Tensor | float,
        rate: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.concentration, self.rate = broadcast_all(concentration, rate)
        if isinstance(concentration, _Number) and isinstance(rate, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.concentration.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Gamma, _instance)
        batch_shape = torch.Size(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        super(Gamma, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        value = _standard_gamma(self.concentration.expand(shape)) / self.rate.expand(
            shape
        )
        value.detach().clamp_(
            min=torch.finfo(value.dtype).tiny
        )  # do not record in autograd graph
        return value

    def log_prob(self, value):
        value = torch.as_tensor(value, dtype=self.rate.dtype, device=self.rate.device)
        if self._validate_args:
            self._validate_sample(value)
        return (
            torch.xlogy(self.concentration, self.rate)
            + torch.xlogy(self.concentration - 1, value)
            - self.rate * value
            - torch.lgamma(self.concentration)
        )

    def entropy(self):
        return (
            self.concentration
            - torch.log(self.rate)
            + torch.lgamma(self.concentration)
            + (1.0 - self.concentration) * torch.digamma(self.concentration)
        )

    @property
    def _natural_params(self) -> tuple[Tensor, Tensor]:
        return (self.concentration - 1, -self.rate)

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x, y):
        return torch.lgamma(x + 1) + (x + 1) * torch.log(-y.reciprocal())

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return torch.special.gammainc(self.concentration, self.rate * value)

    def icdf(self, value):
        value = torch.as_tensor(value, dtype=self.rate.dtype, device=self.rate.device)
        concentration, value = broadcast_all(self.concentration, value)
        finfo = torch.finfo(value.dtype)
        # No inverse-incomplete-gamma op exists, so invert the CDF numerically.
        with torch.no_grad():
            q = value.clamp(min=finfo.tiny, max=1 - finfo.eps)
            z = torch.special.ndtri(q)
            c = (9 * concentration).reciprocal()
            w = 1 - c + z * c.sqrt()
            y = torch.where(
                w > 0,
                concentration * w.pow(3),
                ((q.log() + torch.lgamma(concentration + 1)) / concentration).exp(),
            ).clamp(min=finfo.tiny)
            lo = torch.zeros_like(y)
            hi = y.clone()
            for _ in range(64):  # grow hi until [lo, hi] brackets the root
                bracketed = torch.special.gammainc(concentration, hi) >= q
                if bool(bracketed.all()):
                    break
                hi = torch.where(~bracketed, 2 * hi, hi)
            for _ in range(100):  # safeguarded Newton: bisect if the step escapes
                g = torch.special.gammainc(concentration, y) - q
                lo = torch.where(g < 0, y, lo)
                hi = torch.where(g > 0, y, hi)
                pdf = (
                    (concentration - 1) * y.log() - y - torch.lgamma(concentration)
                ).exp()
                newton = y - g / pdf
                outside = (newton <= lo) | (newton >= hi) | ~torch.isfinite(newton)
                y_next = torch.where(outside, (lo + hi) / 2, newton)
                if bool(((y_next - y).abs() <= finfo.eps * y_next).all()):
                    y = y_next
                    break
                y = y_next

        # Reattach gradients after the no_grad solve with a straight-through surrogate.
        conc = concentration.detach()
        log_pdf = (conc - 1) * y.log() - y - torch.lgamma(conc)
        pdf = log_pdf.exp().clamp(min=finfo.tiny)
        shape_grad = torch._standard_gamma_grad(conc, y)
        value_term = (value - value.detach()) / pdf
        conc_term = (concentration - concentration.detach()) * shape_grad
        y = y + value_term + conc_term
        x = y / self.rate
        x = torch.where(value == 0, torch.zeros_like(x), x)
        x = torch.where(value == 1, torch.full_like(x, float("inf")), x)
        x = torch.where((value < 0) | (value > 1), torch.full_like(x, float("nan")), x)
        return x
