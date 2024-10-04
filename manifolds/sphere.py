from typing import Union, Tuple, Optional
import geoopt
import torch
from utils.math_utils import sin_div
from geoopt.manifolds.stereographic.math import geodesic
from torch_scatter import scatter_sum


EPS = {torch.float32: 1e-4, torch.float64: 1e-7}


class Sphere(geoopt.Sphere):
    def __init__(self, learnable=False):
        super(Sphere, self).__init__()
        self.k = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=learnable)

    def origin(
        self,
        *size: Union[int, Tuple[int]],
        dtype=None,
        device=None,
        seed: Optional[int] = 42
    ) -> torch.Tensor:
        pole = torch.zeros(*size, dtype=dtype).to(device)
        pole.narrow(-1, 0, 1).add_(-1)
        return pole

    def cinner(self, x, y):
        return x @ y.transpose(-1, -2)

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        if v is None:
            v = u
        inner = (u * v).sum(-1, keepdim=keepdim)
        return inner

    def geodesic(self, t, x, y):
        return geodesic(t, x, y, k=self.k.to(x.device))

    def Frechet_mean(self, x, weights=None, dim=0, keepdim=False, sum_idx=None):
        if weights is None:
            z = torch.sum(x, dim=dim, keepdim=keepdim) if sum_idx is None \
                else scatter_sum(x, index=sum_idx, dim=dim)
        else:
            z = torch.sum(x * weights, dim=dim, keepdim=keepdim) if sum_idx is None \
                else scatter_sum(x * weights, index=sum_idx, dim=dim)
        denorm = self.inner(None, z, keepdim=keepdim)
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        z = 1. / self.k.sqrt() * z / denorm
        return z

    def expmap0(self, u: torch.Tensor, dim=-1):
        """Choose South Pole"""
        pole = torch.zeros_like(u)
        pole.narrow(dim, 0, 1).add_(-1)
        return self.expmap(pole, u)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        norm_u = u.norm(dim=-1, keepdim=True)
        exp = x * torch.cos(norm_u) + u * sin_div(norm_u)
        retr = self.projx(x + u)
        cond = norm_u > EPS[norm_u.dtype]
        return torch.where(cond, exp, retr)

    def logmap0(self, y: torch.Tensor):
        x = self.origin(y.shape, dtype=y.dtype, device=y.device)
        u = self.proju(x, y - x)
        dist = self.dist(x, y, keepdim=True)
        cond = dist.gt(EPS[x.dtype])
        result = torch.where(
            cond, u * dist / u.norm(dim=-1, keepdim=True).clamp_min(EPS[x.dtype]), u
        )
        return result

    def proju0(self, u: torch.Tensor) -> torch.Tensor:
        x = self.origin(u.shape, dtype=u.dtype, device=u.device)
        u = u - (x * u).sum(dim=-1, keepdim=True) * x
        return self._project_on_subspace(u)

    def norm(self, u: torch.Tensor, x: torch.Tensor = None, *, keepdim=False) -> torch.Tensor:
        return torch.norm(u, dim=-1, keepdim=keepdim)

    def random_normal(
            self, *size, mean=0, std=1, dtype=None, device=None
    ):
        tens = (
                torch.randn(*size, device=device, dtype=dtype)
                * std
                + mean
        )
        return geoopt.ManifoldTensor(self.expmap0(tens), manifold=self)