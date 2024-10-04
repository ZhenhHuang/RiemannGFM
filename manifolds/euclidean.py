import torch
import geoopt
from torch_scatter import scatter_sum


class Euclidean(geoopt.manifolds.Euclidean):
    def __init__(self):
        super().__init__()

    def expmap0(self, v):
        return v

    def logmap0(self, v):
        return v

    def proju0(self, v):
        return v

    def dist(self, x: torch.Tensor, u: torch.Tensor, *, keepdim=False):
        return (x - u).norm(dim=-1, keepdim=keepdim)

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        if v is None:
            v = u
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def Frechet_mean(self, x, weights=None, dim=0, keepdim=False, sum_idx=None):
        if weights is None:
            z = torch.sum(x, dim=dim, keepdim=keepdim) if sum_idx is None \
                else scatter_sum(x, index=sum_idx, dim=dim)
        else:
            z = torch.sum(x * weights, dim=dim, keepdim=keepdim) if sum_idx is None \
                else scatter_sum(x * weights, index=sum_idx, dim=dim)
        denorm = self.inner(None, z, keepdim=keepdim)
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        z = z / denorm
        return z

    def norm(self, u: torch.Tensor, x: torch.Tensor=None, *, keepdim=False):
        return torch.norm(u, dim=-1, keepdim=keepdim)