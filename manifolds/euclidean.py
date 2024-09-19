import torch
import geoopt


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

    def norm(self, u: torch.Tensor, x: torch.Tensor=None, *, keepdim=False):
        return torch.norm(u, dim=-1, keepdim=keepdim)

    def jacobian_expmap_v(self, x, v):
        return torch.eye(v.shape[1], device=v.device).unsqueeze(0)

    def jacobian_expmap_x(self, x, v):
        return torch.eye(x.shape[1], device=x.device).unsqueeze(0)