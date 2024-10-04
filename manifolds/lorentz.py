import geoopt
import torch
import geoopt.manifolds.lorentz.math as lmath
from utils.math_utils import sinh_div, arcosh, cosh, sinh
from torch_scatter import scatter_sum


EPS = {torch.float32: 1e-4, torch.float64: 1e-7}


class Lorentz(geoopt.Lorentz):
    def __init__(self, k=1.0, learnable=False):
        super(Lorentz, self).__init__(k, learnable)

    def cinner(self, x, y):
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        return x @ y.transpose(-1, -2)

    def geodesic(self, t, x, y):
        k_sqrt = torch.sqrt(self.k)
        nomin = arcosh(-self.inner(None, x / k_sqrt, y / k_sqrt))
        v = self.logmap(x, y)
        return cosh(nomin * t) * x + k_sqrt * sinh(nomin * t) * v / self.norm(v, keepdim=True)

    def expmap(
        self, x: torch.Tensor, u: torch.Tensor, *, norm_tan=False, project=False, dim=-1
    ) -> torch.Tensor:
        nomin = self.norm(u, keepdim=True, dim=dim)
        p = (
                cosh(nomin / torch.sqrt(self.k)) * x
                + sinh_div(nomin / torch.sqrt(self.k)) * u
        )
        return p

    def to_poincare(self, x, dim=-1):
        dn = x.size(dim) - 1
        return x.narrow(dim, 1, dn) / (x.narrow(dim, 0, 1) + torch.sqrt(self.k))

    def from_poincare(self, x, dim=-1, eps=1e-6):
        x_norm_square = torch.sum(x * x, dim=dim, keepdim=True)
        res = (
                torch.sqrt(self.k)
                * torch.cat((1 + x_norm_square, 2 * x), dim=dim)
                / (1.0 - x_norm_square + eps)
        )
        return res

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

    def proju0(self, v: torch.Tensor, *, dim=-1) -> torch.Tensor:
        o = self.origin(v.shape, dtype=v.dtype, device=v.device)
        return self.proju(o, v, dim=dim)

    def logmap0(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        K = self.k
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d).view(-1, d)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=1e-8)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[:, 0:1] / sqrtK, min=1.0 + EPS[x.dtype])
        res[:, 1:] = sqrtK * arcosh(theta) * y / y_norm
        return res