import geoopt
import torch
import geoopt.manifolds.lorentz.math as lmath
from utils.math_utils import sinh_div, cosh_div_square, sinh_div_cube, arcosh, cosh, sinh


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

    def Frechet_mean(self, x, weights=None, dim=0, keepdim=False):
        if weights is None:
            z = torch.sum(x, dim=dim, keepdim=True)
        else:
            z = torch.sum(x * weights, dim=dim, keepdim=keepdim)
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

    def jacobian_expmap_v(self, x, v):
        v_norm = self.norm(v, keepdim=True)  # (N, 1)
        v_hat = v.clone()
        v_hat.narrow(-1, 0, 1).mul_(-1)  # (N, D)
        eye = torch.eye(x.shape[1], device=x.device).unsqueeze(0)
        first_term = sinh_div(v_norm).unsqueeze(-1) * (eye + x.unsqueeze(-1) @ v_hat.unsqueeze(1))
        second_term = (cosh_div_square(v_norm) - sinh_div_cube(v_norm)).unsqueeze(-1) * v.unsqueeze(-1) @ v_hat.unsqueeze(1)
        return first_term + second_term

    def jacobian_expmap_x(self, x, v):
        u = self.norm(v, keepdim=True)  # (N, 1)
        eye = torch.eye(x.shape[1], device=x.device).unsqueeze(0)
        return torch.cosh(u).unsqueeze(-1) * eye


if __name__ == '__main__':
    from torch.autograd.functional import jacobian
    def func(x, v):
        m = Lorentz()
        return m.expmap(x, v)
    x, v = torch.tensor([[2., 1., 1., 1.]]), torch.tensor([[1., 0., 0., 2.]])
    j = jacobian(func, (x, v))[1]
    jv = Lorentz().jacobian_expmap_v(x, v)
    print(j, j.shape)
    print(jv, jv.shape)