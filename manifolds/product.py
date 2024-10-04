import torch
import geoopt
from typing import Tuple, Union


def _calculate_target_batch_dim(*dims: int):
    return max(dims) - 1


class ProductSpace(geoopt.ProductManifold):
    def __init__(self, *manifolds_with_shape: Tuple[geoopt.Manifold, Union[Tuple[int, ...], int]]):
        super(ProductSpace, self).__init__(*manifolds_with_shape)

    def logmap0(self, x: torch.Tensor) -> torch.Tensor:
        target_batch_dim = x.dim() - 1
        logmapped_tensors = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            logmapped = manifold.logmap0(point)
            logmapped = logmapped.reshape((*logmapped.shape[:target_batch_dim], -1))
            logmapped_tensors.append(logmapped)
        return torch.cat(logmapped_tensors, -1)

    def proju0(self, u: torch.Tensor) -> torch.Tensor:
        target_batch_dim = u.dim() - 1
        projected = []
        for i, manifold in enumerate(self.manifolds):
            tangent = self.take_submanifold_value(u, i)
            proj = manifold.proju0(tangent)
            proj = proj.reshape((*proj.shape[:target_batch_dim], -1))
            projected.append(proj)
        return torch.cat(projected, -1)

    def random_normal(self, *size, mean=0, std=1, dtype=None, device=None):
        shape = geoopt.utils.size2shape(*size)
        batch_shape = shape[:-1]
        points = []
        for manifold, shape in zip(self.manifolds, self.shapes):
            points.append(
                manifold.random_normal(batch_shape + shape, mean=mean, std=std, dtype=dtype, device=device)
            )
        tensor = self.pack_point(*points)
        return geoopt.ManifoldTensor(tensor, manifold=self)

    def Frechet_mean(self, x, weights=None, dim=0, keepdim=False, sum_idx=None):
        target_batch_dim = x.dim() - 1
        mid_tensors = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            mid = manifold.Frechet_mean(point, weights, dim, keepdim, sum_idx)
            mid = mid.reshape((*mid.shape[:target_batch_dim], -1))
            mid_tensors.append(mid)
        return torch.cat(mid_tensors, -1)