import torch


EPS = {torch.float32: 1e-4, torch.float64: 1e-7}


cosh_bounds = {torch.float32: 85, torch.float64: 700}
sinh_bounds = {torch.float32: 85, torch.float64: 500}


def cosh(x):
    x.data.clamp_(max=cosh_bounds[x.dtype])
    return torch.cosh(x)


def sinh(x):
    x.data.clamp_(max=sinh_bounds[x.dtype])
    return torch.sinh(x)


def tanh(x):
    return x.tanh()


def cos(x):
    return torch.cos(x)


def sin(x):
    return torch.sin(x)


class SinhDiv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y_stable = torch.ones_like(x)
        y = sinh(x) / x
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        y = (x * cosh(x) - sinh(x)) / x ** 2
        y_stable = torch.zeros_like(x)
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y) * grad_output


sinh_div = SinhDiv.apply


class SinhDivCube(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y_stable = torch.ones_like(x) / 6
        y = sinh(x) / x ** 3
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        y = (x * cosh(x) - 3 * sinh(x)) / x ** 4
        y_stable = torch.zeros_like(x)
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y) * grad_output


sinh_div_cube = SinhDivCube.apply


class CoshDivSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y_stable = torch.ones_like(x) * 0.5
        y = cosh(x) / x ** 2
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        y = (x * sinh(x) - 2 * cosh(x)) / x ** 3
        y_stable = torch.zeros_like(x)
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y) * grad_output


cosh_div_square = CoshDivSquare.apply


class SinDiv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y_stable = torch.ones_like(x)
        y = torch.sin(x) / x
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        y = (x * torch.cos(x) - torch.sin(x)) / x ** 2
        y_stable = torch.zeros_like(x)
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y) * grad_output


sin_div = SinDiv.apply


class CosDivSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y_stable = -torch.ones_like(x) * 0.5
        y = cos(x) / x ** 2
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        y = (-x * sin(x) - 2 * cos(x)) / x ** 3
        y_stable = torch.zeros_like(x)
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y) * grad_output


cos_div_square = CosDivSquare.apply


class SinDivCube(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y_stable = -torch.ones_like(x) / 6
        y = sin(x) / x ** 3
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        y = (x * cos(x) - 3 * sin(x)) / x ** 4
        y_stable = torch.ones_like(x) / 24
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y) * grad_output


sin_div_cube = SinDivCube.apply


class Acosh(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        x = torch.clamp(x, min=1+EPS[x.dtype])
        z = torch.sqrt(x * x - 1)
        ctx.save_for_backward(z)
        return torch.log(x + z)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        z.data.clamp(min=EPS[z.dtype])
        z = g / z
        return z, None


arcosh = Acosh.apply