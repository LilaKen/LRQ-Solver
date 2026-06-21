import torch


class LpLoss(torch.nn.Module):
    def __init__(self, d=2, p=2, size_average=True, reduction=True, enable_dp=False):
        super().__init__()
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.enable_dp = enable_dp

    def abs(self, x, y):
        assert x.shape == y.shape
        num_examples = x.shape[0]

        h = 1.0 / (x.shape[1] - 1.0)
        w = h ** (self.d / self.p)
        diff = x.reshape(num_examples, -1) - y.reshape(num_examples, -1)
        all_norms = w * torch.linalg.vector_norm(diff, ord=self.p, dim=1)

        if self.reduction:
            return all_norms.mean() if self.size_average else all_norms.sum()
        return all_norms

    def rel(self, x, y):
        assert x.shape == y.shape, f"{x.shape} != {y.shape}"
        if x.ndim == 2:
            start_dim = 0
            reduce_dim = None
        elif x.ndim == 3:
            start_dim = 1
            reduce_dim = 1
        else:
            raise ValueError(
                f"Input x and y must be [N,C] or [B,N,C], got {x.ndim} and {y.ndim}"
            )

        diff = (x - y).flatten(start_dim=start_dim)
        target = y.flatten(start_dim=start_dim)
        diff_norms = torch.linalg.vector_norm(diff, ord=self.p, dim=reduce_dim)
        y_norms = torch.linalg.vector_norm(target, ord=self.p, dim=reduce_dim)
        loss = diff_norms / (y_norms + 1e-12)

        if self.reduction:
            return loss.mean() if self.size_average else loss.sum()
        if loss.ndim > 1:
            return loss.mean(dim=1)
        return loss

    def forward(self, x, y):
        return self.rel(x, y)
