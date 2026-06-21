from .LRQSolver import LRQSolver

__all__ = ["LRQSolver", "count_params"]


def count_params(model):
    return sum(param.numel() for param in model.parameters())
