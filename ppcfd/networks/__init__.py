import operator
from functools import reduce



try:
    import open3d
except ImportError:
    print("open3d are not imported")
import sys

from .MLP import MLP
from .LRQSolver import LRQSolver
from .Transformer import Point_Transformer
from .MLP import MLP_with_boundary_condition
from .ConvUNet2 import UNet3DWithSamplePoints

__all__ = [
    "RegPointNet",
    "MLP",
    "Point_Transformer",
    "LRQSolver",
]


def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.shape + (2,) if p.is_complex() else p.shape))
    return c
