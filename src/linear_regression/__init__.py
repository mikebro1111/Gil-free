from .base import BaseRegression
from .normal_equation import NormalEquationRegression
from .gradient_descent import GradientDescentRegression
from .parallel_gd import ParallelGradientDescentRegression
from .sklearn_wrapper import SklearnWrapper

__all__ = [
    'BaseRegression',
    'NormalEquationRegression',
    'GradientDescentRegression',
    'ParallelGradientDescentRegression',
    'SklearnWrapper'
] 