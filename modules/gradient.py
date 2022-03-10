from typing import Callable, Mapping, List, Tuple
from typing import TypeVar

import numpy as np

T = TypeVar('T')


def gradient_descent(
        func: Callable[[T], T],
        grad_func: Callable[[T], T],
        step_func,  # (step: int, func: (T) -> T, grad_func: (T) -> T, pred: T) -> float
        x0: T,
        diff_limit: float = 1e-8
) -> List[T]:
    res: List[T] = [x0]

    step: int = 0
    while True:
        pred = res[-1]
        grad = grad_func(pred)
        k = step_func(step=step, func=func, grad=grad, pred=pred)
        res.append(pred - k * grad)

        step = step + 1
        if np.linalg.norm(func(res[-1]) - func(pred)) < diff_limit:
            return res


def dichotomy(
        func: Callable[[T], T],
        left: float,
        right: float,
        eps: float = 1e-3
):
    while right > left + eps:
        middle = (left + right) / 2
        if func(middle - eps) < func(middle + eps):
            right = middle
        else:
            left = middle

    return (left + right) / 2


def linear_search(
        func: Callable[[T], T],
        left: float,
        delta: float = 1e-3,
        eps: float = 1e-3,
        factor: float = 2.
) -> float:
    right = left + delta
    while func(right) <= func(left) + eps:
        delta *= factor
        right += delta

    return right


def generate_square(
        x2: float,
        y2: float,
        xy: float
) -> Tuple[Callable[[float, float], float], Callable[[float, float], Tuple[float, float]]]:
    def f(x: float, y: float) -> float:
        # f = a * xx + b * yy + c * xy
        return x2 * x * x + y2 * y * y + xy * x * y

    def grad(x: float, y: float) -> Tuple[float, float]:
        # dx = 2ax + cy
        # dy = 2by + cx
        # grad = (dx, dy)
        return x2 * x * 2 + xy * y, y2 * y * 2 + xy * x

    return f, grad


