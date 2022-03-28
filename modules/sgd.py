from typing import Callable, List
from typing import TypeVar

import numpy as np

T = TypeVar('T')


def stochastic_gradient_descent(
        func: Callable[[T], T],
        grad_func: Callable[[T], T],
        step_func,
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
