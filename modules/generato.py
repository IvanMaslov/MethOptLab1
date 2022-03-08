import sympy as sy
import numpy as np


def gen_from_symbolic(_func, _vars):
    func = sy.lambdify(_vars, _func)

    func_grad = [_func.diff(_s) for _s in _vars]
    func_grad = [sy.lambdify(_vars, func_grad[i]) for i, _ in enumerate(_vars)]
    func_grad = (
        lambda grad:
        lambda *xs: np.array([df(*xs) for df in grad])
    )(func_grad)

    return func, func_grad


def gen_qrt(n: int):
    def get_rand(): return np.random.normal(-10, 10)

    _vars = [sy.Symbol(f'x{i}') for i in range(n)]

    _func = get_rand()

    for var in _vars:
        c = get_rand()
        pow_c = get_rand()
        _func = _func + pow_c * var ** 2 + c * var

    return gen_from_symbolic(_func, _vars)
