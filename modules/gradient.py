
def f(x: float):
    b = 1000.0
    c = 2000.0
    return (x - b) * (x - c)


def f_grad(funct, arg:float):
    eps = 0.000001
    return (funct(arg) - funct(arg - eps)) / eps


def sgd_fixed_learning_rate(funct, lr: float = 1e-1, diff_limit: float = 1e-8):
    eval_f = 1
    eval_gf = 0
    optimum = 0
    prev_value = funct(optimum)
    diff_value = 1000000.0
    while diff_value > diff_limit:
        optimum -= lr * f_grad(funct, optimum)
        new_value = funct(optimum)
        eval_f += 1
        eval_gf += 1
        diff_value = abs(new_value - prev_value)
        prev_value = new_value
    return optimum, eval_f, eval_gf


def sgd_fixed_learning_rate_table(f, lr_vector, diff_limit_vector):
    res =  "| lr | dl | opt | f | grad |\n"
    res += "| --- | ---- | --- | --- | --- |\n"
    for diff_limit in diff_limit_vector:
        for lr in lr_vector:
            r = sgd_fixed_learning_rate(f, lr, diff_limit)
            res += f'| {lr} | {diff_limit} | {r[0]} | {r[1]} | {r[2]} |\n'
    return res


def make_gradient():
    print(sgd_fixed_learning_rate_table(f, [1e-1, 1e-2, 1e-3, 1e-4], [1e-5, 1e-8]))




def make_golden_ratio():
    pass