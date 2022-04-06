import numpy as np
import random
import matplotlib.pyplot as plt


def gradient_descent(gradient_func, start_point, iterations, eps):
    current_point = start_point
    trajectory = [current_point]
    for it in range(iterations):
        next_gradient = gradient_func.next_gradient(current_point)
        next_point = current_point - next_gradient

        distance = np.linalg.norm(current_point - next_point)
        if distance < eps:
            return trajectory, it

        current_point = next_point
        trajectory.append(current_point)

    return trajectory, iterations


class MSELoss:
    def function(self, regression, points, state):
        sum_square_error = 0.0
        for p in points:
            sum_square_error += (p[1] - regression.function(state, p[0])) ** 2
        return sum_square_error / len(points)

    def gradient(self, regression, points, state):
        sum_square_error = np.array([0.0] * len(state + 1))
        for p in points:
            sum_square_error -= 2 * (p[1] - regression.function(state, p[0])) * regression.gradient(state, p[0])
        return sum_square_error / len(points)


class StandardGradient:
    def __init__(self, regression, points, n, error_func, step):
        self.regression = regression
        self.points = points
        self.n = n
        self.error_func = error_func
        self.step = step

    def next_gradient(self, current_point):
        result = self.step * self.error_func.gradient(self.regression, random.sample(self.points, self.n),
                                                      current_point)
        return result


class MomentumGradient:
    def __init__(self, regression, points, n, error_func, mu, step):
        # mu = 0.9
        self.regression = regression
        self.points = points
        self.n = n
        self.error_func = error_func
        self.mu = mu
        self.step = step
        self.prev_gradient = np.array([0.0] * (len(points[0][0]) + 1))

    def next_gradient(self, current_point):
        result = self.mu * self.prev_gradient + self.step * self.error_func.gradient(self.regression,
                                                                                     random.sample(self.points, self.n),
                                                                                     current_point)
        self.prev_gradient = result
        return result


class NesterovGradient:
    def __init__(self, regression, points, n, error_func, mu, step):
        # mu = 0.9
        self.regression = regression
        self.points = points
        self.n = n
        self.error_func = error_func
        self.mu = mu
        self.step = step
        self.prev_gradient = np.array([0.0] * (len(points[0][0]) + 1))

    def next_gradient(self, current_point):
        result = self.mu * self.prev_gradient + self.step * self.error_func.gradient(self.regression,
                                                                                     random.sample(self.points, self.n),
                                                                                     current_point + self.mu * self.prev_gradient)
        self.prev_gradient = result
        return result


class AdagradGradient:
    def __init__(self, regression, points, n, error_func, step):
        self.regression = regression
        self.points = points
        self.n = n
        self.error_func = error_func
        self.step = step
        self.s = np.array([0.0] * (len(points[0][0]) + 1))

    def next_gradient(self, current_point):
        current_gradient = self.error_func.gradient(self.regression, random.sample(self.points, self.n), current_point)
        self.s = self.s + np.square(current_gradient)
        result = np.multiply(self.step / np.sqrt(self.s), current_gradient)
        return result


class RMSPropGradient:
    def __init__(self, regression, points, n, error_func, mu, step):
        # mu = 0.9
        self.regression = regression
        self.points = points
        self.n = n
        self.error_func = error_func
        self.mu = mu
        self.step = step
        self.s = np.array([0.0] * (len(points[0][0]) + 1))

    def next_gradient(self, current_point):
        current_gradient = self.error_func.gradient(self.regression, random.sample(self.points, self.n), current_point)
        self.s = self.mu * self.s + (1 - self.mu) * np.square(current_gradient)
        result = np.multiply(self.step / np.sqrt(self.s), current_gradient)
        return result


class AdamGradient:
    def __init__(self, regression, points, n, error_func, beta1, beta2, step):
        # beta1 = 0.9
        # beta2 = 0.999
        self.regression = regression
        self.points = points
        self.n = n
        self.error_func = error_func
        self.beta1 = beta1
        self.beta2 = beta2
        self.step = step
        self.g = np.array([0.0] * (len(points[0][0]) + 1))
        self.v = np.array([0.0] * (len(points[0][0]) + 1))
        self.it = 1

    def next_gradient(self, current_point):
        current_gradient = self.error_func.gradient(self.regression, random.sample(self.points, self.n), current_point)
        self.g = self.beta1 * self.g + (1 - self.beta1) * current_gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(current_gradient)
        self.it = self.it + 1
        g_temp = self.g / (1 - self.beta1 ** (self.it - 1))
        v_temp = self.v / (1 - self.beta2 ** (self.it - 1))
        result = self.step * g_temp / (np.sqrt(v_temp) + 1e-8)
        return result


class LinearRegression:
    def __init__(self):
        self.function_calls = 0
        self.gradient_calls = 0

    def function(self, state, point):
        self.function_calls += 1

        res = state[0]
        for i in range(len(point)):
            res += state[i + 1] * point[i]
        return res

    def gradient(self, state, point):
        self.gradient_calls += 1

        return np.concatenate(([1.0], point))


def generate_points(f, number_of_points, number_of_dimensions):
    shifts = [random.uniform(-10, 10) for i in range(number_of_dimensions)]
    multipliers = [random.uniform(0.1, 2) for i in range(number_of_dimensions)]

    result = []
    for i in range(number_of_points):
        point = []
        for j in range(number_of_dimensions):
            x = (np.random.normal(0, 1) + shifts[j]) * multipliers[j]
            point.append(x)
        result.append((point, f(point)))

    return result


def f(xs):
    result = 1
    for i in range(len(xs)):
        result += (2 + i) * xs[i]
    return result


def draw_batch_size_to_iteration_plot(points, step, title):
    regression = LinearRegression()
    error_func = MSELoss()
    eps = 1e-1

    batch_sizes = range(1, len(points) + 1, 2)
    iterations = []

    for batch_size in batch_sizes:
        it = gradient_descent(
            gradient_func=StandardGradient(
                regression=regression,
                points=points,
                n=batch_size,
                error_func=error_func,
                step=step
            ),
            start_point=np.array([0.0] * (len(points[0][0]) + 1)),
            iterations=1000,
            eps=eps
        )[1]
        iterations.append(it)

    plt.title(title)
    plt.xlabel('Batch size')
    plt.ylabel(f'Iterations before distance between points <= {eps}')
    plt.plot(batch_sizes, iterations)


def build_plot(label, train_points, eps, get_gradient_func):
    import time
    regression = LinearRegression()
    error_func = MSELoss()

    start_time = time.time()
    trajectory = gradient_descent(
        gradient_func=get_gradient_func(regression, train_points, error_func),
        start_point=np.array([0.0] * (len(train_points[0][0]) + 1)),
        iterations=1000,
        eps=eps
    )[0][1:]

    print(label)
    print(f'function calls: {regression.function_calls}')
    print(f'gradient calls: {regression.gradient_calls}')
    print(f'seconds: {time.time() - start_time}')
    return plt.plot(range(len(trajectory)), [error_func.function(regression, train_points, x) for x in trajectory],
                    label=label)[0]
