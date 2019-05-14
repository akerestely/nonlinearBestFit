# compute alpha using 'p' function, with any x values

import numpy as np

def f(k: np.ndarray, t: float, x: np.ndarray):
    return np.sum(np.power(t, k * x))
def g(k: np.ndarray, t: float, x: np.ndarray, y: np.ndarray):
    return np.sum(y * np.power(t, k * x))
def h(k: np.ndarray, t: float, x: np.ndarray):
    return np.sum(x * np.power(t, k * x))
def l(k: np.ndarray, t: float, x: np.ndarray, y: np.ndarray):
    return np.sum(x * y * np.power(t, k * x))

def get_denom(t: float, x: np.ndarray):
    '''Get denominator for calculating A and B'''
    return f(0, t, x) * f(2, t, x) - f(1, t, x) ** 2

def get_A(t: float, x: np.ndarray, y: np.ndarray):
    num = g(1, t, x, y) * f(0, t, x) - g(0, t, x, y) * f(1, t, x)
    return num / get_denom(t, x)

def get_B(t: float, x: np.ndarray, y: np.ndarray):
    num = g(0, t, x, y) * f(2, t, x) - g(1, t, x, y) * f(1, t, x)
    return num / get_denom(t, x)

def p(t: float, x: np.ndarray, y: np.ndarray):
    """eq_6"""
    return get_A(t, x, y) * h(2, t, x) + get_B(t, x, y) * h(1, t, x) - l(1, t, x, y)

def best_fit(x: np.ndarray, y: np.ndarray, bounds = (1e-5, 1 - 1e-5)) -> (float, float, float):
    from scipy.optimize import bisect
    t = bisect(p, *bounds, args=(x, y))     # find t
    alpha = -np.log(t)      # get alpha from t
    A = get_A(t, x, y)
    B = get_B(t, x, y)
    return A, B, alpha