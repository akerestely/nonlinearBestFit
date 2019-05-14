# compute alpha using get_alpha function

import numpy as np

def f(k: np.ndarray, alpha: float, x: np.ndarray):
    return np.sum(np.exp(-k * alpha * x))
def g(k: np.ndarray, alpha: float, x: np.ndarray, y: np.ndarray):
    return np.sum(y * np.exp(-k * alpha * x))
def h(k: np.ndarray, alpha: float, x: np.ndarray):
    return np.sum(x * np.exp(-k * alpha * x))
def l(k: np.ndarray, alpha: float, x: np.ndarray, y: np.ndarray):
    return np.sum(x * y * np.exp(-k * alpha * x))

def get_denom(alpha: float, x: np.ndarray):
    '''Get denominator for calculating A and B'''
    return f(0, alpha, x) * f(2, alpha, x) - f(1, alpha, x) ** 2

def get_A(alpha: float, x: np.ndarray, y: np.ndarray):
    num = g(1, alpha, x, y) * f(0, alpha, x) - g(0, alpha, x, y) * f(1, alpha, x)
    return num / get_denom(alpha, x)

def get_B(alpha: float, x: np.ndarray, y: np.ndarray):
    num = g(0, alpha, x, y) * f(2, alpha, x) - g(1, alpha, x, y) * f(1, alpha, x)
    return num / get_denom(alpha, x)

def get_alpha(alpha: float, x: np.ndarray, y: np.ndarray):
    """eq_6"""
    return get_A(alpha, x, y) * h(2, alpha, x) + get_B(alpha, x, y) * h(1, alpha, x) - l(1, alpha, x, y)

def best_fit(x: np.ndarray, y: np.ndarray, bounds = (1e-5, 1e2)) -> (float, float, float):
    from scipy.optimize import bisect
    alpha = bisect(get_alpha, *bounds, args=(x, y))
    A = get_A(alpha, x, y)
    B = get_B(alpha, x, y)
    return A, B, alpha