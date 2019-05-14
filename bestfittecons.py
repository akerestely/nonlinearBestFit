# compute alpha using 'p' function, which considers x vector to have consecutive numbers
# but can easily be modified to work for any an x containing any numbers

import numpy as np

def p(t: float, x: np.ndarray, y: np.ndarray):
    m1 = (len(x) * np.sum(y * t ** x) - np.sum(y) * np.sum(t ** x)) * np.sum(x * t ** (2*x))
    m2 = (np.sum(y) * np.sum(t ** (2 * x)) - np.sum(y * t ** x) * np.sum(t ** x)) * np.sum(x * t ** x)
    m3 = np.sum(x * y * t ** x) * (len(x) * np.sum(t ** (2 * x)) - np.sum(t ** x) ** 2)
    return m1 + m2 - m3

def best_fit(x: np.ndarray, y: np.ndarray, bounds: tuple = (1e-5, 1 - 1e-5)) -> (float, float, float):
    x = np.arange(len(x))   # consecutive values
    from scipy.optimize import bisect
    t = bisect(p, *bounds, args=(x, y))
    alpha = -np.log(t)
    from bestfit import get_A, get_B
    A = get_A(alpha, x, y)
    B = get_B(alpha, x, y)    
    return A, B, alpha