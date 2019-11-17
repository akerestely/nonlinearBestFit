# compute best fit parameters using pseudo log-linear algorithm

import numpy as np

def fit(x: np.ndarray, y: np.ndarray, ptsForB: int = 2, plotLinreg = False):
    '''
    param ptsForB: use this much points from end of data for calculating B,
        and also exclude this much points when calculating A and alpha
    '''
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(x[:-ptsForB], np.log(y[:-ptsForB]))
    
    if np.isnan(slope) or np.isnan(intercept):
        raise ArithmeticError("Determined A or alpha value is NaN")

    if plotLinreg:
        import matplotlib.pyplot as plt
        plt.scatter(x, np.log(y))
        plt.plot(x, intercept + slope * x)
        plt.show()

    A = np.exp(intercept)
    alpha = -slope
    B = np.mean(y[-ptsForB] - A * np.exp(-alpha*x[-ptsForB]))
    return (A, B, alpha)