#%%
%load_ext autoreload
#%%
%autoreload 2

#%%
import numpy as np
import pandas as pd
from numpy.random import random

np.random.seed(421)

def hCG(x: np.ndarray, A: float, B: float, alpha: float):
    return A * np.exp(-alpha * x) + B

def gen_rand_points(n: int, A: float = 1000, B: float = 3, alpha: float = 0.1, noise: float = 2, consecutive: bool = False):
    """
    :param n: number of points to generate
    :param A, B, alpha: parameters to hCG function
    :param noise: randomly add this much to the result of the hCG function
    """
    sparsity = 1
    if consecutive is False:
        x = random(n) * n * sparsity 
        x.sort()    # just for plot visual effect; does not change results
    else :
        x = np.linspace(0, n-1, n) * sparsity
        
    y = hCG(x, A, B, alpha)
    ynoise = random(n) * noise - noise / 2
    y += ynoise
    return x, y

def load_data() -> pd.DataFrame:
    url = "measurements.csv"
    data = pd.read_csv(url)

    # remove unused columns
    data = data.loc[:, data.columns.str.startswith('MTH')]

    def name_to_weekcount(s:str) -> int:
        tokens = s.split('-')
        import re
        mth = int(re.search(r'\d+', tokens[0]).group(0)) - 1
        wk = 0
        if len(tokens) is not 1:
            wk = int(re.search(r'\d+', tokens[1]).group(0)) - 1
        return mth * 4 + wk

    # rename columns
    data.columns = pd.Series(data.columns).apply(name_to_weekcount)
    return data

def get_x_y(data: pd.DataFrame, row: int) -> (np.ndarray, np.ndarray) :
    my_data = data.loc[row:row, :].dropna(axis=1)
    x = np.array(my_data.columns[:])    # time
    y = my_data.iloc[0,:].values        # measurement
    return x, y

def plot_real_data(data, from_row = None, to_row = None):
    figsize = None
    if from_row is not None and to_row is not None:
        count = to_row - from_row
        if count > 1:
            figsize = (10, 5 * count)
    data.T.iloc[:, from_row:to_row].plot(kind="line", marker='o', subplots=True, figsize=figsize)

def plot_function(func, x: np.ndarray, y: np.ndarray):
    import matplotlib.pyplot as plt
    range_param = np.linspace(0, 1)
    pt = [func(t, x, y) for t in range_param]
    plt.plot(range_param, pt)
    plt.show()

    import pandas as pd
    df = pd.DataFrame(columns=["t", "p(t)"])
    df["param"] = range_param
    df["func(param)"] = pt
    print(df)

def print_rmse_methods(x: np.ndarray, y: np.ndarray, paramsIter: tuple, paramsCalc:tuple):
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    rmseIter = sqrt(mean_squared_error(y, hCG(x, *paramsIter)))
    rmseCalc = sqrt(mean_squared_error(y, hCG(x, *paramsCalc)))
    print(f"Iterative method RMSE: {rmseIter}")
    print(f" Calculus method RMSE: {rmseCalc}")

def plot_methods(x: np.ndarray, y: np.ndarray, paramsIter: tuple = None, paramsCalc: tuple = None, data_id: str=""):
    from sklearn.metrics import mean_squared_error
    from math import sqrt

    import matplotlib.pyplot as plt
    plt.xlabel(r"$time$")
    plt.ylabel(r"$hCG(time)$")
    plt.plot(x, y, 'bo', label=f"data {data_id}")
    if paramsIter is not None:
        rmseIter = sqrt(mean_squared_error(y, hCG(x, *paramsIter)))
        plt.plot(x, hCG(x, *paramsIter), 'r-',
            label='Iterative fit: A=%5.2f, B=%5.2f, alpha=%5.2f, rmse=%5.2f' % (*paramsIter, rmseIter))
    
    if paramsCalc is not None:
        rmseCalc = sqrt(mean_squared_error(y, hCG(x, *paramsCalc)))
        plt.plot(x, hCG(x, *paramsCalc), 'g-',
            label='Calculus fit: A=%5.2f, B=%5.2f, alpha=%5.2f, rmse=%5.2f' % (*paramsCalc, rmseCalc))
    plt.legend()
    plt.show()
    # print_rmse_methods(x, y, paramsIter, paramsCalc)

def get_params_iterative(x: np.ndarray, y: np.ndarray) -> (float, float, float):
    from scipy.optimize import curve_fit
    popt, _ = curve_fit(hCG, x, y)   # uses Levenberg-Marquardt iterative method
    return tuple(popt)

def get_params_calculus(x: np.ndarray, y: np.ndarray) -> (float, float, float):
    from bestfit import best_fit
    return best_fit(x, y)

def get_params_calculus_t(x: np.ndarray, y: np.ndarray) -> (float, float, float):
    from bestfitte import best_fit
    return best_fit(x, y)

def plot_results(x: np.ndarray, y: np.ndarray, ptsStart: int = None, ptsEnd: int = None, data_id: str=""):
    try:
        params1 = get_params_iterative(x[ptsStart:ptsEnd], y[ptsStart:ptsEnd])
    except:
        params1 = None
    try:
        params2 = get_params_calculus_t(x[ptsStart:ptsEnd], y[ptsStart:ptsEnd])
    except:
        params2 = None
    plot_methods(x, y, params1, params2, data_id)

def plot_and_get_real_data(row: int) -> (np.ndarray, np.ndarray):
    data = load_data()
    plot_real_data(data, row, row+1)
    return get_x_y(data, row)
#%%
x, y = gen_rand_points(20, alpha = 0.1, noise=400, consecutive=False)
plot_results(x, y, ptsEnd=None, data_id='generated')

#%%
data = load_data()
data = data[data.count(axis=1) > 2]
for row_idx in data.index:
    x, y = get_x_y(data, row_idx)
    plot_results(x, y, ptsEnd=None, data_id=str(row_idx))