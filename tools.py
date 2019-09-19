import numpy as np
import pandas as pd

np.random.seed(421)

def hCG(x: np.ndarray, A: float, B: float, alpha: float):
    return A * np.exp(-alpha * x) + B

def gen_rand_points(n: int, A: float = 1000, B: float = 3, alpha: float = 0.1, noise: float = 2, consecutive: bool = False):
    """
    :param n: number of points to generate
    :param A, B, alpha: parameters to hCG function
    :param noise: randomly add this much to the result of the hCG function
    """
    from numpy.random import random

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
    data.T.iloc[:, from_row:to_row].dropna(axis=0).plot(kind="line", marker='o', subplots=True, figsize=figsize)

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

def print_rmse_methods(x: np.ndarray, y: np.ndarray, paramsList: list):
    """
    param paramsList: array of tuples, where tuple contains A, B and alpha
    """
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    for i, params in enumerate(paramsList):
        rmse = sqrt(mean_squared_error(y, hCG(x, *params)))
        print(f"Method {i} RMSE: {rmse}")

def plot_methods(x: np.ndarray, y: np.ndarray, paramsList:list , paramsNames: list = [], data_id: str=""):
    """
    param paramsList: array of tuples, where tuple contains A, B and alpha
    param paramsNames: array of strings, where each sting represents the name of the corresponding param tuple.
        The names will appear on the plot. Optional, in which case the name will be the index in the array.
    """    
    from sklearn.metrics import mean_squared_error
    from math import sqrt

    import matplotlib.pyplot as plt
    plt.xlabel(r"$time$")
    plt.ylabel(r"$hCG(time)$")
    plt.plot(x, y, 'bo', label=f"data {data_id}")
    print(paramsNames)
    for i, params in enumerate(paramsList):
        rmse = sqrt(mean_squared_error(y, hCG(x, *params)))
        name = paramsNames[i] if i < len(paramsNames) else ("Method " + str(i))
        plt.plot(x, hCG(x, *params),
            label=f'{name}: A=%5.2f, B=%5.2f, alpha=%5.2f, rmse=%5.2f' % (*params, rmse))
    plt.legend()
    plt.show()
    # print_rmse_methods(x, y, params, paramsCalc)

def plot_results(x: np.ndarray, y: np.ndarray, ptsStart: int = 0, ptsEnd: int = None, ptsTrain: int = None, data_id: str=""):
    """
    :param ptsStart: use x, y values starting from this point
    :param ptsEnd: use x, y values ending at this point
    :param ptsTrain: use this much x, y values for training starting from ptsStart
    """
    ptsEnd = ptsEnd or len(x)
    ptsTrain = ptsTrain or (ptsEnd - ptsStart)
    if ptsStart + ptsTrain > ptsEnd:
        raise ValueError("Invalid interval for points")

    x_train = x[ptsStart : ptsStart + ptsTrain]
    y_train = y[ptsStart : ptsStart + ptsTrain]

    paramsList = []
    paramsNames = []
    try:
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(hCG, x_train, y_train)   # uses Levenberg-Marquardt iterative method
        paramsList.append(tuple(popt))
        paramsNames.append("Iterative")
    except:
        pass
    try:
        from bestfitte import best_fit
        paramsList.append(best_fit(x_train, y_train))
        paramsNames.append("BestFit")
    except:
        pass
    try:
        from pseloglin import fit
        paramsList.append(fit(x_train, y_train))
        paramsNames.append("PseLogLin")
    except:
        pass
    plot_methods(x[ptsStart:ptsEnd], y[ptsStart:ptsEnd], paramsList, paramsNames, data_id)

def plot_and_get_real_data(row: int) -> (np.ndarray, np.ndarray):
    data = load_data()
    plot_real_data(data, row, row+1)
    return get_x_y(data, row)

def get_real_data(row: int) -> (np.ndarray, np.ndarray):
    data = load_data()
    return get_x_y(data, row)

def plot_with_inner_plot(x: np.ndarray, y: np.ndarray, limX1: float, limX2: float, limY1: float, limY2: float, zoom: float = 2.5, loc='upper right'):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    plt.xlabel("$time$")
    plt.ylabel("$hCG(time)$")

    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    axins = zoomed_inset_axes(ax, zoom, loc=loc)
    axins.scatter(x, y)

    axins.set_xlim(limX1, limX2)
    axins.set_ylim(limY1, limY2)

    #plt.yticks(visible=False)
    #plt.xticks(visible=False)

    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")