import numpy as np
import pandas as pd

np.random.seed(421)

def hCG(x: np.ndarray, A: float, B: float, alpha: float):
    return A * np.exp(-alpha * x) + B

def gen_rand_points(n: int, A: float = 1000, B: float = 3, alpha: float = 0.01, noise: float = 2, consecutive: bool = False):
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

def gen_rand_points_and_plot(n: int, A: float, B: float, alpha: float, noise: float, consecutive: bool):
    x, y = gen_rand_points(20, A = 1000, B = 3, alpha = 1, noise=0, consecutive=False)
    import matplotlib.pyplot as plt
    plt.scatter(x, y)
    plt.xlabel("$time$")
    plt.ylabel("$hCG(time)$")
    plt.show()
    return x, y

def load_data(required_data_points: int = 3) -> pd.DataFrame:
    url = "data/measurements.csv"
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
    # discard entries which have less than required_data_points measurements 
    data = data[data.count(axis=1) > required_data_points]
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

def print_rmse_methods(x: np.ndarray, y: np.ndarray, paramsList: list):
    """
    param paramsList: array of tuples, where tuple contains A, B and alpha
    """
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    for i, params in enumerate(paramsList):
        rmse = sqrt(mean_squared_error(y, hCG(x, *params)))
        print(f"Method {i} RMSE: {rmse}")

def plot_methods(x: np.ndarray, y: np.ndarray, paramsList:list , paramsNames: list = [], data_id: str="", showPlot: bool = True):
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
    #print(paramsNames)
    for i, params in enumerate(paramsList):
        rmse = sqrt(mean_squared_error(y, hCG(x, *params)))
        name = paramsNames[i] if i < len(paramsNames) else ("Method " + str(i))
        plt.plot(x, hCG(x, *params),
            label=f'{name}: A=%5.2f, B=%5.2f, alpha=%5.2f, rmse=%5.2f' % (*params, rmse))
    plt.legend()
    if showPlot:
        plt.show()
    # print_rmse_methods(x, y, params, paramsCalc)

def plot_results(x: np.ndarray, y: np.ndarray, ptsStart: int = 0, ptsEnd: int = None, ptsTrain: int = None, data_id: str="", showPlot:bool = True, allAlgorithms:bool = True):
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
    if allAlgorithms:
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

    if allAlgorithms:
        try:
            from pseloglin import fit
            paramsList.append(fit(x_train, y_train))
            paramsNames.append("PseLogLin")
        except:
            pass
    plot_methods(x[ptsStart:ptsEnd], y[ptsStart:ptsEnd], paramsList, paramsNames, data_id, showPlot)

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

def find_and_plot_best_fit(x: np.ndarray, y: np.ndarray):
    import bestfitte
    A, B, alpha = bestfitte.best_fit(x, y)
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y, hCG(x, A, B, alpha)))
    import matplotlib.pyplot as plt
    plt.scatter(x, y, label='data')
    plt.plot(x, hCG(x, A, B, alpha),
                label=f'A=%5.2f, B=%5.2f, alpha=%5.2f, rmse=%5.2f' % (A, B, alpha, rmse))
    plt.legend()
    plt.show()

def find_and_plot_best_fit_param_noise_grid(paramsList, noises):
    import matplotlib.pyplot as plt
    plt.figure(figsize = (20, 10))

    for i, params in enumerate(paramsList):
        for j, noise in enumerate(noises):
            n:int = 20
            x, y = gen_rand_points(n, *params, noise)
            plt.subplot(len(paramsList), len(noises), i * len(noises) + j + 1)
            plt.scatter(x, y)

            import bestfitte
            A, B, alpha = bestfitte.best_fit(x, y)
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(y, hCG(x, A, B, alpha)))
            import matplotlib.pyplot as plt
            plt.scatter(x, y)
            plt.plot(np.arange(n), hCG(np.arange(n), A, B, alpha), 
            label=f'A=%5.2f, B=%5.2f, alpha=%5.2f, noise=%5.2f, \nA=%5.2f, B=%5.2f, alpha=%5.2f, rmse=%5.2f' % (*params, noise, A, B, alpha, rmse))
            plt.legend()

def compare_results_on_datasets(datasets: list):
    '''
        datasets parameter is a list of datasets which contain (x_data, y_data, dataset_name) tuples
    '''
    import matplotlib.pyplot as plt
    plt.figure(figsize = (9*len(datasets), 5))

    for i, dataset in enumerate(datasets):
        x, y, name = dataset
        plt.subplot(1, len(datasets), i + 1)
        plot_results(x, y, data_id = name, showPlot=False)

def compare_time_on_datasets(datasets: list = None):
    '''
        datasets parameter is a list of datasets which contain (x_data, y_data, dataset_name) tuples
            if omitted, 10 random dataset will be generated
    '''

    if datasets is None:
        # generate 10 random datasets
        paramsList = []
        for _ in range(10):
            paramsList.append((
                np.random.random_integers(3, 20), #n
                np.random.random() * 1e3, # A
                np.random.random() * 1e1, # B
                np.random.random() * 1e1, # alpha
                np.random.random() * 1 # noise
            ))
        datasets = []
        for params in paramsList:
            datasets.append(gen_rand_points(*params) +
            (f'n=%d, A=%5.2f, B=%5.2f, alpha=%5.2f, noise=%5.2f' % params,))

    from scipy.optimize import curve_fit
    from bestfitte import best_fit
    from pseloglin import fit
    from time import perf_counter

    rows = []
    for dataset in datasets:
        x, y, name = dataset
        measurements = {'Dataset' : name}

        start = perf_counter()
        try:
            curve_fit(hCG, x, y)        
            end = perf_counter()
            measurements["Iterative"] = end - start
        except:
            measurements["Iterative"] = np.nan
        
        start = perf_counter()
        try:
            best_fit(x, y)
            end = perf_counter()
            measurements["BestFit"] = end - start
        except:
            measurements["BestFit"] = np.nan

        start = perf_counter()
        try:
            fit(x, y)
            end = perf_counter()
            measurements["PseLogLin"] = end - start
        except:
            measurements["PseLogLin"] = np.nan
            
        rows.append(measurements)
    
    import pandas as pd
    df = pd.DataFrame(rows, columns=["Dataset", "Iterative", "BestFit", "PseLogLin"])
    df.loc['mean'] = df.mean()
    df["Dataset"].values[-1] = "Mean"
    #print(df.to_latex(index=False))
    return df

def compare_with_less_trained(x: np.ndarray, y: np.ndarray, trainPoints):
    '''
    trainPoints, array with the number of points to use for train on each subplot
    '''
    import matplotlib.pyplot as plt
    plt.figure(figsize = (9 * len(trainPoints), 10))
    plt.subplot(2, len(trainPoints), len(trainPoints) / 2 + 1)
    plot_results(x, y, showPlot=False, allAlgorithms=False, data_id="All")

    for i, ptsTrain in enumerate(trainPoints):
        plt.subplot(2, len(trainPoints), len(trainPoints) + i + 1)
        plot_results(x, y, ptsTrain = ptsTrain, showPlot=False, allAlgorithms=False, data_id=str(ptsTrain) + " points")
        plt.plot(x[ptsTrain:], y[ptsTrain:], "o", color="orange")