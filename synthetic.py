#%%
import tools

#%%
# plot a data sample, with the first few values zoomed in
x, y = tools.get_real_data(row = 24)
tools.plot_with_inner_plot(x, y, 0.5, 6.5, 0, 6000)

#%%
# generate synthetic data
x, y = tools.gen_rand_points(20, A = 1000, B = 3, alpha = 1, noise=0, consecutive=False)

#%%
# plot synthetic data, and retain generated data
x, y = tools.gen_rand_points_and_plot(20, A = 1000, B = 3, alpha = 1, noise=0, consecutive=False)

#%%
# use best fit method to find the original parameters and plot the resulting curve
tools.find_and_plot_best_fit(x, y)

#%%
# generate grid plot with the following configuration
paramsList = [(100, 3, 1), (100, 3, 0.1), (100, 3, 0.01)]
noises = [5, 10, 50]
tools.find_and_plot_best_fit_param_noise_grid(paramsList, noises)

#%%
# compare results for three different datasets
paramsList = [
    # A, B, alpha, noise
    (100, 3, 1, 5),
    (100, 3, 0.1, 25),
    (10, 3, 0.01, 2)
]
datasets = []
for params in paramsList:
    datasets.append(tools.gen_rand_points(20, *params) +
    (f'A=%5.2f, B=%5.2f, alpha=%5.2f, noise=%5.2f' % params,))
tools.compare_results_on_datasets(datasets)

#%%
# iterative usually fails
paramsList = [
    # n, A, B, alpha, noise
    (20, 100, 3, 0.01, 5)
]
datasets = []
for params in paramsList:
    datasets.append(tools.gen_rand_points(*params) +
    (f'A=%5.2f, B=%5.2f, alpha=%5.2f, noise=%5.2f' % params[1:],))
tools.compare_results_on_datasets(datasets)

#%%
# print the runtime of 10 random datasets on the three different algorithms
# runtime of iterative is better than that of best fit, because best fit has no optimizations
df = tools.compare_time_on_datasets()
df