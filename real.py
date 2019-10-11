#%%
import tools

#%%
# load real dataset where there are more than three values per sample 
data = tools.load_data()
data

#%%
# compute and plot the curves on 3 real datasets
datasets = [
    tools.get_x_y(data, 7) + ("1",),
    tools.get_x_y(data, 30) + ("2",),
    tools.get_x_y(data, 35) + ("3",)
]
tools.compare_results_on_datasets(datasets)

#%%
# for a dataset use only a subset of the points for training
x, y = tools.get_x_y(data, 15)
tools.compare_with_less_trained(x, y, [3, 5, 7])

#%%
# for a dataset use only a subset of the points for training
# observe that initially the points don't follow an exponential decay
x, y = tools.get_x_y(data, 32)
tools.compare_with_less_trained(x, y, [3, 5, 13])

#%%
# compute best fit curves for all real data samples
for row_idx in data.index:
    x, y = tools.get_x_y(data, row_idx)
    tools.plot_results(x, y, ptsStart=0, ptsTrain=None, data_id=str(row_idx))

#%%
# for debugging:
# load just a single data sample
x, y = tools.plot_and_get_real_data(row = data.index[33])