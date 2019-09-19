#%%
%load_ext autoreload
#%%
%autoreload 2

#%%
import tools

#%%
# generate some data, than plot with best fit curves
x, y = tools.gen_rand_points(20, alpha = 2.1, noise=2, consecutive=False)
tools.plot_results(x, y, ptsStart=0, ptsTrain=5, ptsEnd=None, data_id='generated')

#%%
# plot a data sample, with the first few values zoomed in
x, y = tools.get_real_data(row = 24)
tools.plot_with_inner_plot(x, y, 0.5, 6.5, 0, 6000)

#%%
# load real data sample
x, y = tools.plot_and_get_real_data(row = 23)
#%%
# than compute and plot best fit curves using a subset of the data points
tools.plot_results(x, y, ptsStart=0, ptsTrain=5, ptsEnd=20, data_id=str(row))

#%%
# compute best fit curves for all real data samples where there are more than
# three values per sample 
data = tools.load_data()
data = data[data.count(axis=1) > 3]
for row_idx in data.index:
    x, y = tools.get_x_y(data, row_idx)
    tools.plot_results(x, y, ptsStart=1, ptsTrain=None, data_id=str(row_idx))