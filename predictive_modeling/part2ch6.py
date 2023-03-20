#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix

## Univariant Plots ##
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

df = pd.read_csv('datasets/pima-indians-diabetes.csv', names=names)

# Historgrams
# A fast way to get an idea of the distribution of each attribute
# Histograms group data into bins and
# provide you a count of the number of observations in each bin
df.hist();

#%%
# Density Plots
# another way of getting a quick idea of the distribution of each attribute
df.plot(kind='density', subplots=True, layout=(3,3), sharex=False);

#%%
# Box and Whisker Plots
# Another useful way to review the distribution of each attribute
# Give an idea of the spread of the data and dots outside of the whiskers
# show candidate outlier values
df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False);

#%%

# Correlation Matrix Plot

corr = df.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr, vmin=-1, vmax=1)
fig.colorbar(cax);


#%%

# Scatter Plot Matrix
# A scatter plot shows the relationship between two variables as dots in two dimensions

scatter_matrix(df);
