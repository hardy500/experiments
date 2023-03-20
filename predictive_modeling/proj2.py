### Regression ###
# Task: Predict house price
# Data: Boston House Price dataset
# Performance: ???

#%%
# 1. Prepare Problem
# a) Load libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

plt.style.use('classic')

filename = 'datasets/housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
'B', 'LSTAT', 'MEDV']

df = pd.read_csv(filename, delim_whitespace=True, names=names)

#%%

# 2. Summarize Data
# a) Descriptive statistics
pd.set_option('display.precision', 2)
#print(df.describe())
#print(df.corr(method='pearson'))

#df.plot(kind='density', subplots=True, layout=(4,4), sharex=False, sharey=False, fontsize=1);
#df.plot(kind='box', subplots=True, layout=(4,4),  sharex=False, sharey=False);
#scatter_matrix(df);

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(df.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = np.arange(0, 14, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names);

#%%

# 3. Prepare Data
# a) Data Cleaning
# b) Feature Selection
# c) Data Transforms

# 4. Evaluate Algorithms
# a) Split-out validation dataset
arr = df.values
x = arr[:,:-1]
y = arr[:,-1]
val_size = 0.2

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_size)
# b) Test options and evaluation metric
n_folds = 10
scoring = 'neg_mean_squared_error'
# c) Spot Check Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

results, names = [], []
for name, model in models:
  kfold = KFold(n_splits=n_folds)
  cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)

  results.append(cv_results)
  names.append(name)
  print(f"{name}: {cv_results.mean():.2f}, {cv_results.std():.2f}")

# d) Compare Algorithms
fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names);

#%%

pipelines = []
pipelines.append(
  ('ScaledEN', Pipeline([
    ('Scalar', StandardScaler()),
    ('EN', ElasticNet())
  ]))
)

pipelines.append(
  ('ScaledKNN', Pipeline([
    ('Scalar', StandardScaler()),
    ('KNN', KNeighborsRegressor())
  ]))
)

pipelines.append(
  ('ScaledCART', Pipeline([
    ('Scalar', StandardScaler()),
    ('CART', DecisionTreeRegressor())
  ]))
)

pipelines.append(
  ('ScaledSVR', Pipeline([
    ('Scalar', StandardScaler()),
    ('SVR', SVR())
  ]))
)

results, names = [], []
for name, model in pipelines:
  kfold = KFold(n_splits=n_folds)
  cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  print(f"{name}: {cv_results.mean():.2f}, {cv_results.std():.2f}")


#%%
# 5. Improve Accuracy

# a) Algorithm Tuning
scalar = StandardScaler().fit(x_train)
x_rescaled = scalar.transform(x_train)
k_values = np.array([1,3,5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits=n_folds)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(x_rescaled, y_train)
print(f"Best: {grid_result.best_score_}, {grid_result.best_params_}")
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, std, param in zip(means, stds, params):
  print(f"{mean:.2f} ({std:.2f}) {param}")

#%%

# b) Ensembles

ensembles = []
ensembles.append(
  ('ScaledAB', Pipeline([
    ('Scalar', StandardScaler()),
    ('AB', AdaBoostRegressor())
  ]))
)

ensembles.append(
  ('ScaledGBM', Pipeline([
    ('Scalar', StandardScaler()),
    ('GBM', GradientBoostingRegressor())
  ]))
)

ensembles.append(
  ('ScaledRF', Pipeline([
    ('Scalar', StandardScaler()),
    ('RF', RandomForestRegressor())
  ]))
)

ensembles.append(
  ('ScaledET', Pipeline([
    ('Scalar', StandardScaler()),
    ('ET', ExtraTreesRegressor())
  ]))
)

results, names = [], []
for name, model in ensembles:
  kfold = KFold(n_splits=n_folds)
  cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  print(f"{name}: {cv_results.mean():.2f}, {cv_results.std():.2f}")

#%%

# Tune scaled GBM
scalar = StandardScaler().fit(x_train)
x_rescaled = scalar.transform(x_train)
param_grid = dict(n_estimators=np.array([50,100,150,200,250,300,350,400]))

model = GradientBoostingRegressor()
kfold = KFold(n_splits=n_folds)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(x_rescaled, y_train)

#%%
print(f"Best: {grid_result.best_score_:.2f}, {grid_result.best_params_}")
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, std,  param in zip(means, stds, params):
  print(f"{mean:.2f} ({std:.2f}) {param}")

#%%
# 6. Finalize Model
scalar = StandardScaler().fit(x_train)
x_rescaled = scalar.transform(x_train)
model = GradientBoostingRegressor(n_estimators=250)
model.fit(x_rescaled, y_train)

# a) Predictions on validation dataset
x_val_rescaled = scalar.transform(x_val)
pred = model.predict(x_val_rescaled)
print(mean_squared_error(y_val, pred))
# b) Create standalone model on entire training dataset
# c) Save model for later use