#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, KFold,
    cross_val_score, GridSearchCV
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score
)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

df = pd.read_csv('datasets/sonar.csv', header=None)
#%%

# NOTE: Classify Mines vs Rocks

# Analyse Data

# histogram
df.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1);
# density
df.plot(kind='density', subplots=True, layout=(8,8), sharex=False, sharey=False, fontsize=1);

#%%

# correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(df.corr(numeric_only=True), vmin=-1, vmax=1, interpolation='none');
fig.colorbar(cax);

#%%

# Validation
array = df.values
x = array[:,:-1].astype(float)
y = array[:,-1]

val = .2
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val)

#%%

# Evaluate Algorithm: Baseline

n_folds = 10
scoring = 'accuracy'

#%%

# Spot-check Algorithm
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results, names = [], []
for name, model in models:
  kfold = KFold(n_splits=n_folds)
  cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)

  results.append(cv_results)
  names.append(name)
  print(f"{name}: {cv_results.mean():.3f} ({cv_results.std():.3f})")

#%%
pipelines = []
pipelines.append(
  ('ScaledLR', Pipeline([
    ('Scalar', StandardScaler()),
    ('LR', LogisticRegression())
  ]))
)

pipelines.append(
  ('ScaledLDA', Pipeline([
    ('Scalar', StandardScaler()),
    ('LDA', LinearDiscriminantAnalysis())
  ]))
)

pipelines.append(
  ('ScaledKNN', Pipeline([
    ('Scalar', StandardScaler()),
    ('KNN', KNeighborsClassifier())
  ]))
)

pipelines.append(
  ('ScaledCART', Pipeline([
    ('Scalar', StandardScaler()),
    ('CART', DecisionTreeClassifier())
  ]))
)

pipelines.append(
  ('ScaledNB', Pipeline([
    ('Scalar', StandardScaler()),
    ('NB', GaussianNB())
  ]))
)

pipelines.append(
  ('ScaledSVM', Pipeline([
    ('Scalar', StandardScaler()),
    ('SVM', SVC())
  ]))
)

results, names = [], []
for name, model in pipelines:
  kfold = KFold(n_splits=n_folds)
  cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)

  results.append(cv_results)
  names.append(name)
  print(f"{name}: {cv_results.mean():.3f} ({cv_results.std():.3f})")

# %%

# Algorithm Tuning

# Tune scaled KNN
scalar = StandardScaler().fit(x_train)
scaled_x = scalar.transform(x_train)
neighbors = [1,3,5,7,9,11,13,15,17,19,21]
param_grid = dict(n_neighbors=neighbors)
model = KNeighborsClassifier()
kfold = KFold(n_splits=n_folds)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(scaled_x, y_train)

print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

#%%

# Tune scaled SVM
scalar = StandardScaler().fit(x_train)
scaled_x = scalar.transform(x_train)
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel=kernel_values)
model = SVC()
kfold = KFold(n_splits=n_folds)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(scaled_x, y_train)

print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

#%%

ensembles = []
ensembles.append(('AB', AdaBoostClassifier()))
ensembles.append(('GBM', GradientBoostingClassifier()))
ensembles.append(('RF', RandomForestClassifier()))
ensembles.append(('ET', ExtraTreeClassifier()))

results, names = [], []
for name, model in ensembles:
  kfold = KFold(n_splits=n_folds)
  cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)

  results.append(cv_results)
  names.append(name)
  print(f"{name}: {cv_results.mean():.3f}, {cv_results.std():.3f}")

#%%

fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names);

#%%

# Finalize Model
scalar = StandardScaler().fit(x_train)
rescaled_x = scalar.transform(x_train)
model = SVC(C=1.5)
model.fit(rescaled_x, y_train)
# estimate accuracy on validation dataset
rescaled_val = scalar.transform(x_val)
pred = model.predict(rescaled_val)
print(accuracy_score(y_val, pred))

print(confusion_matrix(y_val, pred))
print(classification_report(y_val, pred))


