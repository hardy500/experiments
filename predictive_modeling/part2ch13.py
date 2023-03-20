#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# load dataset
names = ['preg', 'plas', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv('datasets/pima-indians-diabetes.csv', names=names)
arr = df.values
x = arr[:,:-1]
y = arr[:,-1]

#%%
### Compare Algorithm ###

# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
  kfold = KFold(n_splits=10)
  cv_results = cross_val_score(model, x, y, cv=kfold, scoring=scoring)

  results.append(cv_results)
  names.append(name)
  print(f"{name}: {cv_results.mean()}, {cv_results.std()}")

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()






