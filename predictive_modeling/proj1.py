# Python Project Template

#%%
# 1. Prepare Problem
# a) Load libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#%%
# b) Load dataset
filename = 'datasets/iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = pd.read_csv(filename, names=names)

#%%

# 2. Summarize Data
# a) Descriptive statistics
print(df.describe())
print()
print(df.groupby('class').size())
#%%
# b) Data visualizations
# univariante
df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False);
df.hist();
# multivariate
scatter_matrix(df);

#%%
# 3. Prepare Data
# a) Data Cleaning
# b) Feature Selection
# c) Data Transforms

#%%
# 4. Evaluate Algorithms
# a) Split-out validation dataset
x = df.values[:,:-1]
y = df.values[:,-1]
val_size = .2
seed = 7
x_train, x_val, y_train, y_val = train_test_split(
    x, y,
    test_size=val_size,
    random_state=seed
)

# b) Test options and evaluation metric
# c) Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='lbfgs', max_iter=1000)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# d) Compare Algorithms
results, names = [], []
for name, model in models:
  kfold = KFold(n_splits=10)
  cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')

  results.append(cv_results)
  names.append(name)
  print(f"{name}: {cv_results.mean():.3f}, {cv_results.std():.3f}")

fig = plt.figure()
#fig.subtitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#%%
# 5. Improve Accuracy
# a) Algorithm Tuning
# b) Ensembles

#%%
# 6. Finalize Model
# a) Predictions on validation dataset

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
pred = knn.predict(x_val)
print(accuracy_score(y_val, pred))
print()
print(confusion_matrix(y_val, pred))
print()
print(classification_report(y_val, pred))


# b) Create standalone model on entire training dataset
# c) Save model for later use