#%%
import pandas as pd
from sklearn.model_selection import (
    train_test_split, KFold, cross_val_score,
    LeaveOneOut, ShuffleSplit
)

from sklearn.linear_model import LogisticRegression

names = ['preg', 'plas', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv('datasets/pima-indians-diabetes.csv', names=names)

#%%
arr = df.values
x = arr[:,:-1]
y = arr[:,-1]

#%%

# Split into Train and Test Sets
test_size = .33
seed = 7

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=test_size, random_state=seed
)

model = LogisticRegression()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print(f"Accuracy: {result:.2f}")

#%%

# K-Fold Cross Validation
k = 10
seed = 7
kfold = KFold(n_splits=k)
model = LogisticRegression()
result = cross_val_score(model, x, y, cv=kfold)
print(f"Accuracy: {result.mean():.2f}")

# %%

#k = 10
#model = LogisticRegression()
#loocv = LeaveOneOut()
#results = cross_val_score(model, x, y, cv=loocv)
#print(f"Accuracy: {results.mean():.2f}")


#%%

# Repeated Random Test-Train Splits
n_splits = 10
test_size = 0.33
seed = 7

kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
model = LogisticRegression()
results = cross_val_score(model, x, y, cv=kfold)
print(f"Accuracy: {results.mean():.2f}, {results.std():.2f}")





