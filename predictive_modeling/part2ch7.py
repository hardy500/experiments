#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    Normalizer,
    Binarizer
    )

## Univariant Plots ##
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

df = pd.read_csv('datasets/pima-indians-diabetes.csv', names=names)

#%%

## Rescale Data ##
# Scale data to have the same scale/normalize

# Rescale data (between 0 and 1)
array = df.values
# separate array into input and output components
x = array[:,:-1]
y = array[:,-1]
scalar = MinMaxScaler(feature_range=(0, 1))
rescaled_x = scalar.fit_transform(x)



#%%

## Standardize Data ##

# Standardize data (0 mean, 1 std)
scalar = StandardScaler().fit(x)
rescaled_x = scalar.transform(x)

#%%

scaler = Normalizer().fit(x)
normalize_x = scalar.transform(x)
normalize_x

#%%

# Binarize Data
np.set_printoptions(precision=3)

binarizer = Binarizer(threshold=0.0).fit(x)
binary_x = binarizer.transform(x)