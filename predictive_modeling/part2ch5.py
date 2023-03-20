#%%
import pandas as pd

names = ['preg', 'plas', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv('datasets/pima-indians-diabetes.csv', names=names)

## Understand data with descriptive statistics ##

df.head()
shape = df.shape # dimension
types = df.dtypes # type for each attribute

# Descriptive Statistic
pd.set_option('display.precision', 3)
df.describe()

# Class Distrubution (Classification Only)
# On classification problems you need to know how balanced the class values are
class_count = df.groupby('class').size()

# Correlation Between Attributes
# Correlation refers to the relationship between two variables
# and how they may or may not change together
df.corr(method='pearson')

# Skew of Univariate Distributions
# Skew refers to a distribution that assumed gaussian (normal/bell curve)
# that is shifted or squashed in one directin or another
# The result show a positive (right) or negative (left) skew
# Values closer to zero show less skew
skew = df.skew()
