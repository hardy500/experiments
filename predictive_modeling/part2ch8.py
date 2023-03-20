#%%
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier



names = ['preg', 'plas', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv('datasets/pima-indians-diabetes.csv', names=names)

#%%

# Univariate Selection
arr = df.values
x = arr[:,:-1]
y = arr[:,-1]

#%%

# Feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(x, y)

# Summarize scores
print(fit.scores_)
features = fit.transform(x)

#%%

# Recursive Feature Elimination

# feature exptraction
model = LogisticRegression()
rfe = RFE(estimator=model, n_features_to_select=3)
fit = rfe.fit(x, y)

print(f"Num Feature: {fit.n_features_}")
print(f"Selected Features: {fit.support_}")
print(f"Feature Ranking: {fit.ranking_}")

#%%

# Principal Component Analysis

# Feature extraction
pca = PCA(n_components=3)
fit = pca.fit(x)
print(f"Explained Variance: {fit.explained_variance_ratio_}")
print(fit.components_.shape)

#%%

# Feature Importance with Extra Trees Classifier

model = ExtraTreesClassifier()
model.fit(x, y)
print(model.feature_importances_)