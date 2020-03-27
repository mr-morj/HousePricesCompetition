import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from math import sqrt
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, RandomizedSearchCV
from multiprocessing import cpu_count
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from mlxtend.regressor import StackingCVRegressor


train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv", index_col=0)


#print(train.head())
#print(train.shape)
#print(test.shape)


#смотрим вібросі по площади
plt.scatter(train["GrLivArea"], train["SalePrice"], alpha=0.9)
plt.xlabel("Ground living area")
plt.ylabel("Sale price")
plt.show()

#убираем вібросі по площади
train = train[train["GrLivArea"] < 4200]
X = pd.concat([train.drop("SalePrice", axis=1), test])

#логарифмируем таргет
y = np.log(train["SalePrice"])

#пропущенніе данніе
nans = X.isna().sum().sort_values(ascending=False)
nans = nans[nans > 0]
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(nans.index, nans.values)
ax.set_ylabel("No. of missing values")
ax.xaxis.set_tick_params(rotation=90)
plt.show()

#категорикал пропущенніе
cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageCond", "GarageQual", "GarageFinish", "GarageType", "BsmtCond", "BsmtExposure", "BsmtQual", "BsmtFinType2", "BsmtFinType1"]
X[cols] = X[cols].fillna("None")

#нумерикал пропущенніе
cols = ["GarageYrBlt", "MasVnrArea", "BsmtHalfBath", "BsmtFullBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "GarageCars"]
X[cols] = X[cols].fillna(X[cols].median())
cols = ["MasVnrType", "MSZoning", "Utilities", "Exterior1st", "Exterior2nd", "SaleType", "Electrical", "KitchenQual", "Functional"]
X[cols] = X.groupby("Neighborhood")[cols].transform(lambda x: x.fillna(x.mode()[0]))
cols = ["GarageArea", "LotFrontage"]
X[cols] = X.groupby("Neighborhood")[cols].transform(lambda x: x.fillna(x.median()))

#пропущенніе данніе проверка
nans = X.isna().sum().sort_values(ascending=False)
nans = nans[nans > 0]
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(nans.index, nans.values)
ax.set_ylabel("No. of missing values")
ax.xaxis.set_tick_params(rotation=90)
plt.show()

#rfntujhbb
cols = ["MSSubClass", "YrSold"]
X[cols] = X[cols].astype("category")


#трасформируем
cols = X.select_dtypes(np.number).columns
X[cols] = RobustScaler().fit_transform(X[cols])

X = pd.get_dummies(X)

#возвращаем тест и треин
X_train1 = X.loc[train.index]
X_test_sub = X.loc[test.index]

#убираем выбросы (НЕДОПИСАНО)


#выбирает треин/тест сет
train_size = 0.75
separator = round(len(X_train1.index)*train_size)


X_train, y_train = X_train1.iloc[0:separator], y.iloc[0:separator]
X_test, y_test = X_train1.iloc[separator:], y.iloc[separator:]


model = XGBRegressor(learning_rate=0.01,n_estimators=20000,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.006)
"""
model = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=4, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=100, n_iter_no_change=None, presort='auto',
             random_state=None, subsample=1.0, tol=0.0001,
             validation_fraction=0.1, verbose=0, warm_start=False)
"""
model.fit(X_train, y_train)   


preds = model.predict(X_test) 
preds_test = model.predict(X_test_sub)
print('Score:', model.score(X_test, y_test)) 
print('MAE: ', mean_absolute_error(y_test, preds))


submission = pd.DataFrame({'Id': X_test_sub.index,
                       'SalePrice': np.exp(preds_test)})
submission.to_csv("submission.csv", index=False)
