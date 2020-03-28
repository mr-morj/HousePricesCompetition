import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer, r2_score
from math import sqrt
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, RandomizedSearchCV
from multiprocessing import cpu_count
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
from xgboost import plot_importance



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
X = pd.concat([train, test])

#логарифмируем таргет
plt.hist(train["SalePrice"], bins = 40)
plt.show()
y = np.log(train["SalePrice"])

plt.hist(y, bins = 40)
plt.show()
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
X[cols] = X[cols].fillna(X[cols].mean())
cols = ["MasVnrType", "MSZoning", "Utilities", "Exterior1st", "Exterior2nd", "SaleType", "Electrical", "KitchenQual", "Functional"]
X[cols] = X.groupby("Neighborhood")[cols].transform(lambda x: x.fillna(x.mode()[0]))
cols = ["GarageArea", "LotFrontage"]
X[cols] = X.groupby("Neighborhood")[cols].transform(lambda x: x.fillna(x.mean()))

#пропущенніе данніе проверка
nans = X.isna().sum().sort_values(ascending=False)
nans = nans[nans > 0]
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(nans.index, nans.values)
ax.set_ylabel("No. of missing values")
ax.xaxis.set_tick_params(rotation=90)
plt.show()

#rfntujhbb
X["TotalSF"] = X["GrLivArea"] + X["TotalBsmtSF"]
X["TotalPorchSF"] = X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]
X["TotalBath"] = X["FullBath"] + X["BsmtFullBath"] + 0.5 * (X["BsmtHalfBath"] + X["HalfBath"])

cols = ["MSSubClass", "YrSold"]
X[cols] = X[cols].astype("category")


#трасформируем
cols = X.select_dtypes(np.number).columns
X[cols] = RobustScaler().fit_transform(X[cols])

#поиск зависимостей
train_corr = X.select_dtypes(include=[np.number])
corr = train_corr.corr()
plt.subplots(figsize=(20,15))
#corr.style.background_gradient(cmap='coolwarm')
sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns)

#оставляем которіе біольше 0.4
top_feature = corr.index[abs(corr['SalePrice']>0.4)]
plt.subplots(figsize=(12, 8))
top_corr = X[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()


#топ по влиятельности на таргет
#corr.sort_values(['SalePrice'], ascending=False, inplace=True)
#print(corr.SalePrice)


def graphics(df, ex, features):
    for ind in features:
        fig, axes = plt.subplots(2, figsize=(8,8))
        axes[0].scatter(df[ind], ex)

        axes[0].grid(True)
        axes[0].set_title(ind + " & price")
        axes[1].boxplot(x=df[ind], vert=False)
        axes[1].grid(True)
        axes[1].set_title(ind + " boxplot")
        fig.tight_layout()
        plt.setp(axes[1], xlabel=ind+' value')
        plt.setp(axes[0], ylabel='price')
        plt.show()
        
def remove_outlier(df_in, col_name):
    
    mean = df_in[col_name].mean()
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    for ind in range(120):
        if (df_in[col_name].iloc[ind]>fence_high) or (df_in[col_name].iloc[ind]<fence_low):
            df_in[col_name].iloc[ind] = mean        

param_with_outliers = ['LotFrontage', 'LotArea', 'YearBuilt', 'MasVnrArea', 'BsmtUnfSF',
                       'TotalBsmtSF', '1stFlrSF', 'LowQualFinSF','GrLivArea', 'GarageArea',
                       'WoodDeckSF', 'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'PoolArea',
                       'MiscVal', 'TotalSF', 'TotalPorchSF']

for param in param_with_outliers:
    X[param] = remove_outlier(X, param) 




#graphics(X_train1, y, cols)

#for param in param_with_outliers:
#    df[param] = remove_outlier(df, param)

X = pd.get_dummies(X)


#возвращаем тест и треин
X_train1 = X.loc[train.index]
X_test_sub = X.loc[test.index]


train_size = 0.75
separator = round(len(X_train1.index)*train_size)


X_train, y_train = X_train1.iloc[0:separator], y.iloc[0:separator]
X_test, y_test = X_train1.iloc[separator:], y.iloc[separator:]

xgb = XGBRegressor(learning_rate=0.01,n_estimators=20000,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:squarederror', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.006)

xgb.fit(X_train, y_train)   
preds = xgb.predict(X_test) 
preds_test_xgb = xgb.predict(X_test_sub)
mae_xgb = mean_absolute_error(y_test, preds)
rmse_xgb = np.sqrt(mean_squared_error(y_test, preds))
score_xgb = xgb.score(X_test, y_test)

print('Score: ', score_xgb)
print('MAE: ', mae_xgb)
print('RMSE: ', rmse_xgb)

submission = pd.DataFrame({'Id': X_test_sub.index,
                       'SalePrice': np.exp(preds_test_xgb)})
submission.to_csv("submission.csv", index=False)
