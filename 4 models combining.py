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
from catboost import CatBoostRegressor





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
X["TotalSF"] = X["GrLivArea"] + X["TotalBsmtSF"]
X["TotalPorchSF"] = X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]
X["TotalBath"] = X["FullBath"] + X["BsmtFullBath"] + 0.5 * (X["BsmtHalfBath"] + X["HalfBath"])

cols = ["MSSubClass", "YrSold"]
X[cols] = X[cols].astype("category")


#трасформируем
cols = X.select_dtypes(np.number).columns
X[cols] = RobustScaler().fit_transform(X[cols])
"""
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

param_with_outliers = ['LotFrontage', 'LotArea', 'YearBuilt', 'GarageArea']

for param in param_with_outliers:
    X[param] = remove_outlier(X, param)
"""

X = pd.get_dummies(X)

#возвращаем тест и треин
X_train1 = X.loc[train.index]
X_test_sub = X.loc[test.index]

#убираем выбросы (НЕДОПИСАНО)


#выбирает треин/тест сет
train_size = 0.8
separator = round(len(X_train1.index)*train_size)


X_train, y_train = X_train1.iloc[0:separator], y.iloc[0:separator]
X_test, y_test = X_train1.iloc[separator:], y.iloc[separator:]


rfr = RandomForestRegressor(criterion='mse', max_features='auto',
                              min_samples_split=2, n_estimators=15)
#paremeters_rf = {"n_estimators" : [5, 10, 15, 20], "criterion" : ["mse" , "mae"], "min_samples_split" : [2, 3, 5, 10], 
#                 "max_features" : ["auto", "log2"]}
#grid_rf = GridSearchCV(model, paremeters_rf, verbose=1, scoring="r2")
#grid_rf.fit(X_train, y_train)
#print(grid_rf.best_params_)


gbr = ensemble.GradientBoostingRegressor(learning_rate=0.02, n_estimators=2000,
                                           max_depth=5, min_samples_split=2,
                                           loss='ls', max_features=35)

xgb = XGBRegressor(learning_rate=0.02,n_estimators=2000,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:squarederror', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.006)

lgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       learning_rate=0.05, 
                                       n_estimators=2000,
                                       max_bin=2000, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                       verbose=-1,
                                       )

cb = CatBoostRegressor(loss_function='RMSE', logging_level='Silent')

#rfr.fit(X_train, y_train)   
#preds = rfr.predict(X_test) 
#preds_test_rfr = rfr.predict(X_test_sub)
#mae_rfr = mean_absolute_error(y_test, preds)
#rmse_rfr = np.sqrt(mean_squared_error(y_test, preds))
#score_rfr = rfr.score(X_test, y_test) 

gbr.fit(X_train, y_train)   
preds = gbr.predict(X_test) 
preds_test_gbr = gbr.predict(X_test_sub)
mae_gbr = mean_absolute_error(y_test, preds)
rmse_gbr = np.sqrt(mean_squared_error(y_test, preds))
score_gbr = gbr.score(X_test, y_test)

lgbm.fit(X_train, y_train)   
preds = lgbm.predict(X_test) 
preds_test_lgbm = lgbm.predict(X_test_sub)
mae_lgbm = mean_absolute_error(y_test, preds)
rmse_lgbm = np.sqrt(mean_squared_error(y_test, preds))
score_lgbm = lgbm.score(X_test, y_test)

xgb.fit(X_train, y_train)   
preds = xgb.predict(X_test) 
preds_test_xgb = xgb.predict(X_test_sub)
mae_xgb = mean_absolute_error(y_test, preds)
rmse_xgb = np.sqrt(mean_squared_error(y_test, preds))
score_xgb = xgb.score(X_test, y_test)

cb.fit(X_train, y_train)   
preds = cb.predict(X_test) 
preds_test_cb = cb.predict(X_test_sub)
mae_cb = mean_absolute_error(y_test, preds)
rmse_cb = np.sqrt(mean_squared_error(y_test, preds))
score_cb = cb.score(X_test, y_test)


#тюн граентного буста (лернинг рей) оптимальній 0.05
"""
learning_rates = [0.75 ,0.5, 0.25, 0.1, 0.05, 0.01]

r2_results = []
rmse_results = []

for eta in learning_rates:
    model = ensemble.GradientBoostingRegressor(learning_rate=eta)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2_clf = r2_score(y_test, preds)
    rmse_clf = np.sqrt(mean_squared_error(y_test, preds))
    r2_results.append(r2_clf)
    rmse_results.append(rmse_clf)
    
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(learning_rates, r2_results, 'b', label='R^2')
line2, = plt.plot(learning_rates, rmse_results, 'r', label='RMSE')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Score')
plt.xlabel('learning_rates')
plt.show()
"""

#тюн граентного буста (н естимейторс) оптимальній больше 150
"""
n_estimators = [1, 2, 16, 32, 64, 100, 200, 500, 2000]

r2_results = []
rmse_results = []

for estimator in n_estimators:
    model = ensemble.GradientBoostingRegressor(n_estimators=estimator)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2_clf = r2_score(y_test, preds)
    rmse_clf = np.sqrt(mean_squared_error(y_test, preds))
    r2_results.append(r2_clf)
    rmse_results.append(rmse_clf)
    
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_estimators, r2_results, 'b', label='R^2')
line2, = plt.plot(n_estimators, rmse_results, 'r', label='RMSE')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Score')
plt.xlabel('n_estimators')
plt.show()
"""

#тюн граентного буста (глубина дерева) оптимальній (4) или 5
"""
max_depths = np.linspace(1, 10, 10, endpoint=True)

r2_results = []
rmse_results = []

for max_depth in max_depths:
    model = ensemble.GradientBoostingRegressor(max_depth=max_depth)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2_clf = r2_score(y_test, preds)
    rmse_clf = np.sqrt(mean_squared_error(y_test, preds))
    r2_results.append(r2_clf)
    rmse_results.append(rmse_clf)
    
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, r2_results, 'b', label='R^2')
line2, = plt.plot(max_depths, rmse_results, 'r', label='RMSE')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Score')
plt.xlabel('max_depths')
plt.show()
"""

#тюн граентного буста (органиченность выборки) оптимальній 0.2 - 0.3
"""
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)

r2_results = []
rmse_results = []

for min_samples_split in min_samples_splits:
    model = ensemble.GradientBoostingRegressor(min_samples_split=min_samples_split)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2_clf = r2_score(y_test, preds)
    rmse_clf = np.sqrt(mean_squared_error(y_test, preds))
    r2_results.append(r2_clf)
    rmse_results.append(rmse_clf)
    
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_splits, r2_results, 'b', label='R^2')
line2, = plt.plot(min_samples_splits, rmse_results, 'r', label='RMSE')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Score')
plt.xlabel('min_samples_splits')
plt.show()
"""

#тюн граентного буста (минимальная выборка в листе) все хуйня
"""

min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)

r2_results = []
rmse_results = []

for min_samples_leaf in min_samples_leafs:
    model = ensemble.GradientBoostingRegressor(min_samples_leaf=min_samples_leaf)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2_clf = r2_score(y_test, preds)
    rmse_clf = np.sqrt(mean_squared_error(y_test, preds))
    r2_results.append(r2_clf)
    rmse_results.append(rmse_clf)
    
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_leafs, r2_results, 'b', label='R^2')
line2, = plt.plot(min_samples_leafs, rmse_results, 'r', label='RMSE')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Score')
plt.xlabel('min_samples_leafs')
plt.show()
"""

#тюн граентного буста (фичеры для сплита) 20-25
"""

max_features = list(range(1,30))

r2_results = []
rmse_results = []

for max_feature in max_features:
    model = ensemble.GradientBoostingRegressor(max_features=max_feature)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2_clf = r2_score(y_test, preds)
    rmse_clf = np.sqrt(mean_squared_error(y_test, preds))
    r2_results.append(r2_clf)
    rmse_results.append(rmse_clf)
    
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_features, r2_results, 'b', label='R^2')
line2, = plt.plot(max_features, rmse_results, 'r', label='RMSE')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Score')
plt.xlabel('max_features')
plt.show()
"""
#mae_rfr=rmse_rfr=score_rfr=0
#мобираем сводную таблицу с параметрами для выбора моделей
model_performances = pd.DataFrame({
    "Model" : ["Gradient Boosting Regression", "XGBoost", "LGBM", "CatBoost"],
    "MAE" : [str(mae_gbr)[0:5], str(mae_xgb)[0:5], str(mae_lgbm)[0:5], str(mae_cb)[0:5]],
    "RMSE" : [str(rmse_gbr)[0:5], str(rmse_xgb)[0:5], str(rmse_lgbm)[0:5], str(rmse_cb)[0:5]],
    "Score" : [str(score_gbr)[0:5], str(score_xgb)[0:5], str(score_lgbm)[0:5], str(score_cb)[0:5]]
})

print("Sorted by MAE:")
print(model_performances.sort_values(by="MAE", ascending=False))

#комбинирование предиктов
def blend_models_predict(X):
    cb_pred = cb.predict(X)
    return ((0.3 * gbr.predict(X)) +  (0.25 * xgb.predict(X)) +  (0.1 * lgbm.predict(X)) + (0.4 * cb_pred))
#[0.35 0.35 0.3] best
   
subm = np.exp(blend_models_predict(X_test_sub))

"""
preds = model.predict(X_test) 
preds_test = model.predict(X_test_sub)
print('Score:', model.score(X_test, y_test)) 
print('MAE: ', mean_absolute_error(y_test, preds))

"""
submission = pd.DataFrame({'Id': X_test_sub.index,
                       'SalePrice': subm})
submission.to_csv("submission.csv", index=False)
