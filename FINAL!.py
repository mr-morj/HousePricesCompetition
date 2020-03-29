import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax


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
X[cols] = X.groupby("Neighborhood")[cols].transform(lambda x: x.fillna(x.mean()))

#пропущенніе данніе проверка
nans = X.isna().sum().sort_values(ascending=False)
nans = nans[nans > 0]
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(nans.index, nans.values)
ax.set_ylabel("No. of missing values")
ax.xaxis.set_tick_params(rotation=90)
plt.show()

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = []
for i in X.columns:
    if X[i].dtype in numeric_dtypes:
        numerics2.append(i)

skew_features = X[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

for i in skew_index:
    X[i] = boxcox1p(X[i], boxcox_normmax(X[i] + 1))
#rfntujhbb
X["TotalSF"] = X["GrLivArea"] + X["TotalBsmtSF"]
X["TotalPorchSF"] = X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]
X["TotalBath"] = X["FullBath"] + X["BsmtFullBath"] + 0.5 * (X["BsmtHalfBath"] + X["HalfBath"])
X['Total_sqr_footage'] = X['BsmtFinSF1'] + X['BsmtFinSF2'] + X['1stFlrSF'] + X['2ndFlrSF']

cols = ["MSSubClass", "YrSold"]
X[cols] = X[cols].astype("category")

X = X.drop(['Utilities', 'Street', 'PoolQC', ], axis=1)





#трасформируем
cols = X.select_dtypes(np.number).columns
X[cols] = RobustScaler().fit_transform(X[cols])
#упрощаем данніе

#X['haspool'] = X['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
#X['has2ndfloor'] = X['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
#X['hasgarage'] = X['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
#X['hasbsmt'] = X['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
#X['hasfireplace'] = X['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
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

overfit = []
for i in X_train1.columns:
    counts = X_train1[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 99.94:
        overfit.append(i)

overfit = list(overfit)
X_train1 = X_train1.drop(overfit, axis=1).copy()
X_test_sub = X_test_sub.drop(overfit, axis=1).copy()
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

xgb = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                       max_depth=3, min_child_weight=0,
                       gamma=0, subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:squarederror', nthread=-1,
                       scale_pos_weight=1, seed=27,
                       reg_alpha=0.00006)

lgbm = LGBMRegressor(objective='regression',
                         num_leaves=4,
                         learning_rate=0.01,
                         n_estimators=5000,
                         max_bin=200,
                         bagging_fraction=0.75,
                         bagging_freq=5,
                         bagging_seed=7,
                         feature_fraction=0.2,
                         feature_fraction_seed=7,
                         verbose=-1,
                                       )

cb = CatBoostRegressor(loss_function='RMSE', logging_level='Silent')



def mean_cross_val(model, X, y):
    score = cross_val_score(model, X, y, cv=5)
    mean = score.mean()
    return mean
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
cv_gbr = mean_cross_val(gbr, X_train1, y)

lgbm.fit(X_train, y_train)   
preds = lgbm.predict(X_test) 
preds_test_lgbm = lgbm.predict(X_test_sub)
mae_lgbm = mean_absolute_error(y_test, preds)
rmse_lgbm = np.sqrt(mean_squared_error(y_test, preds))
score_lgbm = lgbm.score(X_test, y_test)
cv_lgbm = mean_cross_val(lgbm, X_train1, y)

"""
xgb.fit(X_train, y_train)   
preds = xgb.predict(X_test) 
preds_test_xgb = xgb.predict(X_test_sub)
mae_xgb = mean_absolute_error(y_test, preds)
rmse_xgb = np.sqrt(mean_squared_error(y_test, preds))
score_xgb = xgb.score(X_test, y_test)
cv_xgb = mean_cross_val(xgb, X_train1, y)
"""

cb.fit(X_train, y_train)   
preds = cb.predict(X_test) 
preds_test_cb = cb.predict(X_test_sub)
mae_cb = mean_absolute_error(y_test, preds)
rmse_cb = np.sqrt(mean_squared_error(y_test, preds))
score_cb = cb.score(X_test, y_test)
cv_cb = mean_cross_val(cb, X_train1, y)

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
    "CV(5)" : [str(cv_gbr)[0:5], 'NaN', str(cv_lgbm)[0:5], str(cv_cb)[0:5]],
    "MAE" : [str(mae_gbr)[0:5], 'NaN', str(mae_lgbm)[0:5], str(mae_cb)[0:5]],
    "RMSE" : [str(rmse_gbr)[0:5], 'NaN', str(rmse_lgbm)[0:5], str(rmse_cb)[0:5]],
    "Score" : [str(score_gbr)[0:5], 'NaN', str(score_lgbm)[0:5], str(score_cb)[0:5]]
})

print("Sorted by Score:")
print(model_performances.sort_values(by="Score", ascending=False))

#комбинирование предиктов

sub_1 = pd.read_csv('House_Prices_submit.csv')


def blend_models_predict(X, a, b, c, d):
    cb_pred = cb.predict(X)
    sb = np.log(np.array(sub_1.iloc[:,1]))

    
    return ((a * gbr.predict(X)) +  (b * sb) +  (c * lgbm.predict(X)) + (d * cb_pred))
#[0.35 0.35 0.3] best
"""
f = open('Comb.txt', 'w')

for i1 in np.arange(0, 1.1, 0.05):
    iter=0
    print('I1=', round(i1,2))
    for i2 in np.arange(0, 1.1-i1, 0.05):
        for i3 in np.arange(0, 1.1-i1-i2, 0.05):
            i1 = round(i1, 2)
            i2 = round(i2, 2)
            i3 = round(i3, 2)
            i4 = 1 - i1 - i2 - i3
            i4 = round(i4, 2)
            if (i1+i2+i3+i4<=1) & (i4>=-0.0):
                subm = (blend_models_predict(X_test, i1, i2, i3, i4))
                mae_comb = mean_absolute_error(y_test, subm)
                rmse_comb = np.sqrt(mean_squared_error(y_test, subm))

                f.write('I1 ='+str(i1)+' I2 ='+str(i2)+' I3 ='+str(i3)+' I4 ='+str(i4)+'\n')
                f.write('MAE ='+ str(mae_comb)+'\n')
                f.write('RMSE ='+ str(rmse_comb)+'\n')

                f.write('-----------------\n')
                iter=iter+1
    print('Iteration ', iter)
    print('-----------------')
f.close()  

"""
subm = np.exp(blend_models_predict(X_test_sub, 0.1, 0.4, 0.1, 0.4))

submission = pd.DataFrame({'Id': X_test_sub.index,
                       'SalePrice': subm})

q1 = submission['SalePrice'].quantile(0.0042)
q2 = submission['SalePrice'].quantile(0.99)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

submission.to_csv("submission.csv", index=False)
