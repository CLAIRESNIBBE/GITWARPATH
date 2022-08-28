import sklearn
import csv
import os
import os.path
import pandas as pd
import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt
from warfit_learn import datasets, preprocessing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.svm import LinearSVR, SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import xgboost
from xgboost import XGBClassifier  # for extreme gradient boosting model
from xgboost import XGBRFRegressor
from xgboost import XGBRegressor
from warfit_learn.estimators import Estimator
from warfit_learn.evaluation import evaluate_estimators
from warfit_learn.metrics.scoring import confidence_interval
from scipy.stats import norm
from mlxtend.regressor import StackingCVRegressor
import warnings
import statistics

def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

warnings.filterwarnings("ignore")
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import ChangedBehaviorWarning
from sklearn.exceptions import DataConversionWarning
from cubist import Cubist

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=ChangedBehaviorWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.reset_option('all')
from sklearn.feature_selection import SelectFwe, f_regression, SelectPercentile
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, \
    MinMaxScaler, FunctionTransformer, Normalizer
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoLarsCV, Lasso, ElasticNet, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import BaggingClassifier, VotingRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, make_union
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import chi2, f_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.kernel_approximation import RBFSampler, Nystroem
from tpot.builtins import StackingEstimator, ZeroCount
# import tensorflow as tf
# from tensorflow import keras
from numpy import loadtxt
# from keras.models import Sequential
# from keras.layers import Dense
from copy import copy

def collect_Metrics(metrics, model, metric):
    container = []
    for i in range(len(metrics)):
        if metrics[metric][metrics['Estimator'].any() == model]:
            container.append(metrics[metric][metrics['Estimator'] == model].values)
    return container

def collect_Results(metrics, model, metric):
    container = []
    for i in range(len(metrics)):
        if (metrics[i]['model'] == model):
            container.append(metrics[i][metric])
    return container


def variance(metric):
    meanvalue = np.mean(metric)
    sumsquares = 0
    for i in range(len(metric)):
        core = abs(metric[i] - meanvalue)
        sumsquares += np.square(core)
    variance = sumsquares / ((len(metric) - 1))
    return variance


def std_deviation(metric):
    return np.sqrt(variance(metric))

def SList(series):
    return np.array(series.values.tolist())

def confintlimit95(metric):
    return 1.96 * np.sqrt(variance(metric) / len(metric))

def format_summary(df_res):
    df_summary = df_res.groupby(['Estimator']).mean()
    df_summary.reset_index(inplace=True)
    for alg in df_res['Estimator'].unique():
        for metric in ['PW20', 'MAE']:
            data = df_res[metric][df_res['Estimator'] == alg].values
            np.sort(data)
            df = pd.DataFrame(data, columns=[metric])
            lo1 = np.percentile(data, 2.5)
            lo, hi = confidence_interval(df_res[metric][df_res['Estimator'] == alg].values)
            mean = df_res[metric][df_res['Estimator'] == alg].mean()
            lo2 = mean - confintlimit95(data)
            hi2 = mean + confintlimit95(data)
            conf2 = f"{mean:.2f}({lo2:.2f}-{hi2:.2f})"
            print("new method", alg, metric, lo2, hi2, mean, conf2)
            for v in [mean, lo, hi]:
                if not (-10000 < v < 10000):
                    print('nan applied: ', alg, metric, lo, hi, mean)
                    mean, lo, hi = np.nan, np.nan, np.nan
                conf = f"{mean:.2f}({lo:.2f}-{hi:.2f})"
                print(alg, metric, lo, hi, mean, conf)
                df_summary[metric][df_summary['Estimator'] == alg] = conf
    return df_summary


def MLAR(trueval, predval):
    # mean log of accuracy ratio
    sum = 0
    for i in range(len(trueval)):
        sum += np.log(predval[i] / trueval[i])
    return (np.exp(sum / len(trueval)) - 1) * 100

def MALAR(trueval, predval):
    # mean absolute log of accuracy ratio
    sum = 0
    for i in range(len(trueval)):
        sum += abs(np.log(predval[i] / trueval[i]))
    return (np.exp(sum / len(trueval)) - 1) * 100

def RSquared(trueval, predval):
    true_mean = np.mean(trueval)
    topsum = 0
    lowersum = 0
    for i in range(len(trueval)):
        topsum += np.square((predval[i] - true_mean))
        lowersum += np.square(trueval[i] - true_mean)
    return topsum / lowersum * 100

def BSA(height, weight):
    return 0.007184 * height ** 0.725 * weight ** 0.425

    if False:
        df_WARPATH[current_metric] = np.array(collect_Metrics(metrics, 'WarPATH', current_metric))
        std = np.square(std_deviation(df_WARPATH[current_metric]))
        var = variance(df_WARPATH[current_metric])
        std_Dev.append({'model': 'WarPATH', 'metric': current_metric, 'SD': std, 'VAR': var})

def ConvertYesNo(variable):
    if variable == "Yes":
        return 1
    elif variable == "No":
        return 0

def MAEScore(true, predicted):
    return mean_absolute_error(true,predicted)

def PercIn20(true, predicted):
    patients_in_20 = 0
    for i in range(len(true)):
        if abs(true[i] - predicted[i]) < 0.2 * true[i]:
            patients_in_20 += 1
    return 100 * patients_in_20 / len(true)

def INRThree(targetINR):
    if (targetINR >= 2.5) & (targetINR <= 3.5):
        return 1
    else:
        return 0

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

def traineval(est: Estimator, grid, xtrain, ytrain, xtest, ytest, squaring):
    results = {'PW20': [], 'MAE': [], 'R2': []}
    res=[]
    print(f'\n{est.identifier}...')
    mae_scorer = make_scorer(MAEScore)
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    gridinitial = grid[est.identifier]
    search = GridSearchCV(est.estimator, gridinitial, scoring='neg_mean_absolute_error',cv=cv)
    gridresult= search.fit(xtrain, ytrain)
    predicts = search.best_estimator_.predict(xtest)
    if squaring:
        ytest = np.square(ytest)
        predicts = np.square(predicts)
    PW20 = PercIn20(ytest, predicts)
    MAE = mean_absolute_error(ytest, predicts)
    R2 = RSquared(ytest, predicts)
    results['PW20'] = [PW20]
    results['MAE'] = [MAE]
    results['R2'] = [R2]
    return results

def main():
    combinedata = False
    scaler = MinMaxScaler()
    dftemplate = pd.DataFrame()
    dfWarPath = pd.DataFrame()
    impNumber = 100
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    pd.set_option('expand_frame_repr', False)
    pd.set_option("display.max_rows", False)
    df = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MiceRWarPATHData.csv", ";")
    filesIWPC = []
    if combinedata == True:
        for root, dirs, files in os.walk(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MICESTATSMODEL"):
            for file in files:
                if file.endswith('.csv'):
                    filesIWPC.append(file)
    kcv = KFold(n_splits=10, random_state=1, shuffle=True)
    for imp in range(impNumber):
        counter = imp + 1
        dfcurrent = df.loc[df[".imp"] == counter]
        suffix = str(counter).zfill(3)
        dfcurrent.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\ImpWarPATH_" + suffix + ".csv", ";")
    filesImp = []
    if os.path.exists(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TRAINSPLIT" + ".csv"):
        if os.path.exists(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TESTSPLIT" + ".csv"):
            trainID = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TRAINSPLIT" + ".csv", ";")
            testID = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TESTSPLIT" + ".csv", ";")
    else:
      fixedtraintest = False
      while fixedtraintest == False:
        for root, dirs, files in os.walk(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations"):
            for file in files:
                if file.endswith('.csv') and 'TEST' not in file and 'TRAIN' not in file and "SPLIT" not in file:
                    # filesImp.append(file)
                    if not fixedtraintest:
                        filedf = pd.read_csv(root + '\\' + file, ";")
                        trainID, testID = train_test_split(filedf, test_size=0.2)
                        trainID.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TRAINSPLIT" + ".csv",";")
                        testID.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TESTSPLIT" + ".csv", ";")
                        fixedtraintest = True
    # for imp in range(impNumber):
    patients_train = []
    patients_train = trainID[".id"].to_list()

    dftrain = df[df['.id'] .isin(patients_train)]
    dftrain.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\Training\train" + suffix + ".csv", ";")
    for imp in range(impNumber):
        counter = imp + 1
        dftrainimp = dftrain.loc[df[".imp"] == counter]
        suffix = str(counter).zfill(3)
        dftrainimp.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\Training\train_" + suffix + ".csv",
                          ";")
    patients_test = []
    patients_test = testID[".id"].to_list()
    dftest = df[df['.id'].isin(patients_test)]
    dftest.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\Testing\test" + suffix + ".csv", ";")
    for imp in range(impNumber):
        counter = imp + 1
        dftestimp = dftest.loc[df[".imp"] == counter]
        suffix = str(counter).zfill(3)
        dftestimp.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\Testing\test_" + suffix + ".csv", ";")

    # for root, dirs, files in os.walk(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\Training\train"):
    counter = 0
    for root, dirs, files in os.walk(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations"):
        for file in files:
            if file.endswith('.csv') and (
                    "train_" not in file and "test_" not in file and "SPLIT" not in file and "TRAIN" not in file and "TEST" not in file) and "ImpWarPATH" in file:
                filedf = pd.read_csv(root + '\\' + file, ";")
                if "Status" not in filedf.columns:
                    filedf["Status"] = ""
                    counter = counter + 1
                for row in filedf.itertuples():
                    checkID = row[4]
                    rowindex = filedf.loc[filedf[".id"] == checkID].index.tolist()[0]  # OR row[0]
                    if checkID in patients_train:
                        filedf.loc[rowindex, 'Status'] = 'train'
                    elif checkID in patients_test:
                        filedf.loc[rowindex, 'Status'] = 'test'
                suffix = str(counter).zfill(3)
                filedf.to_csv(
                    r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\Split\ImpWarPATHSPLIT_" + suffix + ".csv",
                    ";")
                filesImp.append(
                    r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\Split\ImpWarPATHSPLIT_" + suffix + ".csv")
    results = []
    for file in filesImp:
        dfnew = pd.read_csv(file, ";")
        fileindex = filesImp.index(file)
        rootIWPC = root.replace("WarImputations", "MICESTATSMODELHIV\\")
        # rootIWPC = root.replace("WarImputations","MICESTATS125\\")
        if combinedata == True:
            IWPC_csv = rootIWPC + filesIWPC[fileindex]
            dfIWPC = pd.read_csv(IWPC_csv, ';')
            dfIWPC.drop(["Unnamed: 0"], axis=1, inplace=True)
        df = fileindex + 1
        dfmod = dfnew
        dfmod.drop(['Gender', 'Country_recruitment'], axis=1, inplace=True)
        # dfIWPC.drop(['AgeDecades'], axis =1, inplace = True)
        # dfIWPC.drop(['INR_Three'], axis=1, inplace=True)
        # dfmod["INR_Three"] = np.where(dfmod["Target_INR"] == "Three", 1, 0)
        dfmod["Target_INR"] = np.where(dfmod["Target_INR"] == "Three", 3.0,
                                       np.where(dfmod["Target_INR"] == "aTwo_point_five", 2.5, 2.0))
        dfmod["Target_INR"] = dfmod.apply(lambda x: INRThree(x["Target_INR"]), axis=1)
        dfmod["Target_INR"] = dfmod['Target_INR'].astype("float")
        dfmod["Inducer"] = dfmod.apply(lambda x: ConvertYesNo(x["Inducer_status"]), axis=1)
        dfmod["Amiodarone"] = dfmod.apply(lambda x: ConvertYesNo(x["Amiodarone_status"]), axis=1)
        dfmod["Smoker"] = dfmod.apply(lambda x: ConvertYesNo(x["Smoking_status"]), axis=1)
        dfmod["Indicationflag"] = dfmod.apply(lambda x: ConvertYesNo(x["Indication"]), axis=1)
        dfmod.drop(["Inducer_status", "Amiodarone_status", "Smoking_status", "Indication"], axis=1, inplace=True)
        # dfmod["AgeDecades"] = np.floor(dfmod["Age_years"] * 0.1).astype("int")
        dfmod["AgeYears"] = dfmod["Age_years"]
        dfmod['AgeYears'] = np.where((dfmod['AgeYears'] <= 18), 18, dfmod['AgeYears'])
        dfmod["HIVPositive"] = np.where(dfmod["HIV_status"] == "Positive", 1, 0)
        dfmod["HIVUnknown"] = np.where(dfmod["HIV_status"] == "Unknown", 1, 0)
        # dfIWPC["HIVPositive"]=0
        # dfIWPC["HIVUnknown"] = 0
        dfmod.drop(["HIV_status"], axis=1, inplace=True)
        # dfmod['BSA'] = dfmod.apply(lambda x: BSA(x["Height_cm"], x["Weight_kg"]), axis=1)
        # dfmod.drop(["Height_cm"], axis = 1, inplace = True)
        # dfmod.drop(["Weight_kg"], axis=1, inplace=True)
        dfmod.drop(["Unnamed: 0"], axis=1, inplace=True)
        dfmod.drop(["Unnamed: 0.1"], axis=1, inplace=True)
        dfmod.drop(["Age_years"], axis=1, inplace=True)
        dfmod.drop([".imp"], axis=1, inplace=True)
        # dfmod.drop([".id"], axis=1, inplace=True)
        # combinedata = True
        suffix = str(df).zfill(3)
        if combinedata == True:
           dfmod = dfmod.sample(frac=1)
           dfIWPC = dfIWPC.sample(frac=1)
           frames = [dfmod, dfIWPC]
           dfmod = pd.concat(frames)
           dfmod = dfmod.sample(frac=1)
           combfilename = "comb" + suffix
           dfmod.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\combinedata\\" + combfilename + ".csv", ";")
        else:
           filename = "dfWarfarin" + suffix
           dfmod.drop(["Unnamed: 0.1.1"], axis=1, inplace=True)
           dfmod.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\PreProcessed\\" + filename + ".csv", ";")
        if True:
            print("On imputation ", df)
            data = dfmod
            print(data.shape)
            test_size = 0.2
            data['Dose_mg_week'] = data['Dose_mg_week'].apply(np.sqrt)
            estimates = []
            target_column = 'Dose_mg_week'
            status_column = "Status"
            train = dfmod.loc[dfmod["Status"] == "train"]
            test = dfmod.loc[dfmod["Status"] == "test"]
            squaring = True
            train = train.drop([status_column], axis=1)
            test = test.drop([status_column], axis=1)
            x_cols = list(train.columns)
            targ_col = list(target_column)
            targetindex = x_cols.index(target_column)
            y_train = train[target_column].values
            x_train = train.drop([target_column], axis=1)
            y_test = test[target_column].values
            x_test = test.drop([target_column], axis=1)
            ytest_numpy = np.array(y_test)
            y_testnew = np.square(ytest_numpy)
            RunRandom = False
            if RunRandom == True:
               #Number of trees in random forest
               n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
               # Number of features to consider at every split
               max_features = ['auto', 'sqrt']
               # Maximum number of levels in tree
               max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
               max_depth.append(None)
               # Minimum number of samples required to split a node
               min_samples_split = [2, 5, 10]
               # Minimum number of samples required at each leaf node
               min_samples_leaf = [1, 2, 4]
               # Method of selecting samples for training each tree
               bootstrap = [True, False]
               random_grid = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'bootstrap': bootstrap}
               rf = RandomForestRegressor()
               # Random search of parameters, using 3 fold cross validation,
               # search across 100 different combinations, and use all available cores
               rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                           random_state=42, n_jobs=-1)
               rf_random.fit(x_train, y_train)
               print(rf_random.best_params_)
               print(rf_random.best_estimator_)
               base_model = RandomForestRegressor(n_estimators=10, random_state=42)
               base_model.fit(x_train, y_train)
               predicts = rf_random.predict(x_test)
               predictsnew = np.square(predicts)
               print(mean_absolute_error(y_testnew, predictsnew))
               base_accuracy = evaluate(base_model, x_test, y_testnew)
               best_random = rf_random.best_estimator_
               random_accuracy = evaluate(best_random, x_test, y_testnew)
               print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))
            # Create the parameter grid based on the results of random search
               param_grid = {
               'bootstrap': [True],
               'max_depth': [80, 90, 100, 110],
               'max_features': ['auto'],
               'min_samples_leaf': [3, 4, 5],
               'min_samples_split': [8, 10, 12],
               'n_estimators': [10000]
               }
               rf = RandomForestRegressor()
               # Instantiate the grid search model
               kcv = KFold(n_splits=10, random_state=1, shuffle=True)
               grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kcv, n_jobs=-1, verbose=2)
               # Fit the grid search to the data
               grid_search.fit(x_train, y_train)
               print(grid_search.best_params_)
               print(grid_search.best_estimator_)
               best_grid = grid_search.best_estimator_
               model = best_grid.fit(x_train,y_train)
               predicts = model.predict(x_test)
               predicts = np.square(predicts)
               metric = mean_absolute_error(predicts,y_testnew)
               print('MAE is ', metric)
               #grid_accuracy = evaluate(best_grid, x_test, y_testnew)
               predicts = grid_search.predict(x_test)
               predictsnew = np.square(predicts)
               print(mean_absolute_error(y_testnew, predictsnew))
               #print('Improvement of {:0.2f}%.'.format(100 * (grid_accuracy - base_accuracy) / base_accuracy))
               RF=RandomForestRegressor(max_depth=80, min_samples_leaf=5, max_features='sqrt',min_samples_split=12,
               n_estimators=2000)
               kcv = KFold(n_splits=10, random_state=1, shuffle=True)
               ab_RF = AdaBoostRegressor(RF, n_estimators=6, random_state=42)
               param_grid = {'n_estimators':  list(range(1, 10, 1)) }
               grid_search=GridSearchCV(estimator=ab_RF,param_grid=param_grid, cv=kcv,n_jobs=-1, verbose=2)
               grid_search.fit(x_train,y_train)
               best_grid= grid_search.best_estimator_
               print("best params are ", grid_search.best_params)
               print("best model is ", grid_search.best_estimator_)
               model = best_grid.fit(x_train,y_train)
               predicts = model.predict(x_test)
               predicts = np.square(predicts)
               metric = mean_absolute_error(predicts,y_testnew)
            RF = RandomForestRegressor(max_depth=80, min_samples_leaf=5, max_features='sqrt', min_samples_split=12,
            n_estimators=2000)
            ab_RF = AdaBoostRegressor(RF, n_estimators=6, random_state=42)
            boost = ab_RF.fit(x_train,y_train)
            predict = ab_RF.predict(x_test)
            predict = np.square(predict)
            mae = mean_absolute_error(predict, y_testnew)
            TuneNativeAPI = False
            if TuneNativeAPI == True:
               dmatrix = xgboost.DMatrix(x_train,y_train)
               params = {'objective': 'reg:squarederror'}
               cv_results = xgboost.cv(dtrain=dmatrix,
                                params=params,
                                nfold=10,
                                metrics={'rmse'})
               print('RMSE: %.2f' % cv_results['test-rmse-mean'].min())
               print(cv_results)

               params = {'n_estimators': [100, 500, 1000]}
               xgbr = XGBRegressor(base_score=0.5, booster='dart', colsample_bylevel=1,
                   colsample_bynode=1, colsample_bytree=0.3, gamma=0, gpu_id=-1,
                   importance_type='gain', interaction_constraints='',
                   learning_rate=0.005, max_delta_step=0, max_depth=1,
                   min_child_weight=1,  monotone_constraints='()',
                   n_estimators=1000, n_jobs=8, num_parallel_tree=25, random_state=20,
                   reg_alpha=0.2, reg_lambda=0.2, scale_pos_weight=1, seed=20,
                   subsample=0.3, tree_method='exact', validate_parameters=1,
                   verbosity=None)

               clf = DaskGridSearchCV(estimator=xgbr,
                                  param_grid=params,
                                  scoring='neg_mean_squared_error',
                                )
               clf.fit(x_train,y_train)
               print("Best parameters:", clf.best_params_)
               print("Lowest RMSE: ", (-clf.best_score_) ** (1 / 2.0))
               print("Best model:", clf.best_estimator_)
               xgRF = xgboost.XGBRFRegressor()
               grid = {'n_estimators':  list(range(10, 1000, 100)),
               'max_depth'   :  list(range(1,15)),
               'reg_lambda'  :  list(np.linspace(0.001, 1.0))}
               clf = DaskGridSearchCV(estimator=xgRF, param_grid=grid,scoring='neg_mean_squared_error', cv = kcv)
               clf.fit(x_train, y_train)
               print('Best estimates ', clf.best_params_)
               print('Best model ', clf.best_estimator_)

            TuneNativeAPI = False
            if TuneNativeAPI == True:
               dtrain = xgboost.DMatrix(x_train, label=y_train)
               #y_test = y_testnew
               dtest = xgboost.DMatrix(x_test, label=y_test)
               mean_train = np.mean(y_train)
               # Get predictions on the test set
               baseline_predictions = np.ones(y_test.shape) * mean_train
               mae_baseline = mean_absolute_error(y_test, baseline_predictions)
               print("Baseline MAE is {:.2f}".format(mae_baseline))

               params = {
               # Parameters that we are going to tune.
                   'max_depth': 1,
                   'min_child_weight': 1,
                   'eta': .3,
                   'subsample': 1,
                   'colsample_bytree': 1,
                   # Other parameters
                   'objective': 'reg:squarederror',
               }
               params['eval_metric'] = "mae"
               num_boost_round = 999
               model = xgboost.train(
                    params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=[(dtest, "Test")],
                    early_stopping_rounds=10
                )
               print("Best MAE: {:.2f} with {} rounds".format( model.best_score, model.best_iteration + 1))
               cv_results = xgboost.cv(
                    params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    seed=42,
                    nfold=5,
                    metrics={'mae'},
                    early_stopping_rounds=10
                )
               print(cv_results)
               print(cv_results['test-mae-mean'].min())
               gridsearch_params = [
                     (max_depth, min_child_weight)
               for max_depth in range(1, 12)
               for min_child_weight in range(1, 8)
               ]
               min_mae = float("Inf")
               best_params = None
               for max_depth, min_child_weight in gridsearch_params:
                   print("CV with max_depth={}, min_child_weight={}".format(
                       max_depth,
                       min_child_weight))
               # Update our parameters
                   params['max_depth'] = max_depth
                   params['min_child_weight'] = min_child_weight
                # Run CV
                   cv_results = xgboost.cv(
                       params,
                       dtrain,
                       num_boost_round=num_boost_round,
                       seed=42,
                       nfold=5,
                       metrics={'mae'},
                       early_stopping_rounds=10
                )
                   # Update best MAE
                   mean_mae = cv_results['test-mae-mean'].min()
                   boost_rounds = cv_results['test-mae-mean'].argmin()
                   print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
                   if mean_mae < min_mae:
                      min_mae = mean_mae
                      best_params = (max_depth, min_child_weight)
               print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
               params['max_depth'] = 1
               params['min_child_weight'] = 1
               gridsearch_params = [
               (subsample, colsample)
               for subsample in [i / 10. for i in range(1, 11)]
               for colsample in [i / 10. for i in range(1, 11)]
               ]
               min_mae = float("Inf")
               best_params = None
               # We start by the largest values and go down to the smallest
               for subsample, colsample in reversed(gridsearch_params):
                  print("CV with subsample={}, colsample={}".format(
                  subsample,
                  colsample))
               # We update our parameters
                  params['subsample'] = subsample
                  params['colsample_bytree'] = colsample
               # Run CV
                  cv_results = xgboost.cv(
                      params,
                      dtrain,
                      num_boost_round=num_boost_round,
                      seed=42,
                      nfold=5,
                      metrics={'mae'},
                      early_stopping_rounds=10
                     )
                # Update best score
                  mean_mae = cv_results['test-mae-mean'].min()
                  boost_rounds = cv_results['test-mae-mean'].argmin()
                  print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
                  if mean_mae < min_mae:
                     min_mae = mean_mae
                     best_params = (subsample, colsample)
               print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
               params['subsample'] = .3
               params['colsample_bytree'] = 1.0
               min_mae = float("Inf")
               best_params = None
               for eta in [.3, .2, .1,.09,.08,.07,.06, .05, .04, .03, .02, .01,.009,.008,.007,.006, .005, 0.004, 0.003, 0.002,0.001]:
                   print("CV with eta={}".format(eta))
                # We update our parameters
                   params['eta'] = eta
                   cv_results = xgboost.cv(
                        params,
                        dtrain,
                        num_boost_round=num_boost_round,
                        seed=42,
                        nfold=5,
                        metrics=['mae'],
                        early_stopping_rounds=100
                        )
               # Update best score
                   mean_mae = cv_results['test-mae-mean'].min()
                   boost_rounds = cv_results['test-mae-mean'].argmin()
                   print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
                   if mean_mae < min_mae:
                      min_mae = mean_mae
                      best_params = eta
               print("Best params: {}, MAE: {}".format(best_params, min_mae))
               params['eta'] = 0.3















if __name__ == "__main__":
    main()