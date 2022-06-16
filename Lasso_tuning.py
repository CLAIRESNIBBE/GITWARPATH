import copy
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
from sklearn.model_selection import GridSearchCV, RepeatedKFold, KFold, RandomizedSearchCV, RepeatedStratifiedKFold
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

def ExitSquareBracket(variable):
    stringvar = str(variable)
    if stringvar.find('[') >= 0 and stringvar.find(']') >=0:
        var1 = stringvar.replace('[','')
        var2 = var1.replace(']','')
        return var2
    else:
        return stringvar

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

def evaluate_models(models, x_train, x_test, y_train, y_test):
    # fit and evaluate the models
    scores = list()
    for _, model in models:
        # fit the model
        model.fit(x_train, y_train)
        # evaluate the model
        yhat = model.predict(x_test)
        mae = mean_absolute_error(y_test, yhat)
        # store the performance
        scores.append(-mae)
    # report model performance
    return scores

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy


def traineval(est: Estimator,  xtrain, ytrain, xtest, ytest, squaring):
    resultsdict = {'PW20': 0, 'MAE': 0, 'R2': 0}
    print(f'\n{est.identifier}...')
    mae_scorer = make_scorer(MAEScore)
    kcv = KFold(n_splits=10, random_state=1, shuffle=True)
    #ridinitial = grid[est.identifier]
    ytest_numpy = np.array(ytest)
    model = est.estimator
    fitted = model.fit(xtrain,ytrain)
    predict=fitted.predict(xtest)
    #search = GridSearchCV(est.estimator, gridinitial, scoring='neg_mean_absolute_error',cv=cv)
    #gridresult= search.fit(xtrain, ytrain)
    #redicts = search.best_estimator_.predict(xtest)
    if squaring:
        ytest = np.square(ytest)
        predict = np.square(predict)
    PW20 = PercIn20(ytest, predict)
    MAE = mean_absolute_error(ytest, predict)
    R2 = RSquared(ytest, predict)
    results2 = cross_val_score(model, xtrain, ytrain, cv=kcv, scoring='neg_mean_absolute_error')
    #print("Accuracy: %.3f%% (%.3f%%)" % (results2.mean() * 100.0, results2.std() * 100.0))
    resultsdict['PW20'] = [PW20]
    resultsdict['MAE'] = [MAE]
    resultsdict['R2'] = [R2]
    return resultsdict
def grid_search(params, reg, x_train, y_train, x_test, y_test):
    kfold = KFold(n_splits=5, shuffle=True, random_state=2)
    grid_reg = GridSearchCV(reg, params, scoring='neg_mean_squared_error', cv=kfold)
    grid_reg.fit(x_train, y_train)
    best_params = grid_reg.best_params_
    print("Best params:", best_params)
    best_score = np.sqrt(-grid_reg.best_score_)
    print("Best score:", best_score)
    predicted = grid_reg.predict(x_test)
    predicted = np.square(predicted)
    mae = mean_absolute_error(y_test, predicted)
    print("MAE:", mae)


import numpy as np
import pandas as pd


def get_grid_df(fitted_gs_estimator):
    res_dict = fitted_gs_estimator.cv_results_

    results_df = pd.DataFrame()
    for key in res_dict.keys():
        results_df[key] = res_dict[key]

    return results_df


def group_report(results_df):
    param_cols = [x for x in results_df.columns if 'param' in x and x != 'params']
    focus_cols = param_cols + ['mean_test_score']

    print("Grid CV Report")

    output_df = pd.DataFrame(columns=['param_type', 'param_set',
                                      'mean_score', 'mean_std'])
    cc = 0
    for param in param_cols:
        for key, group in results_df.groupby(param):
            output_df.loc[cc] = (param, key, group['mean_test_score'].mean(), group['mean_test_score'].std())
            cc += 1
    return output_df


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
    if  True:
        for root, dirs, files in os.walk(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MICESTATSMODEL"):
            for file in files:
                if file.endswith('.csv'):
                    filesIWPC.append(file)

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
            trainDF = pd.DataFrame(trainID)
            trainSize = len(trainDF)
    else:
        fixedtraintest = False
        while fixedtraintest == False:
            for root, dirs, files in os.walk(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations"):
                for file in files:
                    if file.endswith('.csv') and 'TEST' not in file and 'TRAIN' not in file and "SPLIT" not in file:
                        # filesImp.append(file)
                        if not fixedtraintest:
                            filedf = pd.read_csv(root + '\\' + file, ";")
                            trainID, testID = train_test_split(filedf, test_size=0.3)
                            trainSize = len(trainID)
                            trainID.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TRAINSPLIT" + ".csv",
                                           ";")
                            testID.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TESTSPLIT" + ".csv",
                                          ";")
                            fixedtraintest = True
        # for imp in range(impNumber):
    patients_train = []
    patients_train = trainID[".id"].to_list()

    dftrain = df[df['.id'].isin(patients_train)]
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
        if combinedata == True:
            rootIWPC = root.replace("WarImputations\\Training", "MICESTATSMODELHIV\\")
            IWPC_csv = rootIWPC + filesIWPC[fileindex]
            IWPCDF = pd.read_csv(IWPC_csv,';')
            sampleSize = int(round(trainSize) * 0.25)
            dfIWPC = IWPCDF.sample(n=sampleSize)
            dfIWPC["Status"] = "train"
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
        if combinedata == True:
            dfIWPC["Target_INR"] = dfIWPC.apply(lambda x: INRThree(x["Target_INR"]), axis=1)
            dfIWPC["Target_INR"] = dfIWPC['Target_INR'].astype("float")
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
        dfmod.drop([".id"], axis=1, inplace=True)
        dfmod.drop(["Unnamed: 0.1.1"], axis=1, inplace=True)
        combinedata = False
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
            dfmod.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\PreProcessed\\" + filename + ".csv", ";")
        if df==1:
            print("On imputation ", df)
            data = dfmod
            print(data.shape)
            data['Dose_mg_week'] = data['Dose_mg_week'].apply(np.sqrt)
            sc = StandardScaler()
            target_column = 'Dose_mg_week'
            status_column = "Status"
            train = dfmod.loc[dfmod["Status"] == "train"]
            test = dfmod.loc[dfmod["Status"] == "test"]
            squaring = True
            train = train.drop([status_column], axis=1)
            test = test.drop([status_column], axis=1)
            y_train = train[target_column].values
            x_train = train.drop([target_column], axis=1)
            x_train = sc.fit_transform(x_train)
            y_test = test[target_column].values
            x_test = test.drop([target_column], axis=1)
            x_test = sc.fit_transform(x_test)
            # Use the random grid to search for best hyperparameters
            # First create the base model to tune
            if False:
                RFR = RandomForestRegressor(max_depth=120, max_features=3, min_samples_leaf=4,
                                           min_samples_split=12, n_estimators=100)
                #grid = {'n_estimators': [10, 50, 100, 500],'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0]}
                grid = {'n_estimators': [10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 500], 'learning_rate': [0.0001, 0.001, 0.005, 0.01, 0.015,0.1, 1.0]}
                #grid['n_estimators'] = [10, 50, 100, 500]
                #grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
                # define the evaluation procedure
                kfold = KFold(n_splits=10, shuffle=True, random_state=2)
                # define the grid search procedure
                grid_search = GridSearchCV(estimator=AdaBoostRegressor(RFR), param_grid=grid, n_jobs=-1, cv=kfold, verbose=3)
                # execute the grid search
                grid_result = grid_search.fit(x_train,y_train)
                # summarize the best score and configuration
                print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
                # summarize all scores that were evaluated
                means = grid_result.cv_results_['mean_test_score']
                stds = grid_result.cv_results_['std_test_score']
                params = grid_result.cv_results_['params']
                for mean, stdev, param in zip(means, stds, params):
                    print("%f (%f) with: %r" % (mean, stdev, param))
                print("end of tuning for AdaBoostRegressor")

            rf= RandomForestRegressor()
            param_grid = {
                'bootstrap': [True],
                'max_depth': [90, 100, 110, 115, 120, 125, 130, 135],
                'max_features': [2, 3, 4, 5],
                'min_samples_leaf': [3, 4, 5, 6],
                'min_samples_split': [8, 9, 10, 11, 12, 13, 4],
                'n_estimators': [80, 90, 100, 110, 120, 130, 140, 200, 300, 1000]
            }

            kfold = KFold(n_splits=5, shuffle=True, random_state=2)
            grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kfold, n_jobs=-1, verbose=3)
            grid_result = grid.fit(x_train, y_train)
            # summarize the best score and configuration
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            # summarize all scores that were evaluated
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
            print("end of tuning for RandomForestRegressor")

            kfold = KFold(n_splits=5, shuffle=True, random_state=2)
            param_grid = {
                'hidden_layer_sizes': [(10,10),(15,15),(20,20),(21,21),(22,22),(23,23),(24,24),(25,25),(26,26),(27,27),(28,28),(29,29),(30,30)],
                'max_iter': [2000,3000,4000,5000,6000,7000],
                'activation': [ 'relu'],
                'solver': ['adam'],
                'alpha': [0.0001, 0.0003,0.0004,0.0005,0.001,0.003,0.004,0.005,0.01,0.015, 0.02, 0.025, 0.03],
                'learning_rate': [ 'adaptive'],
            }
            grid = GridSearchCV(mlp_reg, param_grid, n_jobs=-1, cv=kfold, verbose=3)
            minscale = MinMaxScaler()
            x_train_scaled = minscale.fit_transform(x_train)
            grid_result = grid.fit(x_train_scaled, y_train)
            # summarize the best score and configuration
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            # summarize all scores that were evaluated
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
            NN2 = MLPRegressor()
            NN = MLPRegressor(hidden_layer_sizes=(10,), activation="relu", random_state=1, max_iter=2000)
            estimates.append(Estimator(NN, 'NN'))
            estimates.append(Estimator(NN, 'NN2'))






            rf = RandomForestRegressor()
            #rf.fit(x_train,y_train)
            #predicted = np.square(rf.predict(x_test))
            #maepredict = mean_absolute_error(y_test,predicted)
            #print('MAE for base model is ', maepredict)
            if False:
                # Number of trees in random forest
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
                # Create the random grid
                random_grid = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'bootstrap': bootstrap}

                # Random search of parameters, using 3 fold cross validation,
                # search across 100 different combinations, and use all available cores
                rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                           random_state=42, n_jobs=-1)
                # Fit the random search model
                rf_random.fit(x_train,y_train)
                predict2 = np.square(rf_random.predict(x_test))
                maepredict2 = mean_absolute_error(y_test, predict2)
                print('MAE after randomized search is ', maepredict2)

            # Create the parameter grid based on the results of random search
            param_grid = {
             'bootstrap': [True],
             'max_depth': [90, 100, 110,115,120,125,130,135],
             'max_features': [2, 3, 4, 5],
             'min_samples_leaf': [3, 4, 5,6],
             'min_samples_split': [8,9, 10, 11,12,13,4],
             'n_estimators': [80,90,100,110, 120,130, 140,200, 300, 1000]
             }

            kfold = KFold(n_splits=5, shuffle=True, random_state=2)
            grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kfold, n_jobs=-1, verbose=3)
            grid_result = grid.fit(x_train, y_train)
            # summarize the best score and configuration
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            # summarize all scores that were evaluated
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
            print("end of tuning for RandomForestRegressor")






            #print(grid_search.best_params_)
            #    predict2 = np.square(grid_search.predict(x_test))
            #    maepredict2 = mean_absolute_error(y_test, predict2)
            #    print('MAE after randomized search is ', maepredict2)
            #    RF = RandomForestRegressor(max_depth=120, max_features=3, min_samples_leaf=4,
            #                           min_samples_split=12, n_estimators=100)
            #hyperparameter_space = {'n_estimators': list(range(1, 102, 2)),
            #                        'learning_rate': [0.01, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}

            #RF = RandomForestRegressor(max_depth=120, max_features=3, min_samples_leaf=4,
            #                           min_samples_split=12, n_estimators=100)

            #hyperparameter_space = {'n_estimators': list(range(5,500,10)),
            #                        'learning_rate': [ 0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1, 0.11, 0.12, 0.13,0.14, 0.15,0.2,0.3,0.4,0.5]}
            #gs = GridSearchCV(AdaBoostRegressor(base_estimator=RF),
            #                  param_grid=hyperparameter_space,
            #                  cv=5, n_jobs=-1, verbose=3)
            #grid_result = gs.fit(x_train, y_train)
            # summarize the best score and configuration
            #print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            # summarize all scores that were evaluated
            #means = grid_result.cv_results_['mean_test_score']
            #stds = grid_result.cv_results_['std_test_score']
            #params = grid_result.cv_results_['params']
            #for mean, stdev, param in zip(means, stds, params):
            #    print("%f (%f) with: %r" % (mean, stdev, param))





            #ada_best = copy.deepcopy(pre_gs_inst.best_params_)
            #ada_best['n_estimators'] = 3000
            #ABRF = AdaBoostRegressor(base_estimator=RF,
            #                         loss='linear', learning_rate= 0.045, n_estimators=50)

            #ABRF.fit(x_train, y_train)
            #y_pred = np.square(ABRF.predict(x_test))
            #print("MAE : ", mean_absolute_error(y_test, y_pred))
            #print("MAPE : ", (np.abs(y_test - y_pred) / y_test).mean())




            #results_df = get_grid_df(pre_gs_inst)
            #group_report(results_df)





            KNN2 = KNeighborsRegressor()
            hp_candidates = [
                {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'weights': ['uniform', 'distance'],
                 'p': [1, 2, 5]}]
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            grid = GridSearchCV(estimator=KNN2,
                                param_grid=hp_candidates,
                                cv=cv,
                                verbose=1,
                                scoring='neg_mean_squared_error',
                                return_train_score=True)
            grid.fit(x_train, y_train)
            predicted =np.square(grid.predict(x_test))
            meanabsoluterror = mean_absolute_error(y_test, predicted)
            #print('MAE ',meanabsoluterror)

            model = XGBRegressor( n_estimators=5000,missing=-999.0)
            model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)],early_stopping_rounds = 100)
            #ypred = model.predict(x_test)
            #ypred = np.square(ypred)
            #meanabsoluterror = mean_absolute_error(y_test, ypred)
            #print(meanabsoluterror)
            if True:
                results = model.evals_result()
                plt.figure(figsize=(10, 7))
                plt.plot(results["validation_0"]["rmse"], label="Training loss")
                plt.plot(results["validation_1"]["rmse"], label="Validation loss")
                plt.axvline(21, color="gray", label="Optimal tree number")
                plt.xlabel("Number of trees")
                plt.ylabel("Loss")
                plt.legend()






            print('end of part 1')
            #params = {'max_depth': [1, 2, 3, 4, 6, 7, 8],'n_estimators': [10]}
            #params = {'max_depth': [1, 2, 3],
            #          'min_child_weight': [1, 2, 3, 4, 5],
            #          'n_estimators': [10]}
            #params = {'max_depth': [2],
            #          'min_child_weight': [2, 3],
            #          'subsample': [0.5, 0.6, 0.7, 0.8,0.9],
            #         'n_estimators': [10, 50]}
            #params = {'max_depth': [1],
            #          'min_child_weight': [1, 2, 3],
            #          'subsample': [0.6, 0.7, 0.8],
            #          'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
            #          'n_estimators': [50]}
            #params={'max_depth': [1],
            #                    'min_child_weight': [13],
            #                    'subsample': [.8],
            #                    'colsample_bytree': [1],
            #                    'colsample_bylevel': [0.6, 0.7, 0.8,
            #                                          0.9, 1],
            #                    'colsample_bynode': [0.6, 0.7, 0.8,
            #                                         0.9, 1],
            #                    'n_estimators': [50]}
            params = {'max_depth': [3, 6, 10],
                      'learning_rate': [0.01, 0.05, 0.1],
                      'n_estimators': [100, 500, 1000],
                      'colsample_bytree': [0.3, 0.7]}


            grid_search(params,XGBRegressor(missing=-999.0),x_train,y_train,x_test,y_test)



            #model = XGBRegressor()
            #model.fit(x_train, y_train)
            predictedfirst = np.square(model.predict(x_test))
            maefirst = mean_absolute_error(y_test,predictedfirst)
            print(maefirst)
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            params = {'max_depth': [1, 2, 3, 4, 6, 7, 8],
                      'n_estimators': [31],
                      }

            grid = GridSearchCV(estimator=model, param_grid=params, scoring='neg_mean_squared_error',cv = cv )
            grid.fit(x_train, y_train)
            print('best_estimator_',grid.best_estimator_)
            print('best_params_', grid.best_params_)
            predicted = np.square(grid.predict(x_test))
            meanabsoluterror = mean_absolute_error(y_test,predicted)
            print(meanabsoluterror)






if __name__ == "__main__":
    main()

