import sklearn
import time
import math
import csv
import os
import os.path
import plotnine
import pandas as pd
from pandas import MultiIndex
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import numpy as np
from tabulate import tabulate
from itertools import combinations, permutations, product
from os.path import isfile, join
from matplotlib import pyplot as plt
from warfit_learn import datasets, preprocessing
from lineartree import LinearForestRegressor
from lineartree import LinearTreeRegressor
from lineartree import LinearBoostRegressor
from scipy.stats import norm, iqr, scoreatpercentile
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from sklearn.linear_model import LinearRegression, Ridge, Perceptron, SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.svm import LinearSVR, SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import RandomizedSearchCV, learning_curve
from sklearn.model_selection import GridSearchCV, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import xgboost
from xgboost import XGBClassifier  # for extreme gradient boosting model
from xgboost import XGBRFRegressor
from xgboost import XGBRegressor
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from hyperopt.fmin import space_eval
from hyperopt.pyll.base import scope
from hpsklearn import HyperoptEstimator
from hpsklearn import any_classifier
from hpsklearn import any_preprocessing
from hpsklearn import any_regressor, random_forest_regressor, xgboost_regression, k_neighbors_regressor, \
    linear_regression, gradient_boosting_regressor
from hpsklearn import ada_boost_regressor, decision_tree_regressor, svr, mlp_regressor, lasso, ridge, elastic_net, \
    lightgbm_regression

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
from sklearn.metrics import r2_score

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
from numpy import loadtxt
from copy import copy


def ExitSquareBracket(variable):
    stringvar = str(variable)
    if stringvar.find('[') >= 0 and stringvar.find(']') >= 0:
        var1 = stringvar.replace('[', '')
        var2 = var1.replace(']', '')
        var2 = var2.replace("'", "")
        return var2
    else:
        return stringvar


def collect_Metrics(metrics, model, metric):
    container = []
    for i in range(len(metrics)):
        if metrics[metric][metrics['Estimator'].any() == model]:
            container.append(metrics[metric][metrics['Estimator'] == model].values)
    return container


def collect_Results(model, metric):
    container = []
    if Estimator == model:
        container.append(metric)
    return container


def variance(metric):
    meanvalue = np.mean(metric)
    sumsquares = 0
    for i in range(len(metric)):
        core = abs(metric[i] - meanvalue)
        sumsquares += np.square(core)
    if len(metric) == 1:
        variance = 0
    else:
        variance = sumsquares / ((len(metric) - 1))
    return variance


def std_deviation(metric):
    return np.sqrt(variance(metric))


def SList(series):
    return np.array(series.values.tolist())


def confintlimit95(metric,mean):
    return 1.96 * np.sqrt(variance(metric,mean) / len(metric))


def TrainOrTest(patientID, TrainList, TestList):
    TrainDF = pd.DataFrame(TrainList.sort())
    TestDF = pd.DataFrame(TestList.sort())
    if (patientID in TrainList):
        return 'train'
    elif (patientID in TestList):
        return 'test'


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


def ConvertYesNo(variable):
    if variable == "Yes":
        return 1
    elif variable == "No":
        return 0


def MAEScore(true, predicted):
    return mean_absolute_error(true, predicted)


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


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


def ExitSquareBracket(variable):
    stringvar = str(variable)
    if stringvar.find('[') >= 0 and stringvar.find(']') >= 0:
        var1 = stringvar.replace('[', '')
        var2 = var1.replace(']', '')
        var2 = var2.replace("'", "")
        return var2
    else:
        return stringvar


def collect_Metrics(metrics, model, metric):
    container = []
    for i in range(len(metrics)):
        if metrics[metric][metrics['Estimator'].any() == model]:
            container.append(metrics[metric][metrics['Estimator'] == model].values)
    return container


def collect_Results(model, metric):
    container = []
    if Estimator == model:
        container.append(metric)
    return container


def variance(metric,mean):
    #meanvalue = np.nanmean(metric)
    meanvalue = mean
    sumsquares = 0
    for i in range(len(metric)):
        core = abs(metric[i] - meanvalue)
        sumsquares += np.square(core)
    if len(metric) == 1:
        variance = 0
    else:
        variance = sumsquares / ((len(metric) - 1))
    return variance

def std_deviation(metric):
    return np.sqrt(variance(metric))

def SList(series):
    return np.array(series.values.tolist())

def confintlimit95(metric,mean):
    return 1.96 * np.sqrt(variance(metric,mean) / len(metric))

def TrainOrTest(patientID, TrainList, TestList):
    TrainDF = pd.DataFrame(TrainList.sort())
    TestDF = pd.DataFrame(TestList.sort())
    if (patientID in TrainList):
        return 'train'
    elif (patientID in TestList):
        return 'test'

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


def ConvertYesNo(variable):
    if variable == "Yes":
        return 1
    elif variable == "No":
        return 0


def MAEScore(true, predicted):
    return mean_absolute_error(true, predicted)


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
        mae = MAEScore(y_test, yhat)
        # store the performance
        scores.append(-mae)
    # report model performance
    return scores

def dropColumn(IWPCparam, columnname, dfColumns, dfmod, dfIWPC):
    if columnname in dfColumns:
        if IWPCparam == "IWPC":
          dfIWPC.drop([columnname],axis=1,inplace=True)
        else:
          dfmod.drop([columnname], axis=1, inplace=True)



#def traineval(est, estimates, xtrain, ytrain, xtest, ytest, squaring, df, randomseed):
def traineval(est: Estimator, xtrain, ytrain, xtest, ytest, squaring, df, randomseed):
    suffix = str(df).zfill(3)
    resultsdict = {'PW20': 0, 'MAE': 0, 'R2': 0, 'Time': ''}
    ml_learner = est
    print(f'\n{ml_learner}...')
    modelID = ml_learner
    kfolds = KFold(n_splits=10, shuffle=True, random_state=randomseed)
    model = est.estimator
    xtraindf = pd.DataFrame(xtrain)
    # xtrain = np.any(np.isnan(xtrain))
    # xtrain = np.all(np.isfinite(xtrain))
    # xtraindf = pd.DataFrame(xtrain)
    # ytrain = np.any(np.isnan(ytrain))
    # ytrain = np.all(np.isfinite(ytrain))
    ytraindf = pd.DataFrame(ytrain)
    # xtest = np.any(np.isnan(xtest))
    xtestdf = pd.DataFrame(xtest)
    # ytest = np.any(np.isnan(ytest))
    # ytest = np.all(np.isfinite(ytest))
    ytestdf = pd.DataFrame(ytest)
    modelID = est.identifier
    xtraindf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\HYPERBASEDATA\xtrain_" + modelID + suffix + ".csv", ";")
    xtestdf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\HYPERBASEDATA\xtest_" + modelID + suffix + ".csv", ";")
    ytraindf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\HYPERBASEDATA\ytrain_" + modelID + suffix + ".csv", ";")
    ytestdf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\HYPERBASEDATA\ytest_" + modelID + suffix + ".csv", ";")
    modelcurrent = HyperoptEstimator(regressor=model, verbose=True)
    start = time.time()
    modelcurrent.fit(xtrain, ytrain)
    end = time.time()
    timeElapsed = end - start
    best = modelcurrent.best_model()
    fitted = best['learner'].fit(xtrain, ytrain)
    bestloss = modelcurrent._best_loss
    ypred = fitted.predict(xtest)
    if squaring:
      ypred2 = np.square(ypred)
      ytest2 = np.square(ytest)
    paramdict = best['learner'].get_params()
    paramdict = {k: [v] for k, v in paramdict.items()}
    dfHyperCurrent = pd.DataFrame(paramdict)
    dfHyperCurrent['imputation'] = df
    dfHyperCurrent['bestloss'] = bestloss
    dfHyperCurrent['mae'] = MAEScore(ytest2, ypred2)
    ml_learner = modelID
    suffix = str(df).zfill(3)
    randstring = "_" + str(randomseed) + "_"
    if df == 1:
        dfHyper = pd.DataFrame()
    else:
        dfpre = df - 1
        suffixpre = str(dfpre).zfill(3)
        dfHyper = pd.read_csv(
            r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\HYPEROPTHYPERPARAMETERS\IWPC\model_" + ml_learner + randstring + suffixpre + ".csv",
            ";")
    frames = (dfHyper, dfHyperCurrent)
    dfHyper = pd.concat(frames)
    dfHyper["model"] = ml_learner
    if "Unnamed: 0" in dfHyper.columns:
        dfHyper.drop(["Unnamed: 0"], axis=1, inplace=True)
    if "Unnamed: 0.1" in dfHyper.columns:
        dfHyper.drop(["Unnamed: 0.1"], axis=1, inplace=True)
    dfHyper.to_csv(
        r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\HYPEROPTHYPERPARAMETERS\IWPC\model_" + ml_learner + randstring + suffix + ".csv", ";")
    MAE = MAEScore(ytest2, ypred2)
    PW20 = PercIn20(ytest2, ypred2)
    dfHyper = pd.read_csv(
        r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\HYPEROPTHYPERPARAMETERS\IWPC\model_" + ml_learner + randstring + suffix + ".csv", ";")
    dfHyper.loc[df - 1, "mae"] = MAE
    dfHyper.loc[df - 1, "pw20"] = PW20
    dfHyper.to_csv(
        r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\HYPEROPTHYPERPARAMETERS\IWPC\model_" + ml_learner + randstring + suffix + ".csv", ";")
    R2 = RSquared(ytest2, ypred2)
    PW20 = PercIn20(ytest2, ypred2)
    resultsdict['PW20'] = [PW20]
    resultsdict['MAE'] = [MAE]
    resultsdict['R2'] = [R2]
    resultsdict['Time'] = [timeElapsed]
    return resultsdict


def main():
    # dfHyper = pd.DataFrame()
    combinedata = True
    scaler = MinMaxScaler()
    fileName1 = "AllImputations.csv"
    fileName1 = fileName1.upper()
    fileName2 = 'IMPWARPATHSUPER.CSV'
    filescheck = []
    filescheck.append(fileName1)
    filescheck.append(fileName2)
    dftemplate = pd.DataFrame()
    dfWarPath = pd.DataFrame()
    impNumber = 50  # was 3
    maxImp = 50
    runImp = 0
    randomseed = 113
    # 99_42 143 33 113 102 0 66
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    pd.set_option('expand_frame_repr', False)
    pd.set_option("display.max_rows", False)
    df = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MiceRWarPATHData.csv", ";")
    filesIWPC = []
    if True:
        runningImp = 0
        for root, dirs, files in os.walk(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MICESTATSMODEL"):
            for file in files:
                if file.endswith('.csv') and runningImp < maxImp:
                    filesIWPC.append(file)
                    runningImp = runningImp + 1
    imp = 50
    for imp in range(impNumber):
        counter = imp + 1
        dfcurrent = df.loc[df[".imp"] == counter]
        suffix = str(counter).zfill(3)
        dfcurrent.to_csv(
            r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\ImpWarPATH_" + suffix + ".csv", ";")
    filesImp = []
    combineImputations = False
    test_size = 0.2
    if combineImputations == True and os.path.exists(
            r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\CombinedWarImputations\AllImputations" + ".csv"):
        if os.path.exists(
                r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\CombinedWarImputations\TRAINSPLIT" + ".csv"):
            trainID = pd.read_csv(
                r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\CombinedWarImputations\TRAINSPLIT" + ".csv")
            if os.path.exists(
                    r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\CombinedWarImputations\TESTSPLIT" + ".csv"):
                testID = pd.read_csv(
                    r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\CombinedWarImputations\TESTSPLIT" + ".csv")
        if False:
            if os.path.exists(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TRAINSPLIT" + ".csv"):
                if os.path.exists(
                        r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TESTSPLIT" + ".csv"):
                    trainID = pd.read_csv(
                        r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TRAINSPLIT" + ".csv",
                        ";")
                    testID = pd.read_csv(
                        r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TESTSPLIT" + ".csv",
                        ";")
                    trainDF = pd.DataFrame(trainID)
                    trainSize = len(trainDF)

    else:
        # fixedtraintest = False
        if True:
            for root, dirs, files in os.walk(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations"):
                for file in files:
                    fileup = file.upper()
                    if file.endswith(
                            '.csv') and 'TEST' not in fileup and 'TRAIN' not in fileup and "SPLIT" not in fileup and fileup not in filescheck:
                        # filesImp.append(file)
                        filedf = pd.read_csv(root + '\\' + file, ";")
                        trainID, testID = train_test_split(filedf, test_size=0.2)
                        trainSize = len(trainID)
                        if trainSize == 232:
                            print("error")
                        if trainSize == 29120:
                            print("error")

                        trainID.to_csv(
                            r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TRAINSPLIT" + ".csv",
                            ";")
                        testID.to_csv(
                            r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TESTSPLIT" + ".csv",
                            ";")
                        # fixedtraintest = True
    metric_columns = ['MAE', 'PW20', 'R2', 'Time']
    if False:
        # for imp in range(impNumber):
        patients_train = []
        patients_train = trainID[".id"].to_list()

        dftrain = df[df['.id'].isin(patients_train)]
        dftrain.to_csv(
            r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\Training\train" + suffix + ".csv", ";")
        for imp in range(impNumber):
            counter = imp + 1
            dftrainimp = dftrain.loc[df[".imp"] == counter]
            suffix = str(counter).zfill(3)
            dftrainimp.to_csv(
                r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\Training\train_" + suffix + ".csv",
                ";")
        patients_test = []
        patients_test = testID[".id"].to_list()
        dftest = df[df['.id'].isin(patients_test)]
        dftest.to_csv(
            r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\Testing\test" + suffix + ".csv", ";")
        for imp in range(impNumber):
            counter = imp + 1
            dftestimp = dftest.loc[df[".imp"] == counter]
            suffix = str(counter).zfill(3)
            dftestimp.to_csv(
                r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\Testing\test_" + suffix + ".csv",
                ";")
        counter = 0
        for root, dirs, files in os.walk(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations"):
            if root == 'C:\\Users\\Claire\\GIT_REPO_1\\CSCthesisPY\\WarImputations':
                for file in files:
                    if runImp < maxImp and file.endswith('.csv') and (
                            "train_" not in file and "test_" not in file and "SPLIT" not in file and "TRAIN" not in file and "TEST" not in file) and "ImpWarPATH" in file:
                        filedf = pd.read_csv(root + '\\' + file, ";")
                        if False:
                            if "Status" not in filedf.columns:
                                filedf["Status"] = ""
                                counter = counter + 1
                            for row in filedf.itertuples():
                                checkID = row[4]
                                rowindex = filedf.loc[filedf[".id"] == checkID].index.tolist()[
                                    0]  # OR row[0]
                                if checkID in patients_train:
                                    filedf.loc[rowindex, 'Status'] = 'train'
                                elif checkID in patients_test:
                                    filedf.loc[rowindex, 'Status'] = 'test'
                        counter = counter + 1
                        suffix = str(counter).zfill(3)
                        filedf.to_csv(
                            r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\Split\ImpWarPATHSPLIT_" + suffix + ".csv",
                            ";")
                        filesImp.append(
                            r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\Split\ImpWarPATHSPLIT_" + suffix + ".csv")
                        runImp = runImp + 1
    else:
        counter = 0
        for root, dirs, files in os.walk(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations"):
            if root == 'C:\\Users\\Claire\\GIT_REPO_1\\CSCthesisPY\\WarImputations':
                for file in files:
                    if runImp < maxImp and file.endswith('.csv') and (
                            "train_" not in file and "test_" not in file and "SPLIT" not in file and "TRAIN" not in file and "TEST" not in file) and "ImpWarPATH" in file:
                        filedf = pd.read_csv(root + '\\' + file, ";")
                        counter = counter + 1
                        suffix = str(counter).zfill(3)
                        filedf.to_csv(
                            r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\Split\ImpWarPATHSPLIT_" + suffix + ".csv",
                            ";")
                        filesImp.append(
                            r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\Split\ImpWarPATHSPLIT_" + suffix + ".csv")
                        runImp = runImp + 1
            # filesImp.append(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\CombinedWarImputations\AllImputations" + ".csv")
    results = []

    if os.path.exists(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TRAINSPLIT" + ".csv"):
        if os.path.exists(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TESTSPLIT" + ".csv"):
            trainID = pd.read_csv(
                r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TRAINSPLIT" + ".csv", ";")
            testID = pd.read_csv(
                r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TESTSPLIT" + ".csv", ";")

    root = 'C:\\Users\\Claire\\GIT_REPO_1\\CSCthesisPY\\WarImputations'
    randstring = "_" + str(randomseed) + "_"
    for file in filesImp:
        # indexTo1(file)
        # file.reset_index()
        dfnew = pd.read_csv(file, ";")
        dfmod = dfnew
        fileindex = filesImp.index(file)
        df = fileindex + 1
        suffix = str(df).zfill(3)
        rootIWPC = root.replace("WarImputations", "MICESTATSMODELHIV\\")
        IWPC_csv = rootIWPC + filesIWPC[fileindex]
        IWPCDF = pd.read_csv(IWPC_csv, ';')
        IWPCDF.reset_index()
        sampleSize = int(round(trainSize))
        dfIWPC, testset = train_test_split(IWPCDF, test_size=0.1, train_size=sampleSize, random_state=randomseed)
        #dfIWPC["Status"] = "train"
        dropColumn("IWPC", "Unnamed: 0", dfIWPC.columns, dfmod, dfIWPC)

        if combinedata == True:
            dfIWPC["Status"] = "train"
            dropColumn("IWPC", "Unnamed: 0", dfIWPC.columns, dfmod, dfIWPC)
            dfIWPC.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Claire_IWPC" + randstring + suffix + ".csv", ";")
            # dfIWPC.drop(["Unnamed: 0"], axis=1, inplace=True)
        # df = fileindex + 1
        sampleSize = int(round(trainSize))
        dfIWPC, testset = train_test_split(IWPCDF, test_size=0.1, train_size=sampleSize, random_state=randomseed)

        dfnew.reset_index()
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
        dfmod["AgeYears"] = dfmod["Age_years"]
        dfmod['AgeYears'] = np.where((dfmod['AgeYears'] <= 18), 18, dfmod['AgeYears'])
        dfmod["HIVPositive"] = np.where(dfmod["HIV_status"] == "Positive", 1, 0)
        dfmod["HIVUnknown"] = np.where(dfmod["HIV_status"] == "Unknown", 1, 0)
        dfmod["Status"] = ""
        if combinedata == True:
            dfIWPC['AgeYears'] = np.where((dfIWPC['AgeYears'] <= 18), 18, dfIWPC['AgeYears'])
            dropColumn("IWPC", "AgeDecades", dfIWPC.columns, dfmod, dfIWPC)
            dropColumn("IWPC", "INR_Three", dfIWPC.columns, dfmod, dfIWPC)
            dfIWPC["HIVPositive"] = 0
            dfIWPC["HIVUnknown"] = 0

        dropColumn("WARPATH", 'HIV_status', dfmod.columns, dfmod, dfIWPC)
        dropColumn("WARPATH", "Unnamed: 0", dfmod.columns, dfmod, dfIWPC)
        dropColumn("WARPATH", "Unnamed: 0.1", dfmod.columns, dfmod, dfIWPC)
        dropColumn("WARPATH", "Unnamed: 0.2", dfmod.columns, dfmod, dfIWPC)
        dropColumn("WARPATH", "Age_years", dfmod.columns, dfmod, dfIWPC)
        dropColumn("WARPATH", ".imp", dfmod.columns, dfmod, dfIWPC)
        dropColumn("WARPATH", ".id", dfmod.columns, dfmod, dfIWPC)
        dropColumn("WARPATH", "Unnamed: 0.1.1", dfmod.columns, dfmod, dfIWPC)

        # suffix = str(df).zfill(3)
        if combineImputations == True:
            filename = "dfWarfarin001allPatients"
            dfmod.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\PreProcessed\\" + filename + ".csv", ";")
        if False:
            dfmod = dfmod.sample(frac=1)
            dfIWPC = dfIWPC.sample(frac=1)
            frames = [dfmod, dfIWPC]
            dfmod = pd.concat(frames, ignore_index=True, sort=True)
            dfmod = dfmod.sample(frac=1)
            combfilename = "comb" + suffix
            dfmod.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\combinedata\\" + combfilename + ".csv",
                         ";")
        else:
            filename = "dfWarfarin" + suffix
            dfmod.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\PreProcessed\\" + filename + ".csv", ";")
        if True:
            print("On imputation ", df)
            dfIWPC.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Claire_IWPC_Formatted" + ".csv", ";")
            data = dfmod
            data.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Claire_Warpath" + ".csv", ";")
            # data.index = data.index + 1
            print(data.shape)
            data['Dose_mg_week'] = data['Dose_mg_week'].apply(np.sqrt)
            dfIWPC['Dose_mg_week'] = dfIWPC['Dose_mg_week'].apply(np.sqrt)
            estimates = []
            target_column = 'Dose_mg_week'
            status_column = "Status"
            test_size = 0.2
            data.reset_index()
            train, test = train_test_split(data, test_size=test_size, random_state=randomseed)  # was 66
            traindf = pd.DataFrame(train)
            traindf.reset_index()
            traindf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Claire_train" + ".csv", ";")
            # dfIWPC.reset_index()
            testdf = pd.DataFrame(test)
            if combinedata == True:
                dfIWPC.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Claire_IWPC_precombine" + ".csv", ";")
                frames = [traindf, dfIWPC]
                traindf = pd.concat(frames)
                # traindf = pd.concat(frames, ignore_index=True, sort=True)
                traindf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Claire_trainplusIWPC" + ".csv", ";")
                # combfilename = "comb" + suffix
                # combfilename.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\combinedata\\" + combfilename + ".csv",";")
            traindf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Train" + suffix + ".csv", ";")
            testdf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Test" + suffix + ".csv", ";")
            testdf['Status'] = 'test'
            traindf['Status'] = 'train'
            frames = (traindf, testdf)
            combdf = pd.concat(frames)
            # combdf.index = combdf.index + 1
            combdf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\TrainPlusTest" + suffix + ".csv", ";")
            combID = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\TrainPlusTest" + suffix + ".csv",
                                 ";")
            # combID.index = combID.index+1
            combID.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\TrainTestStatus" + suffix + ".csv", ";")
            combID['NewStatus'] = combID['Status']
            # combID['NewStatus'] = 'train'
            # combID['NewStatus'] = combID.apply(lambda x:TrainOrTest(x["Unnamed: 0"],trainID[".id"].tolist(), testID[".id"].tolist()), axis=1)
            combID.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\TrainTestStatus" + suffix + ".csv", ";")
            squaring = True
            combIDcopy = combID
            train = combID[combID.NewStatus == "train"]
            test = combID[combID.NewStatus == "test"]
            train = train.drop([status_column], axis=1)
            test = test.drop([status_column], axis=1)
            train = train.drop(['NewStatus'], axis=1)
            test = test.drop(['NewStatus'], axis=1)
            x_cols = list(train.columns)
            # _cols_notarg = x_cols.remove(target_column)
            targ_col = list(target_column)
            # train = scaler.fit_transform(train)
            # test = scaler.transform(test)
            # train = pd.DataFrame(train, columns = x_cols)
            # test = pd.DataFrame(test, columns = x_cols)
            targetindex = x_cols.index(target_column)
            y_train = train[target_column].values
            x_train = train.drop([target_column], axis=1)
            y_test = test[target_column].values
            x_test = test.drop([target_column], axis=1)

        estimates = []
        LR = LinearRegression()
        estimates.append(Estimator(LR, 'LR'))

        MLPR_Scaling = False
        if MLPR_Scaling == True:
            sc_X = StandardScaler()
            x_traincopy = x_train
            x_testcopy = x_test
            x_train = sc_X.fit_transform(x_train)
            x_test = sc_X.transform(x_test)
        if True:
            # MLPR1 = MLPRegressor(hidden_layer_sizes=(90,5,), activation="relu", learning_rate='adaptive', max_iter=1000,learning_rate_init=0.001)
            MLPR2 = MLPRegressor(hidden_layer_sizes=(196,), learning_rate='adaptive',
                                 learning_rate_init=0.002,
                                 max_iter=2000, activation="relu")

            if True:
                # GBR = GradientBoostingRegressor()
                # estimates.append(Estimator(GBR,'GBR'))
                # XGBR = XGBRegressor()
                # estimates.append(Estimator(XGBR,'XGBR'))
                # RR=Ridge()[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
                # RR = Ridge()
                # LAS = Lasso()
                # ELNET = ElasticNet()
                # estimates.append(Estimator(LAS,'LASSO'))
                # estimates.append(Estimator(ELNET,'ELNET'))
                # estimates.append(Estimator(RR,'RIDGE'))
                # KNNR = KNeighborsRegressor()
                # estimates.append(Estimator(KNNR, 'KNN'))
                #svr = sklearn.svm.SVR()
                #estimates.append(Estimator(svr, 'SVREG'))
                MLPR = MLPRegressor()
                estimates.append(Estimator(MLPR,'MLPR'))
                # RF = RandomForestRegressor()
                # estimates.append(Estimator(RF, 'RF'))
                # DTR = DecisionTreeRegressor()
                # estimates.append(Estimator(DTR,'DTR'))
                # LGB = lgb.LGBMRegressor()
                # estimates.append(Estimator(LGB,'LGB'))
                if "Unnamed: 0" in x_train.columns:
                    x_train.drop(["Unnamed: 0"], axis=1, inplace=True)
                if "Unnamed: 0" in x_test.columns:
                    x_test.drop(["Unnamed: 0"], axis=1, inplace=True)
                if "Unnamed: 0.1" in x_train.columns:
                    x_train.drop(["Unnamed: 0.1"], axis=1, inplace=True)
                if "Unnamed: 0.1" in x_test.columns:
                    x_test.drop(["Unnamed: 0.1"], axis=1, inplace=True)
                for _, est in enumerate(estimates):
                    resultsdict = traineval(est, x_train, y_train, x_test, y_test, squaring=squaring, df=df,randomseed=randomseed)
                    res_dict = {
                        'Estimator': [est.identifier for x in range(len(resultsdict['PW20']))],
                        'PW20': resultsdict['PW20'],
                        'MAE': resultsdict['MAE'],
                        'R2': resultsdict['R2'],
                        'Time': resultsdict['Time']}
                    if resultsdict['MAE'] > [5]:
                        results.append(res_dict)

                df_res = pd.DataFrame()
                for res in results:
                    df_res = df_res.append(pd.DataFrame.from_dict(res))
                print(f"\n\n{df_res.groupby(['Estimator']).agg(np.mean)}\n")

    # dfResults = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfResults" + ".csv", ";")

    dfResults = pd.DataFrame(results)
    dfResults["PW20"] = dfResults.apply(lambda x: ExitSquareBracket(x["PW20"]), axis=1).astype(float)
    dfResults["MAE"] = dfResults.apply(lambda x: ExitSquareBracket(x["MAE"]), axis=1).astype(float)
    dfResults["R2"] = dfResults.apply(lambda x: ExitSquareBracket(x["R2"]), axis=1).astype(float)
    dfResults["Time"] = dfResults.apply(lambda x: ExitSquareBracket(x["Time"]), axis=1).astype(float)
    dfResults["Estimator"] = dfResults.apply(lambda x: ExitSquareBracket(x["Estimator"]), axis=1)
    dfResults.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfResultsPRE" + ".csv", ";")
    if "Unnamed: 0" in dfResults.columns:
        dfResults.drop(["Unnamed: 0"], axis=1, inplace=True)
    MLPR_rows = dfResults.loc[dfResults['Estimator'] == 'MLPR']
    if len(MLPR_rows) > 0:
        MAElist = MLPR_rows['MAE'].tolist()
        maxlimit = iqr(MAElist) + 1.5 * scoreatpercentile(MAElist, 75)
        newMAElist = [x for x in MAElist if x <= maxlimit]
        newMeanMae = np.mean(newMAElist)
        for i in dfResults.index:
            if dfResults.loc[i, 'MAE'] > maxlimit:
                dfResults.at[i, 'MAE'] = np.nan
                dfResults.at[i, 'PW20'] = np.nan
                dfResults.at[i, 'R2'] = np.nan
                dfResults.at[i, 'Time'] = np.nan

    dfSummary = dfResults.groupby('Estimator').apply(np.mean)
    dfSummary.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfSummary_" + str(randomseed) + ".csv", ";")

    stddev = []
    confinterval = []
    for i in range(len(metric_columns)):
        for _, est in enumerate(estimates):
            current_estimator = est.identifier
            current_metric = metric_columns[i]
            current_mean = dfSummary.loc[current_estimator][current_metric]
            # metric_values = dfResults.apply(lambda x:collect_Results(x['Estimator'],current_metric,axis=1))
            metric_values = np.where(dfResults['Estimator'] == current_estimator, dfResults[current_metric],
                                     9999)
            metriclist = np.array(metric_values)
            # metriclist = [j for j in metriclist if j != np.nan]
            metriclistcopy = []
            for j in metriclist:
                if j != 9999:
                    if math.isnan(j) == False:
                        metriclistcopy.append(j)
            metriclist = metriclistcopy
            current_stddev = confintlimit95(metriclist, current_mean)
            confinterval.append(
                {'estimator': current_estimator, 'metric': current_metric, 'mean': current_mean,
                 '95% CI lower bound': current_mean - current_stddev,
                 '95% CI upper bound': current_mean + current_stddev})
        # stddev.append({'metric':current_metric,'standarddev':current_stdde  v,'mean':current_mean})
    dfResults.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfResults" + ".csv", ";")

    dfConfidence = pd.DataFrame(confinterval,
                                columns=['estimator', 'metric', 'mean', '95% CI lower bound', '95% CI upper bound'])
    dfConfidence.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfConfidence" + ".csv", ";")
    print("STOP HERE")

if __name__ == "__main__":
    main()


































































































































































