import sklearn
import time
import csv
import os
import os.path
import pandas as pd
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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import xgboost
from xgboost import XGBClassifier  # for extreme gradient boosting model
from xgboost import XGBRFRegressor
from xgboost import XGBRegressor
from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from hyperopt.fmin import space_eval
from hyperopt.pyll.base import scope
from hpsklearn import HyperoptEstimator
from hpsklearn import any_classifier
from hpsklearn import any_preprocessing
from hpsklearn import any_regressor, random_forest_regressor, xgboost_regression, k_neighbors_regressor, linear_regression,gradient_boosting_regressor
from hpsklearn import ada_boost_regressor, decision_tree_regressor, svr, mlp_regressor, lasso, ridge, elastic_net

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


def confintlimit95(metric):
    return 1.96 * np.sqrt(variance(metric) / len(metric))

def TrainOrTest(patientID,TrainList, TestList):
    TrainDF = pd.DataFrame(TrainList.sort())
    TestDF = pd.DataFrame(TestList.sort())
    if (patientID in TrainList):
        return 'train'
    elif (patientID in TestList):
        return 'test'


    #newList = []
    #for patient in patientIDList:
    #  currpatient = patient
    #  for fixedpatient in TrainList:
    #    if (fixedpatient == currpatient):
    #      newList.append('train')
    #  for fixedpatient in TestList:
    #    if (fixedpatient == currpatient):
    #      newList.append('test')
    #return newList

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
        mae = mean_absolute_error(y_test, yhat)
        # store the performance
        scores.append(-mae)
    # report model performance
    return scores


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


def confintlimit95(metric):
    return 1.96 * np.sqrt(variance(metric) / len(metric))

def TrainOrTest(patientID,TrainList, TestList):
    TrainDF = pd.DataFrame(TrainList.sort())
    TestDF = pd.DataFrame(TestList.sort())
    if (patientID in TrainList):
        return 'train'
    elif (patientID in TestList):
        return 'test'


    #newList = []
    #for patient in patientIDList:
    #  currpatient = patient
    #  for fixedpatient in TrainList:
    #    if (fixedpatient == currpatient):
    #      newList.append('train')
    #  for fixedpatient in TestList:
    #    if (fixedpatient == currpatient):
    #      newList.append('test')
    #return newList

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
        mae = mean_absolute_error(y_test, yhat)
        # store the performance
        scores.append(-mae)
    # report model performance
    return scores

def traineval(est, estimates, xtrain, ytrain, xtest, ytest, squaring, df):
    suffix = str(df).zfill(3)
    resultsdict = {'PW20': 0, 'MAE': 0, 'R2': 0, 'Time':''}
    ml_learner = est
    print(f'\n{ml_learner}...')
    modelID = ml_learner
    model = estimates[est]
    xtraindf = pd.DataFrame(xtrain)
    #xtrain = np.any(np.isnan(xtrain))
    #xtrain = np.all(np.isfinite(xtrain))
    #xtraindf = pd.DataFrame(xtrain)
    #ytrain = np.any(np.isnan(ytrain))
    #ytrain = np.all(np.isfinite(ytrain))
    ytraindf = pd.DataFrame(ytrain)
    #xtest = np.any(np.isnan(xtest))
    xtestdf = pd.DataFrame(xtest)
    #ytest = np.any(np.isnan(ytest))
    #ytest = np.all(np.isfinite(ytest))
    ytestdf = pd.DataFrame(ytest)
    xtraindf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\HYPERBASEDATA\xtrain_" + ml_learner + suffix + ".csv", ";")
    xtestdf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\HYPERBASEDATA\xtest_" + ml_learner + suffix + ".csv", ";")
    ytraindf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\HYPERBASEDATA\ytrain_" + ml_learner + suffix + ".csv", ";")
    ytestdf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\HYPERBASEDATA\ytest_" + ml_learner + suffix + ".csv", ";")
    if est=="MLPR":
        maxEvals = 50
        #[8, 16, 32, (8, 8), (16, 16)]
        space = {'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(10,),(40,),(85,),(30,),(85,3),(15,3,),(90,5,)]),
                 'learning_rate': hp.choice('learning_rate',['adaptive']),
                 'activation': hp.choice('activation', ['relu']),
                 'learning_rate_init':hp.choice('learning_rate_init',[0.001, 0.0015,0.002]),
                 'max_iter': hp.choice('max_iter', [1000,  1500, 2000, 2500,  3000, 3500, 4000])
                 }

        def hyperparameter_tuning(space):
            scaler = MinMaxScaler()
            model = MLPRegressor(hidden_layer_sizes=space['hidden_layer_sizes'], learning_rate=space['learning_rate'],
                                 max_iter=int(space['max_iter']), activation=space['activation'],
                                 learning_rate_init=float(space['learning_rate_init']))
            #pipeline = make_pipeline(scaler,model)
            evaluation = [(xtrain, ytrain), (xtest, ytest)]
            model.fit(xtrain, ytrain)
            pred = model.predict(xtest)
            ytest2 = np.square(ytest)
            pred2 = np.square(pred)
            R2 = r2_score(ytest2, pred2)
            print("SCORE:", R2)
            returndict = {'loss': 1 - R2, 'status': STATUS_OK, 'model': model}
            return returndict

        start = time.time()
        trials = Trials()
        best = fmin(fn=hyperparameter_tuning, space=space, algo=tpe.suggest,max_evals=maxEvals, trials=trials)
        end = time.time()
        timeElapsed = end - start
        modelcurrent = trials.results[-1]['model']
        fitted=modelcurrent.fit(xtrain,ytrain)
        ypred = fitted.predict(xtest)
        if squaring:
            ypred2 = np.square(ypred)
            ytest2 = np.square(ytest)
        paramdict = modelcurrent.get_params()
    else:
        modelcurrent=HyperoptEstimator(regressor=model, verbose=True)
        start = time.time()
        modelcurrent.fit(xtrain, ytrain)
        end = time.time()
        timeElapsed = end - start
        best = modelcurrent.best_model()
        fitted = best['learner'].fit(xtrain, ytrain)
        ypred = fitted.predict(xtest)
        if squaring:
            ypred2 = np.square(ypred)
            ytest2 = np.square(ytest)
        paramdict = best['learner'].get_params()
    paramdict = {k: [v] for k, v in paramdict.items()}
    dfHyperCurrent = pd.DataFrame(paramdict)
    dfHyperCurrent['imputation'] = df
    dfHyperCurrent['mae'] = mean_absolute_error(ytest2, ypred2)
    ml_learner = modelID
    suffix = str(df).zfill(3)
    if df == 1:
      dfHyper = pd.DataFrame()
    else:
      dfpre = df - 1
      suffixpre = str(dfpre).zfill(3)
      dfHyper = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\HYPEROPTHYPERPARAMETERS\model_" + ml_learner + suffixpre + ".csv",";")
    frames = (dfHyper, dfHyperCurrent)
    dfHyper = pd.concat(frames)
    dfHyper["model"] = ml_learner
    if "Unnamed: 0" in dfHyper.columns:
        dfHyper.drop(["Unnamed: 0"], axis=1, inplace=True)
    if "Unnamed: 0.1" in dfHyper.columns:
        dfHyper.drop(["Unnamed: 0.1"], axis=1, inplace=True)
    dfHyper.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\HYPEROPTHYPERPARAMETERS\model_" + ml_learner + suffix + ".csv", ";")
    MAE = mean_absolute_error(ytest2, ypred2)
    dfHyper = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\HYPEROPTHYPERPARAMETERS\model_" + ml_learner + suffix + ".csv", ";")
    dfHyper.loc[df - 1, "mae"] = MAE
    dfHyper.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\HYPEROPTHYPERPARAMETERS\model_" + ml_learner + suffix + ".csv", ";")
    R2 = RSquared(ytest2, ypred2)
    PW20 = PercIn20(ytest2, ypred2)
    resultsdict['PW20'] = [PW20]
    resultsdict['MAE'] = [MAE]
    resultsdict['R2'] = [R2]
    resultsdict['Time'] = [timeElapsed]
    return resultsdict

def main():
    # dfHyper = pd.DataFrame()
    combinedata = False
    scaler = MinMaxScaler()
    dftemplate = pd.DataFrame()
    dfWarPath = pd.DataFrame()
    impNumber = 50  # was 3
    maxImp = 50  # was 3
    runImp = 0
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    pd.set_option('expand_frame_repr', False)
    pd.set_option("display.max_rows", False)
    df = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MiceRWarPATHData.csv", ";")
    filesIWPC = []
    if True:
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
    combineImputations = False
    test_size = 0.2
    if combineImputations == True and os.path.exists(
            r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\CombinedWarImputations\AllImputations" + ".csv"):
        if os.path.exists(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\CombinedWarImputations\TRAINSPLIT" + ".csv"):
            trainID = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\CombinedWarImputations\TRAINSPLIT" + ".csv")
            if os.path.exists(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\CombinedWarImputations\TESTSPLIT" + ".csv"):
                testID = pd.read_csv(
                    r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\CombinedWarImputations\TESTSPLIT" + ".csv")
        if False:
            if os.path.exists(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TRAINSPLIT" + ".csv"):
                if os.path.exists(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TESTSPLIT" + ".csv"):
                    trainID = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TRAINSPLIT" + ".csv",
                                          ";")
                    testID = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TESTSPLIT" + ".csv",
                                         ";")
                    trainDF = pd.DataFrame(trainID)
                    trainSize = len(trainDF)

    else:
        # fixedtraintest = False
        if False:
            for root, dirs, files in os.walk(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations"):
                for file in files:
                    if file.endswith('.csv') and 'TEST' not in file and 'TRAIN' not in file and "SPLIT" not in file:
                        # filesImp.append(file)
                        filedf = pd.read_csv(root + '\\' + file, ";")
                        trainID, testID = train_test_split(filedf, test_size=0.2)
                        trainSize = len(trainID)
                        trainID.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TRAINSPLIT" + ".csv",
                                       ";")
                        testID.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TESTSPLIT" + ".csv",
                                      ";")
                        # fixedtraintest = True
    metric_columns = ['MAE', 'PW20', 'R2', 'Time']
    if False:
        # for imp in range(impNumber):
        patients_train = []
        patients_train = trainID[".id"].to_list()

        dftrain = df[df['.id'].isin(patients_train)]
        dftrain.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\Training\train" + suffix + ".csv", ";")
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
        dftest.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\Testing\test" + suffix + ".csv", ";")
        for imp in range(impNumber):
            counter = imp + 1
            dftestimp = dftest.loc[df[".imp"] == counter]
            suffix = str(counter).zfill(3)
            dftestimp.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\Testing\test_" + suffix + ".csv",
                             ";")
        counter = 0
        for root, dirs, files in os.walk(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations"):
            if root == 'C:\\Users\\Claire\\GIT_REPO_1\\CSCthesisPY\\WarImputations':
                for file in files:
                    if Imp < maxImp and file.endswith('.csv') and (
                            "train_" not in file and "test_" not in file and "SPLIT" not in file and "TRAIN" not in file and "TEST" not in file) and "ImpWarPATH" in file:
                        filedf = pd.read_csv(root + '\\' + file, ";")
                        if False:
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
            trainID = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TRAINSPLIT" + ".csv", ";")
            testID = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\TESTSPLIT" + ".csv", ";")

    root = 'C:\\Users\\Claire\\GIT_REPO_1\\CSCthesisPY\\WarImputations'
    for file in filesImp:
        dfnew = pd.read_csv(file, ";")
        fileindex = filesImp.index(file)
        if combinedata == True:
            rootIWPC = root.replace("WarImputations", "MICESTATSMODELHIV\\")
            IWPC_csv = rootIWPC + filesIWPC[fileindex]
            IWPCDF = pd.read_csv(IWPC_csv, ';')
            sampleSize = int(round(trainSize * 0.5))
            dfIWPC = IWPCDF.sample(n=sampleSize)
            dfIWPC["Status"] = "train"
            dfIWPC.drop(["Unnamed: 0"], axis=1, inplace=True)
        df = fileindex + 1
        dfmod = dfnew
        # gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg                                                                                                                                                                                               llllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkklllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        b v         vbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb nbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
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
        dfmod.drop(["Unnamed: 0"], axis=1, inplace=True)
        dfmod.drop(["Unnamed: 0.1"], axis=1, inplace=True)
        dfmod.drop(["Age_years"], axis=1, inplace=True)
        dfmod.drop([".imp"], axis=1, inplace=True)
        dfmod.drop([".id"], axis=1, inplace=True)
        dfmod.drop(["Unnamed: 0.1.1"], axis=1, inplace=True)
        suffix = str(df).zfill(3)
        if combineImputations == True:
            filename = "dfWarfarin001allPatients"
            dfmod.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\PreProcessed\\" + filename + ".csv", ";")
        elif combinedata == True:
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
        if True:
            print("On imputation ", df)
            data = dfmod
            data.index = data.index + 1
            print(data.shape)
            data['Dose_mg_week'] = data['Dose_mg_week'].apply(np.sqrt)
            estimates = []
            target_column = 'Dose_mg_week'
            status_column = "Status"
            # unnamed_column = "Unnamed: 0.1.1"
            # train = data.loc[data["Status"] == "train"]
            # test = data.loc[data["Status"] == "test"]
            test_size = 0.2
            train, test = train_test_split(data, test_size=test_size, random_state=66)
            traindf = pd.DataFrame(train)
            testdf = pd.DataFrame(test)

            traindf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Train" + suffix + ".csv", ";")
            testdf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Test" + suffix + ".csv", ";")
            testdf['Status'] = 'test'
            traindf['Status'] = 'train'
            frames = (traindf, testdf)
            combdf = pd.concat(frames)
            # combdf.index = combdf.index + 1
            combdf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\TrainPlusTest" + suffix + ".csv", ";")
            combID = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\TrainPlusTest" + suffix + ".csv", ";")
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

            # LAS = Lasso(alpha=0.009)
            # LAS2 = Lasso()
            # RF = RandomForestRegressor(max_depth=80, max_features='sqrt', min_samples_leaf=5,
            #                           min_samples_split=12, n_estimators=2000)
            # ADB = AdaBoostRegressor(RF, n_estimators=6, random_state=42)
            # estimates.append(Estimator(RF, 'RF'))
            # estimates.append(Estimator(ADB, 'AdaBoostRF'))
            LR = LinearRegression()
            estimates.append(Estimator(LR, 'LR'))
            # estimates.append(Estimator(LAS2, 'LAS2'))
            # XGB
            # modelX2 = XGBRegressor()
            # modelX = XGBRegressor(booster='gblinear',
            #                     max_depth=1,
            #                     min_child_weight=13,
            #                    subsample=0.8,
            #                     colsample_bylevel=0.6,
            #                     colsample_bynode=0.9,
            #                     n_estimators=50,
            #                     learning_rate= 1,
            #                     )

            # RIDGE
            # grid = dict()
            # ridge_alphas = np.linspace(0, 0.02, 11)
            # ridge_solvers = []
            # ridge_solvers.append('svd')
            # ridge_solvers.append('cholesky')
            # ridge_solvers.append('lsqr')
            # ridge_solvers.append('sag')
            # RR2 = Ridge()
            # RR = Ridge(alpha=0.02, solver="lsqr")
            # estimates.append(Estimator(RR2, 'RR2'))
            # estimates.append(Estimator(RR, 'RR'))
            # estimates.append(Estimator(modelX2, 'XGBR2'))
            # estimates.append(Estimator(modelX,'XGBR'))
            # LASSO
            # lasso_alphas = np.linspace(0, 0.02, 11)
            # LAS = Lasso(alpha=0.002)
            # estimates.append(Estimator(LAS, 'LAS'))

            # EL
            # EL2 = ElasticNet()
            # EL = ElasticNet(alpha=0.01, l1_ratio=0.01)
            # ABEL = AdaBoostRegressor(EL)
            # ratios = np.arange(0, 1, 0.01)
            # alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
            # estimates.append(Estimator(EL2, 'EL2'))
            # estimates.append(Estimator(EL, 'EL'))
            # estimates.append(Estimator(ABEL,'ABEL'))

            # KNN2 = KNeighborsRegressor()
            # KNN = KNeighborsRegressor(n_neighbors=15,p=2,weights="uniform")
            # estimates.append(Estimator(KNN, 'KNN'))
            # estimates.append(Estimator(KNN2, 'KNN2'))

            # n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=10)]
            # max_features = ['auto', 'sqrt']
            # max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
            # min_samples_split = [2, 5, 10]
            # min_samples_leaf = [1, 2, 4]
            # bootstrap = [True, False]
            # random_grid = {'n_estimators': n_estimators}

            # grid = {'RR': {'alpha': ridge_alphas, 'solver': ridge_solvers},
            #        'RF': {'n_estimators':1000},
            #        'LAS': {'alpha': lasso_alphas},
            #        'EL': {'alpha': alphas, 'l1_ratio': ratios},
            #        'LR': {},
            #        'KNN': param_grid
            #        }
            # NN2 = MLPRegressor()
            # NN = MLPRegressor(hidden_layer_sizes=(100,), activation="relu", random_state=1, max_iter=2000)
            # estimates.append(Estimator(NN, 'NN'))
            # estimates.append(Estimator(NN, 'NN2'))
        estimates = []
        estimates.append(Estimator(LR, 'LR'))
        # XGBR = XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.01, colsample_bytree=0.5, subsample=0.9)
        # estimates.append(Estimator(LR, 'LR'))
        # estimates.append(Estimator(pipeline_scaled, 'MLPR1'))
        # estimates.append(Estimator(XGBR,'XGBR'))

        # MLPR = MLPRegressor(alpha=0.05, hidden_layer_sizes=(100, 50, 30), learning_rate='adaptive', max_iter=100,
        #                    solver='sgd')
        MLPR_Scaling = False
        if MLPR_Scaling == True:
            sc_X = StandardScaler()
            x_traincopy = x_train
            x_testcopy = x_test
            x_train = sc_X.fit_transform(x_train)
            x_test = sc_X.transform(x_test)
        if True:
            # MLPR1 = MLPRegressor(hidden_layer_sizes=(90,5,), activation="relu", learning_rate='adaptive', max_iter=1000,learning_rate_init=0.001)
            MLPR2 = MLPRegressor(hidden_layer_sizes=(196,), learning_rate='adaptive', learning_rate_init=0.002,
                                 max_iter=2000, activation="relu")
            # MLPR2A = MLPRegressor(hidden_layer_sizes=(15, 3,), learning_rate='adaptive', learning_rate_init=0.0039, max_iter=1500,activation="relu")
            # MLPR2B = MLPRegressor(hidden_layer_sizes=(15, 3,), learning_rate='adaptive', learning_rate_init=0.0034,max_iter=2600,activation="relu")
            # MLPR2C = MLPRegressor(hidden_layer_sizes=(15, 3,), learning_rate='adaptive', learning_rate_init=0.0031,max_iter=3100,activation="relu")
            # MLPR2D = MLPRegressor(hidden_layer_sizes=(15, 3,), learning_rate='adaptive', learning_rate_init=0.0026,max_iter=2800,activation="relu")
            # MLPR3 = MLPRegressor(hidden_layer_sizes=(85,3,), activation="relu", learning_rate='adaptive', max_iter=2000, learning_rate_init=0.003)
            # MLPR4 = MLPRegressor(hidden_layer_sizes=(30,), activation="relu", learning_rate='adaptive',max_iter=3000, learning_rate_init=0.0022)
            # MLPR4B = MLPRegressor(hidden_layer_sizes=(30,), activation="relu", learning_rate='adaptive',max_iter=3400, learning_rate_init=0.0031)
            # MLPR4C = MLPRegressor(hidden_layer_sizes=(30,), activation="relu", learning_rate='adaptive',max_iter=3000, learning_rate_init=0.005)
            # MLPR4D = MLPRegressor(hidden_layer_sizes=(30,), activation="relu", learning_rate='adaptive',max_iter=1800, learning_rate_init=0.0029)

            # MLPR5 = MLPRegressor(hidden_layer_sizes=(40,), learning_rate='adaptive', learning_rate_init=0.003,max_iter=2500, activation="relu")
            # MLPR6 = MLPRegressor(hidden_layer_sizes=(65,), activation="relu", learning_rate='adaptive', max_iter=2500, learning_rate_init=0.003)

            # pipeline1_scaled = Pipeline([('scale', MinMaxScaler()), ('alg', MLPR1)])
            # pipeline2_scaled = Pipeline([('scale', MinMaxScaler()), ('alg', MLPR2)])
            # pipeline2A_scaled = Pipeline([('scale', MinMaxScaler()), ('alg', MLPR2A)])
            # pipeline2B_scaled = Pipeline([('scale', MinMaxScaler()), ('alg', MLPR2B)])
            # pipeline2C_scaled = Pipeline([('scale', MinMaxScaler()), ('alg', MLPR2C)])
            # pipeline2D_scaled = Pipeline([('scale', MinMaxScaler()), ('alg', MLPR2D)])
            # pipeline3_scaled = Pipeline([('scale', MinMaxScaler()), ('alg', MLPR3)])
            # pipeline4_scaled = Pipeline([('scale', MinMaxScaler()), ('alg', MLPR4)])
            # pipeline4A_scaled = Pipeline([('scale', MinMaxScaler()), ('alg', MLPR4A)])
            # pipeline4B_scaled = Pipeline([('scale', MinMaxScaler()), ('alg', MLPR4B)])
            # pipeline4C_scaled = Pipeline([('scale', MinMaxScaler()), ('alg', MLPR4C)])
            # pipeline4D_scaled = Pipeline([('scale', MinMaxScaler()), ('alg', MLPR4D)])
            # pipeline5_scaled = Pipeline([('scale', MinMaxScaler()), ('alg', MLPR5)])
            # pipeline6_scaled = Pipeline([('scale', MinMaxScaler()), ('alg', MLPR6)])
            # estimates.append(Estimator(pipeline1_scaled,'MLPR1'))
            # estimates.append(Estimator(pipeline3_scaled,'MLPR3'))
            # estimates.append(Estimator(pipeline4_scaled,'MLPR4'))
            # estimates.append(Estimator(pipeline4A_scaled, 'MLPR4A'))
            # estimates.append(Estimator(pipeline4B_scaled, 'MLPR4B'))
            # estimates.append(Estimator(pipeline4C_scaled, 'MLPR4C'))
            # estimates.append(Estimator(pipeline4D_scaled, 'MLPR4D'))
            # estimates.append(Estimator(pipeline2_scaled, 'MLPR2'))
            # estimates.append(Estimator(pipeline2A_scaled, 'MLPR2A'))
            # estimates.append(Estimator(pipeline2B_scaled, 'MLPR2B'))
            # estimates.append(Estimator(pipeline2C_scaled, 'MLPR2C'))
            # estimates.append(Estimator(pipeline2D_scaled, 'MLPR2D'))
            # estimates.append(Estimator(pipeline5_scaled,'MLPR5'))
            # estimates.append(Estimator(pipeline6_scaled,'MLPR6'))

            if True:
                # RF = RandomForestRegressor()
                KNNR = KNeighborsRegressor()
                pipeline_KNNR_scaled = Pipeline([('scale', MinMaxScaler()), ('alg', KNNR)])
                # RF = RandomForestRegressor(max_depth=100, max_features=2, min_samples_leaf=60,min_samples_split=8, n_estimators=100)
                # RF = RandomForestRegressor(max_depth=125, max_features=2, min_samples_leaf=3,min_samples_split=8, n_estimators=200)
                # RF = RandomForestRegressor(max_depth=120, max_features=3, min_samples_leaf=4,min_samples_split=12, n_estimators=100)
                # RF = RandomForestRegressor(){'bootstrap': True, 'max_depth': 120, 'max_features': 3, 'min_samples_leaf': 4, 'min_samples_split': 12,
                # 'n_estimators': 100}
                # ................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................,,,,,,,,,,,
                # ABRF = AdaBoostRegressor(base_estimator=RF,  learning_rate=0.015, n_estimators=15)
                # ABRF2 = AdaBoostRegressor(base_estimator=RF, learning_rate=0.01, n_estimators=10)
                # RF = RandomForestRegressor()
                # estimates.append(Estimator(RF,'RF'))
                # estimates.append(Estimator(pipeline_KNNR_scaled, 'KNN'))
                # estimates.append(Estimator(KNNR, 'KNN'))
                # estimates.append(Estimator(ABRF,'ABRF'))
                # estimates.append(Estimator(ABRF2, 'ABRF2'))
                # model = Lasso()
                # pipeline_LASSO_scaled = Pipeline([('scale', MinMaxScaler()), ('alg', model)])
                # estimates.append(Estimator(pipeline_LASSO_scaled, 'LASSO'))
                # model = Ridge()
                # pipeline_Ridge_scaled = Pipeline([('scale', MinMaxScaler()), ('alg', model)])
                # estimates.append(Estimator(pipeline_Ridge_scaled, 'RIDGE'))
                # model = ElasticNet()
                # pipeline_ELNET_scaled = Pipeline([('scale', MinMaxScaler()), ('alg', model)])
                # estimates.append(Estimator(pipeline_ELNET_scaled, 'ELNET'))
                # model = sklearn.svm.SVR()
                # pipeline_SVREG_scaled = Pipeline([('scale', MinMaxScaler()), ('alg', model)])
                # estimates.append(Estimator(pipeline_SVREG_scaled,"SVREG"))
                suffix = str(df).zfill(3)
                ml_weak_learner = 'RF'
                dfHyper = pd.read_csv(
                    r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\HYPERPARAMETERS\model_" + ml_weak_learner + suffix + ".csv",
                    ";")
                listCols = dfHyper.columns.to_list()
                for i in listCols:
                    if "Unnamed:" in i:
                        dfHyper.drop([i], axis=1, inplace=True)
                dfHyper["MAE"] = -dfHyper["score"]
                dfSort = dfHyper.sort_values(by='MAE', ascending=True)
                dfSort = dfSort.reset_index(drop=True)
                param_max_depth = dfSort['max_depth'][0]
                param_max_features = dfSort['max_features'][0]
                param_min_impurity_decrease = dfSort['min_impurity_decrease'][0]
                param_min_samples_leaf = dfSort['min_samples_leaf'][0]
                param_n_estimators = dfSort['n_estimators'][0]
                RF = RandomForestRegressor(n_estimators=param_n_estimators, max_depth=param_max_depth,
                                           max_features=param_max_features,
                                           min_samples_leaf=param_min_samples_leaf,
                                           min_impurity_decrease=param_min_impurity_decrease)

                ABRF = AdaBoostRegressor(base_estimator=RF)
                # estimates.append(Estimator(RF, 'RF'))
                # estimates.append(Estimator(RF, 'RF2'))
                # DTR = DecisionTreeRegressor()
                # estimates.append(Estimator(DTR,'DTR'))
                # estimates.append(Estimator(ABRF,'ABRF'))
                DTR = DecisionTreeRegressor()
                estimates.append(Estimator(DTR, 'DTR'))

                if False:
                    minmae = 99
                    learning_rate_values = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
                    estimator_values = [30, 300, 3000]
                    depths = [None, 1, 2, 3, 4]
                    samples = [1, 0.9, 0.8, 0.7, 0.6, 0.5]
                    for value in learning_rate_values:
                        for estimator in estimator_values:
                            for depth in depths:
                                for sample in samples:
                                    gbr = GradientBoostingRegressor(max_depth=depth, subsample=sample,
                                                                    n_estimators=estimator, random_state=66,
                                                                    learning_rate=value)
                                    gbr_result = gbr.fit(x_train, y_train)
                                    y_pred = gbr.predict(x_test)
                                    mae = mean_absolute_error(y_test, y_pred)
                                    if mae < minmae:
                                        minmae = mae
                                    print('Estimator:', estimator, 'Depth:', depth, 'Sample:', sample, 'LearningRate:',
                                          value, ', Score:', mae)
                    print('smallest mae is ', minmae)
                LR = linear_regression("mylinear")
                GBR = gradient_boosting_regressor("mygbr")
                XGB = xgboost_regression("myxgb")
                KNN = k_neighbors_regressor("myknn")
                DTR = decision_tree_regressor("mydtr")
                SVR = sklearn.svm.SVR()
                RIDGE = ridge("myridge")
                LASSO = lasso("mylasso")
                ELNET = elastic_net("myelasticnet")
                RF = random_forest_regressor("myrf")
                MLPR = mlp_regressor("mymlpr")

                # estimates = {'LR':LR, 'GBR':GBR, 'XGB':XGB, 'KNN':KNN, 'DTR':DTR,'RF':RF}
                # stimates = {'LR':LR, 'RF':RF}
                # estlist = ['LR','RF']
                estlist = ['LR','SVREG']
                estimates = {'LR':LR, 'SVREG':SVR}
                # estimates = {'LR':LR,'Lasso':LASSO, 'Ridge':RIDGE, 'ELNet':ELNET}
                # estlist = {'LR', 'Lasso', 'Ridge', 'ELNet'}
                #estimates = {'LR':LR, 'MLPR': MLPR}
                #estlist = {'LR','MLPR'}
                #estimates = {'LR':LR, 'GBR':GBR}
                #estlist = {'LR','GBR'}
                #estimates = {'LR':LR, 'DTR':DTR}
                #estlist = {'LR','DTR'}
                if False:
                    estimates.append(Estimator(LR, 'LR'))
                    estimates.append(Estimator(GBR, 'GBR'))
                    estimates.append(Estimator(XGB, 'XGB'))
                    estimates, append(Estimator(KNN, 'KNN'))
                    estimates.append(Estimator(DTR, 'DTR'))
                    estimates.append(Estimator(RF, 'RF'))
                if "Unnamed: 0.1" in x_train.columns:
                    x_train.drop(["Unnamed: 0.1"], axis=1, inplace=True)
                if "Unnamed: 0.1" in x_test.columns:
                    x_test.drop(["Unnamed: 0.1"], axis=1, inplace=True)

                for est in estlist:
                    # resultsdict = traineval(est, estimates, x_train, y_train, x_test, y_test, squaring=squaring, df=df)
                    resultsdict = traineval(est, estimates, x_train, y_train, x_test, y_test, squaring=squaring, df=df)
                    # print("Accuracy: %.3f%% (%.3f%%)" % (results2.mean() * 100.0, results2.std() * 100.0))
                    res_dict = {
                        'Estimator': [est for x in range(len(resultsdict['PW20']))],
                        'PW20': resultsdict['PW20'],
                        'MAE': resultsdict['MAE'],
                        'R2': resultsdict['R2'],
                        'Time': resultsdict['Time']}
                    results.append(res_dict)
                df_res = pd.DataFrame()
                for res in results:
                    df_res = df_res.append(pd.DataFrame.from_dict(res))
                    print(f"\n\n{df_res.groupby(['Estimator']).agg(np.mean)}\n")

    dfResults = pd.DataFrame(results)
    dfResults["PW20"] = dfResults.apply(lambda x: ExitSquareBracket(x["PW20"]), axis=1).astype(float)
    dfResults["MAE"] = dfResults.apply(lambda x: ExitSquareBracket(x["MAE"]), axis=1).astype(float)
    dfResults["R2"] = dfResults.apply(lambda x: ExitSquareBracket(x["R2"]), axis=1).astype(float)
    dfResults["Time"] = dfResults.apply(lambda x: ExitSquareBracket(x["Time"]), axis=1).astype(float)
    dfResults["Estimator"] = dfResults.apply(lambda x: ExitSquareBracket(x["Estimator"]), axis=1)
    dfSummary = dfResults.groupby('Estimator').apply(np.mean)
    stddev = []
    confinterval = []
    dfResults.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfResults" + ".csv", ";")
    dfSummary.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfSummary" + ".csv", ";")
    for i in range(len(metric_columns)):
       for est in estimates:
          current_estimator = est
          current_metric = metric_columns[i]
          current_mean = dfSummary.loc[current_estimator][current_metric]
          metric_values = np.where(dfResults['Estimator'] == current_estimator, dfResults[current_metric],9999)
          metriclist = np.array(metric_values)
          metriclist = [j for j in metriclist if j != 9999]
          current_stddev = confintlimit95(metriclist)
          confinterval.append({'estimator': current_estimator, 'metric': current_metric, 'mean': current_mean,
                                         '95% CI lower bound': current_mean - current_stddev,
                                         '95% CI upper bound': current_mean + current_stddev})
    dfResults.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfResults" + ".csv", ";")
    dfSummary.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfSummary" + ".csv", ";")
    dfConfidence = pd.DataFrame(confinterval,
                                columns=['estimator', 'metric', 'mean', '95% CI lower bound', '95% CI upper bound'])
    dfConfidence.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfConfidence" + ".csv", ";")
    print("STOP HERE")


if __name__ == "__main__":
    main()


































































































































































