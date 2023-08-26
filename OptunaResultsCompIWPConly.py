import sklearn
import time
import math
import csv
import os
import os.path
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pathlib import Path
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
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, Ridge, Perceptron
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.svm import LinearSVR, SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
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
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from hpsklearn import HyperoptEstimator
from hpsklearn import any_classifier
from hpsklearn import any_preprocessing
from hpsklearn import any_regressor
from warfit_learn.estimators import Estimator
from warfit_learn.evaluation import evaluate_estimators
from warfit_learn.metrics.scoring import confidence_interval
from scipy.stats import norm, iqr, scoreatpercentile
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
from tpot import TPOTRegressor
from numpy import loadtxt
from copy import copy

def ExitSquareBracket(variable, floatbool):
    stringvar = str(variable)
    if stringvar.find('[') >= 0 and stringvar.find(']') >= 0:
        var1 = stringvar.replace('[', '')
        var2 = var1.replace(']', '')
        var2 = var2.replace("'", "")
        return var2
    else:
        if floatbool:
            return float(stringvar)
        else:
            return stringvar

def collect_Metrics(metrics, model, metric):
    container = []
    for i in range(len(metrics)):
        if metrics[i]['model'] == model:
          container.append(metrics[i]['metric'])
    return container

def collect_Metric(metrics, model, seed, metric):
    container = []
    for i in range(len(metrics)):
        if (metrics[i]['model'] == model) and (metrics[i]['seed'] == seed):
            container.append(metrics[i][metric])
    return container


def collect_Results(model, metric):
    container = []
    if Estimator == model:
        container.append(metric)
    return container


def variance(metric, mean):
    # meanvalue = np.nanmean(metric)
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


def std_deviation(metric, mean):
    return np.sqrt(variance(metric, mean))


def SList(series):
    return np.array(series.values.tolist())


def confintlimit95(metric, mean):
    return 1.96 * np.sqrt(variance(metric, mean) / len(metric))


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
            lo2 = mean - confintlimit95(data, mean)
            hi2 = mean + confintlimit95(data, mean)
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

def tune(objective, df, model, randomseed):
    ntrials = 50
    suffix = str(df).zfill(3)
    study = optuna.create_study(direction="maximize")
    # study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=ntrials)
    optuna_results = study.trials_dataframe()
    optuna_results['data'] = 'IWPC'
    optuna_results['direction'] = 'max'
    optuna_results['mlmodel'] = model
    params = study.best_params
    best_score = study.best_value
    best_trial = study.best_trial
    optuna_results['besttrial'] = best_trial
    optuna_results['bestvalue'] = best_score
    optuna_results['bestparam'] = params
    # optuna_results.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\OptunaAllModelW\model_" + model + "_" + str(randomseed) + "_" + suffix + ".csv", ";")
    plot_optimization_history(study)
    params = study.best_params
    best_score = study.best_value
    paramdict = {k: [v] for k, v in params.items()}
    dfHyperCurrent = pd.DataFrame(paramdict)
    dfHyperCurrent['model'] = model
    dfHyperCurrent['imputation'] = df
    dfHyperCurrent['score'] = best_score
    dfHyperCurrent['trials'] = ntrials
    suffix = str(df).zfill(3)
    if df == 1:
        dfHyper = pd.DataFrame()
    else:
        dfpre = df - 1
        suffixpre = str(dfpre).zfill(3)
        dfHyper = pd.read_csv(
            r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\OPTUNAHYPERPARAMETERS\model_" + model + "_" + str(
                randomseed) + "_" + suffixpre + ".csv", ";")
        if "Unnamed: 0" in dfHyper.columns:
            dfHyper.drop(["Unnamed: 0"], axis=1, inplace=True)
        if "Unnamed: 0.1" in dfHyper.columns:
            dfHyper.drop(["Unnamed: 0.1"], axis=1, inplace=True)
    frames = (dfHyper, dfHyperCurrent)
    dfHyper = pd.concat(frames)
    dfHyper.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\OPTUNAHYPERPARAMETERS\model_" + model + "_" + str(
        randomseed) + "_" + suffix + ".csv", ";")
    print(f"Best score: {best_score} \nOptimized parameters: {params}")
    return params, optuna_results


def dropColumn(IWPCparam, columnname, dfColumns, dfmod, dfIWPC, IWPCDF):
    if columnname in dfColumns:
        if IWPCparam == "IWPC":
            dfIWPC.drop([columnname], axis=1, inplace=True)
        elif IWPCparam == "IWPCDF":
            IWPCDF.drop([columnname], axis=1, inplace=True)
        else:
            dfmod.drop([columnname], axis=1, inplace=True)


def indexTo1(df):
    df.index = np.arange(1, len(df) + 1)

def tune(objective, df, model, randomseed):
    ntrials = 50
    suffix = str(df).zfill(3)
    study = optuna.create_study(direction="maximize")
    #study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=ntrials)
    optuna_results = study.trials_dataframe()
    optuna_results['data'] = 'IWPC'
    optuna_results['direction'] = 'max'
    optuna_results['mlmodel'] = model
    params = study.best_params
    best_score = study.best_value
    best_trial = study.best_trial
    optuna_results['besttrial'] = best_trial
    optuna_results['bestvalue'] = best_score
    optuna_results['bestparam'] = params
    #optuna_results.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\OptunaAllModelW\model_" + model + "_" + str(randomseed) + "_" + suffix + ".csv", ";")
    plot_optimization_history(study)
    params = study.best_params
    best_score = study.best_value
    paramdict = {k: [v] for k, v in params.items()}
    dfHyperCurrent = pd.DataFrame(paramdict)
    dfHyperCurrent['model'] = model
    dfHyperCurrent['imputation'] = df
    dfHyperCurrent['score'] = best_score
    dfHyperCurrent['trials'] = ntrials
    suffix = str(df).zfill(3)
    if df == 1:
        dfHyper = pd.DataFrame()
    else:
        dfpre = df - 1
        suffixpre = str(dfpre).zfill(3)
        dfHyper = pd.read_csv(
            r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\OPTUNAHYPERPARAMETERS\model_" + model + "_" + str(
                randomseed) + "_" + suffixpre + ".csv", ";")
        if "Unnamed: 0" in dfHyper.columns:
            dfHyper.drop(["Unnamed: 0"], axis=1, inplace=True)
        if "Unnamed: 0.1" in dfHyper.columns:
            dfHyper.drop(["Unnamed: 0.1"], axis=1, inplace=True)
    frames = (dfHyper, dfHyperCurrent)
    dfHyper = pd.concat(frames)
    dfHyper.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\OPTUNAHYPERPARAMETERS\model_" + model + "_" + str(
        randomseed) + "_" + suffix + ".csv", ";")
    print(f"Best score: {best_score} \nOptimized parameters: {params}")
    return params, optuna_results

def dropColumn(IWPCparam, columnname, dfColumns, dfmod, dfIWPC, IWPCDF):
    if columnname in dfColumns:
        if IWPCparam == "IWPC":
           dfIWPC.drop([columnname],axis=1,inplace=True)
        elif IWPCparam == "IWPCDF":
           IWPCDF.drop([columnname],axis=1,inplace=True)
        else:
           dfmod.drop([columnname], axis=1, inplace=True)


def indexTo1(df):
    df.index = np.arange(1, len(df) + 1)

def traineval(est: Estimator, xtrain, ytrain, xtest, ytest, squaring, df, randomseed):
    resultsdict = {'PW20': 0, 'MAE': 0, 'R2': 0, 'Time': '', 'Alg': ''}
    ml_learner = est.identifier
    # RANDOM_SEED = 66
    RANDOM_SEED = randomseed

    # 10-fold CV
    kfolds = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    model = est.estimator
    modelID = est.identifier
    gridFit = True
    # print(f'\n{est.identifier}...')
    #if est.identifier != "LR":  # tuning is not required for LR
    if est.identifier == "LR":
        start = time.time()

        def LR_Objective(trial):
            #scaler = MinMaxScaler()
            LR_Model = LinearRegression()
            #pipeline = make_pipeline(scaler, LR_Model)
            score = cross_val_score(LR_Model, xtrain, ytrain, cv=kfolds,
                                    scoring="neg_mean_absolute_error").mean()
            return score

        LR_params, dfOptuna = tune(LR_Objective, df, modelID, randomseed)
        end = time.time()
        model = LinearRegression(**LR_params)


    elif est.identifier == "XGBR":
            start = time.time()
            def XGBR_Objective(trial):
                _booster = trial.suggest_categorical('booster', ["gbtree"])
                _max_depth = trial.suggest_int('max_depth', 1, 4, step=1)
                _colsample_bynode = trial.suggest_categorical('colsample_bynode', [1])
                _learning_rate = trial.suggest_float('learning_rate', 0.01, 0.05, step=0.01)
                _n_estimators = trial.suggest_int('n_estimators', 400, 800, step=200)
                _min_child_weight = trial.suggest_int('min_child_weight', 21, 51, step=10)
                _subsample = trial.suggest_categorical('subsample', [0.877056, 0.957056, 0.677056, 0.757056, 0.917056])
                _colsample_bytree = trial.suggest_categorical('colsample_bytree',
                                                              [0.530084, 0.770084, 0.790084, 0.910084, 0.610084])
                _colsample_bylevel = trial.suggest_categorical('colsample_bylevel',
                                                               [0.66163511, 0.74163511, 0.94163511, 0.62163511,
                                                                0.90163511])
                # _gamma = trial.suggest_float('gamma',0.000,3.308368474,step=0.002)
                XGBR_model = XGBRegressor(max_depth=_max_depth, min_child_weight=_min_child_weight,
                                          colsample_bytree=_colsample_bytree, colsample_bylevel=_colsample_bylevel,
                                          colsample_bynode=_colsample_bynode, subsample=_subsample,
                                          n_estimators=_n_estimators)
                score = cross_val_score(XGBR_model, xtrain, ytrain, cv=kfolds, scoring="neg_mean_absolute_error").mean()
                return score

            XGBR_params, dfOptuna = tune(XGBR_Objective, df, est.identifier, randomseed)
            end = time.time()
            model = XGBRegressor(**XGBR_params)

    elif est.identifier == "RF":
            start = time.time()

            def RF_Objective(trial):
                # _min_samples_leaf = trial.suggest_categorical('min_samples_leaf', [2,3,4,5,6,7,8,9,10,11,12])
                # _min_samples_leaf = trial.suggest_categorical('min_samples_leaf',[8,9,11,12])
                _min_samples_leaf = trial.suggest_categorical('min_samples_leaf', [9, 11])
                _min_impurity_decrease = trial.suggest_categorical('min_impurity_decrease', [0.0])
                # _max_depth = trial.suggest_categorical('max_depth', [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
                # _max_depth = trial.suggest_categorical('max_depth', [4,6,10])
                _max_depth = trial.suggest_categorical('max_depth', [4])
                # _n_estimators = trial.suggest_categorical('n_estimators', [50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200])
                _n_estimators = trial.suggest_categorical('n_estimators', [60, 80, 110])
                # _min_samples_split = trial.suggest_categorical("min_samples_split", [2, 3, 4, 5, 6, 7, 8, 9, 10])
                # _min_samples_split = trial.suggest_categorical("min_samples_split",[4,5,6,7,8,9,10])
                _min_samples_split = trial.suggest_categorical("min_samples_split", [5, 6, 7])
                # _max_features=trial.suggest_float("max_features",0.0,50.0,step=5.0)
                # _n_estimators = trial.suggest_categorical("n_estimators", [50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200])
                # _max_depth = trial.suggest_categorical("max_depth", [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
                # _min_samp_split = trial.suggest_categorical("min_samples_split", [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
                # _min_samples_leaf = trial.suggest_categorical("min_samples_leaf", [2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0])
                # _max_features = trial.suggest_categorical("max_features", [10,15,20,25,30,35,40,45,50])
                RF_model = RandomForestRegressor(
                    max_depth=_max_depth,
                    min_samples_split=_min_samples_split,
                    min_impurity_decrease=_min_impurity_decrease,
                    min_samples_leaf=_min_samples_leaf,
                    n_estimators=_n_estimators

                )

                score = cross_val_score(RF_model, xtrain, ytrain, cv=kfolds, scoring="neg_mean_absolute_error").mean()
                return score
            RF_params, dfOptuna = tune(RF_Objective, df, est.identifier, randomseed)
            end = time.time()
            model = RandomForestRegressor(**RF_params)

    elif est.identifier == 'DTR':
                # define parameter space
                start = time.time()

                def DTR_objective(trial):
                    _min_samples_leaf = trial.suggest_categorical("min_samples_leaf", [1])
                    _max_depth = trial.suggest_categorical('max_depth', [2])
                    _min_impurity_decrease = trial.suggest_categorical("min_impurity_decrease", [0.0])
                    _max_features = trial.suggest_categorical('max_features',
                                                              [0.355298191, 0.55247903, 0.5616223889, 0.900168726])
                    _min_samples_split = trial.suggest_categorical("min_samples_split", [2])
                    _max_leaf_nodes = trial.suggest_categorical("max_leaf_nodes", [None, 10])
                    _min_weight_fraction_leaf = trial.suggest_categorical("min_weight_fraction_leaf", [0.0])
                    _splitter = trial.suggest_categorical('splitter', ["random"])

                    DTR_model = DecisionTreeRegressor(min_samples_leaf=_min_samples_leaf, max_depth=_max_depth,
                                                      max_features=_max_features,
                                                      max_leaf_nodes=_max_leaf_nodes,
                                                      min_impurity_decrease=_min_impurity_decrease,
                                                      min_weight_fraction_leaf=_min_weight_fraction_leaf,
                                                      min_samples_split=_min_samples_split,
                                                      splitter=_splitter)

                    score = cross_val_score(DTR_model, xtrain, ytrain, cv=kfolds,
                                            scoring="neg_mean_absolute_error").mean()
                    return score

                DTR_params, dfOptuna = tune(DTR_objective, df, modelID, randomseed)
                end = time.time()
                model = DecisionTreeRegressor(**DTR_params)


    elif est.identifier == "GBR":
                start = time.time()

                def GBR_Objective(trial):
                    # _learning_rate = trial.suggest_categorical("learning_rate",[0.159470245, 0.034293879, 0.055418218, 0.030973395, 0.005561334, 0.132329369, 0.03803106, 0.072507733, 0.032894225,
                    #                                                           0.0743424267, 0.023528704, 0.003805887, 0.009816521, 0.067644981, 0.075711757, 0.062691453, 0.096134727])
                    _learning_rate = trial.suggest_categorical("learning_rate",
                                                               [0.023528704, 0.067644981, 0.030973395, 0.003805887,
                                                                0.132329369, 0.03803106, 0.072507733, 0.009816521])

                    _max_depth = trial.suggest_categorical("max_depth", [2, 3, 4, 5])
                    # _n_estimators = trial.suggest_categorical("n_estimators",[229,196,119,397,543,22,45,59,955,70,502,800,494,59,645,108,25])
                    _n_estimators = trial.suggest_categorical("n_estimators", [25, 45, 59, 119, 543])
                    _min_samples_leaf = trial.suggest_categorical("min_samples_leaf", [15, 21, 23])
                    _min_samples_split = trial.suggest_categorical("min_samples_split", [2, 3])
                    GBR_model = GradientBoostingRegressor(learning_rate=_learning_rate, max_depth=_max_depth,
                                                          n_estimators=_n_estimators,
                                                          min_samples_leaf=_min_samples_leaf,
                                                          min_samples_split=_min_samples_split)
                    score = cross_val_score(GBR_model, xtrain, ytrain, cv=kfolds,
                                            scoring="neg_mean_absolute_error").mean()
                    return score

                GBR_params, dfOptuna = tune(GBR_Objective, df, modelID, randomseed)
                end = time.time()
                model = GradientBoostingRegressor(**GBR_params)

    elif est.identifier == "MLPR":
                # activation = 'identity', alpha = 0.009260944818691528,
                # beta_1 = 0.8304148442169565, beta_2 = 0.9847593650340831,
                # epsilon = 4.968151316490382e-06, learning_rate = 'adaptive',
                # learning_rate_init = 0.014389028832229495, max_fun = 27459,
                # max_iter = 240, momentum = 0.871153667593362,
                # power_t = 0.6130708398995247, random_state = 2, solver = 'lbfgs',
                # tol = 0.008075623520655316,
                # validation_fraction = 0.036450106641084085
                start = time.time()

                def MLPR_Objective(trial):
                    scaler = MinMaxScaler()
                    _mlpr_hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes",
                                                                         [(2,), (3,), (1,), (3, 1,), (3, 2,), (3, 3,),
                                                                          (2, 1,), (2, 2,)])

                    # _mlpr_hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes",[(2,),(4,),(8,),(4,1,),(4,2,),(4,4,),(2,1,),(2,2,),(8,2,),(8,4,),(8,1,),(8,6,)])
                    # _mlpr_hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes",[(20,),(10,),(11,),(12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,
                    # _mlpr_learning_rate_init = trial.suggest_categorical("learning_rate_init",[ 0.0001, 0.0015,  0.002]) # WAS [0.001]
                    _mlpr_max_iter = trial.suggest_categorical("max_iter",
                                                               [240, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000,
                                                                9500, 10000])  # WAS [1000,1500,2000]
                    # _mlpr_max_iter = trial.suggest_categorical("max_iter", [4500, 5000, 5500, 6000, 6500, 7000, 7500,8000])  # WAS [1000,1500,2000]
                    _mlpr_solver = trial.suggest_categorical("solver", ['lbfgs'])
                    _mlpr_tol = trial.suggest_categorical("tol", [0.008075623520655316])

                    # _mlpr_activation = trial.suggest_categorical('activation',['logistic','relu'])
                    _mlpr_learning_rate_init = trial.suggest_categorical('learning_rate_init', [0.014389028832229495])
                    _mlpr_epsilon = trial.suggest_categorical('epsilon', [4.968151316490382e-06])
                    _mlpr_activation = trial.suggest_categorical('activation', ['identity'])
                    _mlpr_alpha = trial.suggest_categorical('alpha', [0.009260944818691528])
                    _mlpr_beta_1 = trial.suggest_categorical('beta_1', [0.8304148442169565])
                    _mlpr_beta_2 = trial.suggest_categorical('beta_2', [0.9847593650340831])
                    _mlpr_power_t = trial.suggest_categorical('power_t', [0.6130708398995247])
                    _mlpr_max_fun = trial.suggest_categorical('max_fun', [27459])
                    _mlpr_momentum = trial.suggest_categorical('momentum', [0.871153667593362])
                    _mlpr_validation_fraction = trial.suggest_categorical('validation_fraction', [0.036450106641084085])
                    # _mlpr_learning_rate = trial.suggest_categorical("learning_rate",['adaptive'])
                    # _mlpr_activation = trial.suggest_categorical("activation",["relu"])
                    # MLPR_Model = MLPRegressor(hidden_layer_sizes = _mlpr_hidden_layer_sizes, learning_rate_init=_mlpr_learning_rate_init,
                    #                          max_iter = _mlpr_max_iter, learning_rate = _mlpr_learning_rate, activation=_mlpr_activation,
                    #                          solver=_mlpr_solver)
                    MLPR_Model = MLPRegressor(hidden_layer_sizes=_mlpr_hidden_layer_sizes,
                                              max_iter=_mlpr_max_iter,
                                              activation=_mlpr_activation,
                                              solver=_mlpr_solver, random_state=randomseed,
                                              tol=_mlpr_tol, learning_rate_init=_mlpr_learning_rate_init,
                                              epsilon=_mlpr_epsilon, alpha=_mlpr_alpha,
                                              beta_1=_mlpr_beta_1, beta_2=_mlpr_beta_2,
                                              power_t=_mlpr_power_t, max_fun=_mlpr_max_fun,
                                              momentum=_mlpr_momentum, validation_fraction=_mlpr_validation_fraction)
                    pipeline = make_pipeline(scaler, MLPR_Model)
                    score = cross_val_score(pipeline, xtrain, ytrain, cv=kfolds,
                                            scoring="neg_mean_absolute_error").mean()
                    return score

                MLPR_params, dfOptuna = tune(MLPR_Objective, df, modelID, randomseed)
                end = time.time()
                model = MLPRegressor(**MLPR_params)


    elif est.identifier == "KNNR":
                start = time.time()

                def KNN_Objective(trial):
                    scaler = MinMaxScaler()
                    _knn_n_neighbors = trial.suggest_int("n_neighbors", 2, 200, step=1)
                    _knn_metric = trial.suggest_categorical("metric", ['euclidean', 'manhattan', 'minkowski'])
                    _knn_weights = trial.suggest_categorical("weights", ['uniform', 'distance'])
                    KNN_Model = KNeighborsRegressor(n_neighbors=_knn_n_neighbors, metric=_knn_metric,
                                                    weights=_knn_weights)
                    pipeline = make_pipeline(scaler, KNN_Model)
                    score = cross_val_score(pipeline, xtrain, ytrain, cv=kfolds,
                                            scoring="neg_mean_absolute_error").mean()
                    return score
                KNN_params, dfOptuna = tune(KNN_Objective, df, modelID, randomseed)
                end = time.time()
                model = KNeighborsRegressor(**KNN_params)

    elif est.identifier in ("LASSO"):
                    start = time.time()
                    def LASSO_Objective(trial):
                        scaler = MinMaxScaler()
                        # _alpha = trial.suggest_float('alpha',0,0.05,step=0.001)
                        _alpha = trial.suggest_categorical("alpha", [0.006644814, 0.010214848, 0.000233117, 0.012706666,
                                                                     0.009199882, 0.000377741, 0.00075485, 0.0000992077,
                                                                     0.000528944])
                        _max_iter = trial.suggest_categorical("max_iter",
                                                              [828, 1181, 1073, 897, 1120, 1156, 880, 1142, 1140])
                        lasso_model = Lasso(alpha=_alpha, max_iter=_max_iter)
                        pipeline = make_pipeline(scaler, lasso_model)
                        score = cross_val_score(pipeline, xtrain, ytrain, cv=kfolds,
                                                scoring="neg_mean_absolute_error").mean()
                        return score

                    LASSO_params, dfOptuna = tune(LASSO_Objective, df, modelID, randomseed)
                    end = time.time()
                    model = Lasso(**LASSO_params)
    elif est.identifier == "RIDGE":
                    start = time.time()

                    def RIDGE_Objective(trial):
                        _alpha = trial.suggest_categorical("alpha",
                                                           [2.236514, 0.060708, 0.204956, 0.133253, 0.512693, 1.269382,
                                                            0.969999, 1.060682, 0.20964, 2.75604])
                        # _alpha = trial.suggest_float("alpha",13,16,step=0.5)
                        _max_iter = trial.suggest_categorical("max_iter",
                                                              [1071, 858, 1108, 1038, 930, 1220, 1214, 999, 1035, 912])
                        ridge_model = Ridge(alpha=_alpha, max_iter=_max_iter)
                        scaler = MinMaxScaler()
                        pipeline = make_pipeline(scaler, ridge_model)

                        score = cross_val_score(pipeline, xtrain, ytrain, cv=kfolds,
                                                scoring="neg_mean_absolute_error").mean()
                        return score

                    RIDGE_params, dfOptuna = tune(RIDGE_Objective, df, modelID, randomseed)
                    end = time.time()
                    model = Ridge(**RIDGE_params)

    elif est.identifier == "ELNET":
                    start = time.time()

                    def ELNET_Objective(trial):
                        scaler = MinMaxScaler()
                        # _alpha = trial.suggest_float('alpha', 0,0.05,step=0.001)
                        # _l1_ratio = trial.suggest_float('l1_ratio', 0.01, 0.99, step=0.01)
                        _alpha = trial.suggest_categorical("alpha",
                                                           [0.000198094, 0.015176516, 0.0, 0.003264721, 0.00066628,
                                                            0.001536277, 0.0004508042, 0.002504531])
                        _l1_ratio = trial.suggest_categorical("l1_ratio",
                                                              [0.023118993, 0.448966271, 0.664744323, 0.34692344,
                                                               0.550401256, 0.245397394, 0.4617206, 0.312367882,
                                                               0.236122903, 0.094718953, 0.631230102])
                        _max_iter = trial.suggest_categorical("max_iter",
                                                              [1159, 818, 1000, 1129, 887, 1036, 876, 1149, 970, 957])
                        ELNET_model = ElasticNet(alpha=0.01, l1_ratio=_l1_ratio, max_iter=_max_iter)
                        pipeline = make_pipeline(scaler, ELNET_model)
                        score = cross_val_score(pipeline, xtrain, ytrain, cv=kfolds,
                                                scoring="neg_mean_absolute_error").mean()
                        return score
                    ELNET_params, dfOptuna = tune(ELNET_Objective, df, modelID, randomseed)
                    end = time.time()
                    model = ElasticNet(**ELNET_params)
                    # param_grid = [{'alg__alpha' : np.logspace(-4, -2, 9)}]
                    # if est.identifier == "ELNET":
                    #    param_grid = [{'alg__alpha': np.logspace(-4, -2, 9),
                    #                   'alg__l1_ratio': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]}]
    elif est.identifier == "SVREG":
                    start = time.time()
                    def SVREG_Objective(trial):
                        scaler = StandardScaler()
                        # _gamma = trial.suggest_categorical('gamma', ['auto', 'scale'])
                        _regParam1 = trial.suggest_categorical('C', [1])
                        _regParam2 = trial.suggest_categorical('epsilon', [0.1])
                        _gamma = trial.suggest_categorical('gamma', ['scale'])
                        # _C = trial.suggest_categorical("C",  [0,0.1,0,2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0])
                        # _epsilon = trial.suggest_categorical("epsilon", [0,0.1,0,2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0])
                        _kernel = trial.suggest_categorical("kernel", ['rbf'])
                        # _coef0 = trial.suggest_float("coef0", 0.01, 1)
                        # svr = SVR(C=_C, epsilon=_epsilon, kernel='poly')
                        SVREG_model = sklearn.svm.SVR(gamma=_gamma, C=_regParam1, epsilon=_regParam2, kernel=_kernel)
                        pipeline = make_pipeline(scaler, SVREG_model)
                        score = cross_val_score(pipeline, xtrain, ytrain, cv=kfolds,
                                                scoring="neg_mean_absolute_error").mean()
                        return score

                    SVREG_params, dfOptuna = tune(SVREG_Objective, df, modelID, randomseed)
                    end = time.time()
                    model = sklearn.svm.SVR(**SVREG_params)

    elif est.identifier == "GBR":
                    start = time.time()

                    def GBR_Objective(trial):
                        _subsample = trial.suggest_float('subsample', 0.5, 1.0, step=0.1)
                        _n_estimators = trial.suggest_int('n_estimators', 50, 3000, step=100)
                        _learning_rate = trial.suggest_float('learning_rate', 0.001, 1.0, step=0.01)
                        _max_depth = trial.suggest_int('max_depth', 1, 10, step=1)
                        GBR_model = GradientBoostingRegressor(subsample=_subsample, n_estimators=_n_estimators,
                                                              learning_rate=_learning_rate,
                                                              max_depth=_max_depth)
                        score = cross_val_score(GBR_model, xtrain, ytrain, cv=kfolds,
                                                scoring="neg_mean_absolute_error").mean()
                        return score

                    GBR_params, dfOptuna = tune(GBR_Objective, df, modelID, randomseed)
                    end = time.time()
                    model = GradientBoostingRegressor(**GBR_params)

    elif est.identifier == "LGB":
                    start = time.time()

                    def LGB_Objective(trial):
                        _num_leaves = trial.suggest_categorical("num_leaves",
                                                                [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
                        _max_depth = trial.suggest_categorical("max_depth",
                                                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                                                17, 18, 19, 20])
                        _learning_rate = trial.suggest_categorical("learning_rate",
                                                                   [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
                                                                    0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                                                    1.0])
                        _n_estimators = trial.suggest_categorical("n_estimators",
                                                                  [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550,
                                                                   600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050,
                                                                   1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500,
                                                                   1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950,
                                                                   2000])
                        _min_child_weight = trial.suggest_categorical("min_child_weight",
                                                                      [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                                                       1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
                                                                       6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])
                        _reg_alpha = trial.suggest_categorical('reg_alpha',
                                                               [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                                                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5,
                                                                2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
                                                                7.5, 8.0, 8.5, 9.0, 9.5, 10.0])
                        _reg_lambda = trial.suggest_categorical('reg_lambda',
                                                                [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                                                 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5,
                                                                 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
                                                                 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])
                        _subsample = trial.suggest_categorical('subsample',
                                                               [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                                                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5,
                                                                2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
                                                                7.5, 8.0, 8.5, 9.0, 9.5, 10.0])

                        lgbr = lgb.LGBMRegressor(objective='regression',
                                                 num_leaves=_num_leaves,
                                                 max_depth=_max_depth,
                                                 learning_rate=_learning_rate,
                                                 n_estimators=_n_estimators,
                                                 min_child_weight=_min_child_weight,
                                                 subsample=_subsample,
                                                 reg_alpha=_reg_alpha,
                                                 reg_lambda=_reg_lambda,
                                                 random_state=RANDOM_SEED,
                                                 )

                        score = cross_val_score(
                            lgbr, xtrain, ytrain, cv=kfolds, scoring="neg_mean_absolute_error"
                        ).mean()
                        return score

                    LGB_params = tune(LGB_Objective, df, modelID, randomseed)
                    end = time.time()
                    model = lgb.LGBMRegressor(**LGB_params)

    elif est.identifier == "STACK":
                    # stack models
                    stack = StackingRegressor(
                        estimators=[
                            ('ridge', ridge),
                            ('lasso', lasso),
                            ('elasticnet', elasticnet),
                            ('randomforest', rf),
                            ('gradientboostingregressor', gbr),
                            ('xgb', xgbr),
                            ('lgb', lgbr),
                            # ('svr', svr), # Not using this for now as its score is significantly worse than the others
                        ],
                        cv=kfolds)
                    stack.fit(X, y)

    fitted = model.fit(xtrain, ytrain)
    timeElapsed = end - start
    ypred = fitted.predict(xtest)
    if squaring:
        ytest = np.square(ytest)
        ypred = np.square(ypred)
    PW20 = PercIn20(ytest, ypred)
    MAE = mean_absolute_error(ytest, ypred)
    randstring = "_" + str(randomseed) + "_"
    if True:
        suffix = str(df).zfill(3)
        dfHyper = pd.read_csv(
            r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\OPTUNAHYPERPARAMETERS\model_" + ml_learner + randstring + suffix + ".csv",";")
        dfHyper.loc[df - 1, "mae"] = MAE
        dfHyper.loc[df - 1, "pw20"] = PW20
        dfHyper.to_csv(
            r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\OPTUNAHYPERPARAMETERS\model_" + ml_learner + randstring + suffix + ".csv",";")
        dfHyper = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\OPTUNAHYPERPARAMETERS\model_" + ml_learner + randstring + suffix + ".csv",";")
        if "Unnamed: 0" in dfHyper.columns:
            dfHyper.drop(["Unnamed: 0"], axis=1, inplace=True)
        if "Unnamed: 0.1" in dfHyper.columns:
            dfHyper.drop(["Unnamed: 0.1"], axis=1, inplace=True)
        dfHyper.to_csv(
            r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\OPTUNAHYPERPARAMETERS\model_" + ml_learner + randstring + suffix + ".csv",";")
    R2 = RSquared(ytest, ypred)
    resultsdict['PW20'] = [PW20]
    resultsdict['MAE'] = [MAE]
    resultsdict['R2'] = [R2]
    resultsdict['Time'] = [timeElapsed]
    resultsdict['Alg'] = [model]

    return resultsdict, dfOptuna


def main():
    firstRun = True
    modelstats = dict()
    interstats = dict()
    modelsumm = dict()
    smpResults = []
    interResults = []
    metric_columns = ['MAE', 'PW20']
    listmodels = ['WarPATH']
    mlmodels = []
    RF = RandomForestRegressor()
    mlmodels.append(Estimator(RF, 'RF'))
    LR = LinearRegression()
    mlmodels.append(Estimator(LR, 'LR'))
    KNNR = KNeighborsRegressor()
    RR = Ridge()
    LAS = Lasso()
    ELNR = ElasticNet()
    svr = sklearn.svm.SVR()
    GBR = GradientBoostingRegressor()
    DTR = DecisionTreeRegressor()
    MLPR = MLPRegressor()
    mlmodels.append(Estimator(MLPR, 'MLPR'))
    mlmodels.append(Estimator(GBR, 'GBR'))
    mlmodels.append(Estimator(svr, 'SVREG'))
    mlmodels.append(Estimator(LAS, 'LASSO'))
    mlmodels.append(Estimator(ELNR, "ELNET"))
    mlmodels.append(Estimator(RR, "RIDGE"))
    mlmodels.append(Estimator(KNNR, "KNNR"))
    mlmodels.append(Estimator(DTR, 'DTR'))
    XGBR = XGBRegressor()
    mlmodels.append(Estimator(XGBR,'XGBR'))
    alg = Estimator.identifier
    modelsumm = {'model': alg}
    for _, est in enumerate(mlmodels):
        estimates = []
        print("Processing ML model ", est.identifier)
        alg = est.identifier
        estimates.append(Estimator(est.estimator, est.identifier))
        randomStates = []
        randomStates.append(0)
        randomStates.append(33)
        randomStates.append(42)
        randomStates.append(66)
        randomStates.append(99)
        randomStates.append(102)
        randomStates.append(113)
        randomStates.append(143)
        for state in range(len(randomStates)):
            randomseed = randomStates[state]
            modelstats = {'model':alg, 'seed': randomseed, 'MAE' : [], 'PW20' : []}
            interstats = {'model': alg, 'seed': randomseed, 'MAE': [], 'PW20': []}
            modelsumm = {'model':alg}
            intersumm = {'model':alg}
            metrics = []
            print("Processing random state", randomseed)
            imps = 50
            Bfactor = (imps+1)/imps
            for j in range(imps):
              suffix = str(j+1).zfill(3)
              dfImp =  pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\OptunaAllModelIWPC\model_" + alg + "_" + str(
                       randomseed) + "_" + suffix + ".csv", ";")
              dfImpInter = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\OptunaIWPConly\model_" + alg + "_" + str(
                  randomseed) + "_050" + ".csv", ";")

              for index, row in dfImp.iterrows():
                mae_value = row['MAE']
                PW20_value = row['PW20']
                smpResults.append({'Imp': (j+1), 'seed': randomseed, 'model': alg, 'MAE': mae_value, 'PW20': PW20_value})
            MAE_list = collect_Metric(smpResults, alg, randomseed,'MAE')
            PW20_list = collect_Metric(smpResults,alg,randomseed,'PW20')
            modelstats.update({'MAE': MAE_list, 'PW20': PW20_list})
            for index, row in dfImpInter.iterrows():
                mae_value = row['mae']
                PW20_value = row['pw20']
                impcounter = row['imputation']
                interResults.append({'Imp': impcounter, 'seed': randomseed, 'model': alg, 'MAE': mae_value, 'PW20': PW20_value})
            MAE_list = collect_Metric(interResults, alg, randomseed, 'MAE')
            PW20_list = collect_Metric(interResults, alg, randomseed, 'PW20')
            interstats.update({'MAE': MAE_list, 'PW20': PW20_list})
            if state == 0:
              dfmodel = pd.DataFrame(modelstats)
              dfintermodel = pd.DataFrame(interstats)
            else:
              dfmodelnew = pd.DataFrame(modelstats)
              dfintermodelnew = pd.DataFrame(interstats)
              frames = (dfmodel, dfmodelnew)
              framesinter = (dfintermodel,dfintermodelnew)
              dfintermodel = pd.concat(framesinter)
              dfmodel = pd.concat(frames)
            #dfmodel.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_models_" + alg + ".csv", ";")
            #dfintermodel.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATHInter_" + alg + ".csv", ";")
            if state+1 == len(randomStates):
               mean_MAE = dfmodel['MAE'].mean()
               mean_MAE_inter = dfintermodel['MAE'].mean()
               mean_PW20 = dfmodel["PW20"].mean()
               mean_PW20_inter = dfintermodel['PW20'].mean()
               std_MAE = dfmodel['MAE'].std()
               std_MAE_inter = dfintermodel['MAE'].std()
               std_PW20 = dfmodel['PW20'].std()
               std_PW20_inter = dfintermodel['PW20'].std()
               var_MAE = dfmodel['MAE'].var()
               var_MAE_inter = dfintermodel['MAE'].var()
               var_PW20 = dfmodel['PW20'].var()
               var_PW20_inter = dfintermodel['PW20'].var()
               std_MAE2 = std_deviation(np.array(dfmodel['MAE']), mean_MAE)
               std_MAE2_inter = std_deviation(np.array(dfintermodel['MAE']), mean_MAE_inter)
               var_MAE2 = variance(np.array(dfmodel['MAE']), mean_MAE)
               var_MAE2_inter = variance(np.array(dfintermodel['MAE']), mean_MAE_inter)
               std_PW202 = np.square(std_deviation(np.array(dfmodel['PW20']), mean_PW20))
               std_PW202_inter = np.square(std_deviation(np.array(dfintermodel['PW20']), mean_PW20_inter))
               var_PW202 = variance(np.array(dfmodel['PW20']), mean_PW20)
               var_PW202_inter = variance(np.array(dfintermodel['PW20']), mean_PW20_inter)
               print('std_MAE', std_MAE)
               print('std_MAE2', std_MAE2)
               print('var_MAE', var_MAE)
               print('var_MAE2', var_MAE2)
               print('std_MAE', std_MAE)
               print('std_MAE2', std_MAE2)
               print('var_MAE', var_MAE)
               print('var_MAE2', var_MAE2)
               print('std_MAE_inter', std_MAE_inter)
               print('std_MAE2_inter', std_MAE2_inter)
               print('var_MAE_inter', var_MAE_inter)
               print('var_MAE2_inter', var_MAE2_inter)
               if firstRun == True:
                 dfsumm = pd.DataFrame()
                 dfsumminter = pd.DataFrame()
                 dfsummadd = pd.DataFrame([{'model':alg,'MAE': mean_MAE,'Stdev_MAE':std_MAE,'Var_MAE':var_MAE,
                                            'PW20': mean_PW20,'Stdev_PW20': std_PW20,'Var_PW20': var_PW20}])
                 dfsumminteradd = pd.DataFrame([{'model':alg,'MAE': mean_MAE_inter,'Stdev_MAE': std_MAE_inter,'Var_MAE':var_MAE_inter,
                                            'PW20': mean_PW20_inter,'Stdev_PW20': std_PW20_inter,'Var_PW20': var_PW20_inter,
                                            'CI_MAE_Lower': 0, 'CI_MAE_Upper': 0, 'CI_PW20_Lower': 0,'CI_PW20_Upper': 0}])

                 frames = (dfsumm, dfsummadd)
                 framesinter = (dfsumminter, dfsumminteradd)
                 dfsumm=pd.concat(frames)
                 dfsumminter = pd.concat(framesinter)
                 firstRun = False
               else:
                 dfsummadd = pd.DataFrame([{'model':alg,'MAE': mean_MAE,'Stdev_MAE':std_MAE,'Var_MAE':var_MAE,
                                            'PW20': mean_PW20,'Stdev_PW20': std_PW20,'Var_PW20': var_PW20}])

                 dfsumminteradd = pd.DataFrame(
                     [{'model': alg, 'MAE': mean_MAE_inter, 'Stdev_MAE': std_MAE_inter, 'Var_MAE': var_MAE_inter,
                      'PW20': mean_PW20_inter, 'Stdev_PW20': std_PW20_inter, 'Var_PW20': var_PW20_inter,
                      'CI_MAE_Lower':0, 'CI_MAE_Upper':0, 'CI_PW20_Lower':0, 'CI_PW20_Upper':0}])
                 framesinter = (dfsumminter, dfsumminteradd)
                 frames = (dfsumm, dfsummadd)
                 framesinter = (dfsumminter, dfsumminteradd)
                 dfsumm = pd.concat(frames)
                 dfsumminter = pd.concat(framesinter)

    dfsumminter.reset_index(level=0, inplace=True)
    dfsumminter.drop(columns=dfsumminter.columns[0], axis = 1, inplace = True)
    dfsumminter.reindex()
    for index, row in dfsumminter.iterrows():
        modelinter = row['model']
        theta_inter_MAE = row['MAE']
        MAE_inter_var = row['Var_MAE']
        theta_inter_PW20 = row['PW20']
        PW20_inter_var = row['Var_PW20']
        for index2, row2 in dfsumm.iterrows():
            modelintra = row2['model']
            if modelintra == modelinter:
               MAE_intra_var = row2['Var_MAE']
               PW20_intra_var = row2['Var_PW20']
               totalvar_MAE = MAE_intra_var + (MAE_inter_var*Bfactor)
               SEPooled_MAE = np.sqrt(totalvar_MAE)
               totalvar_PW20 = PW20_intra_var + PW20_inter_var*Bfactor
               SEPooled_PW20 = np.sqrt(totalvar_PW20)
               CI_lower_MAE, CI_upper_MAE   = theta_inter_MAE-1.96*SEPooled_MAE,theta_inter_MAE+1.96*SEPooled_MAE
               CI_lower_PW20, CI_upper_PW20 = theta_inter_MAE-1.96*SEPooled_PW20,theta_inter_PW20+1.96*SEPooled_PW20
               dfsumminter.at[index, 'CI_MAE_Lower']   = CI_lower_MAE
               dfsumminter.at[index, 'CI_MAE_Upper']  =  CI_upper_MAE
               dfsumminter.at[index, 'CI_PW20_Lower'] = CI_lower_PW20
               dfsumminter.at[index, 'CI_PW20_Upper'] = CI_upper_PW20

    print(dfsumm)
    dfsumm.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_ALLmodelsIWPC_"  + ".csv", ";")
    print(dfsumminter)
    dfsumminter.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_ALLmodelsInterIWPC_" + ".csv", ";")

if __name__ == "__main__":
    main()

