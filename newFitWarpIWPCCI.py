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
from random import sample
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

def tupleTransform(stringmod):
    stringlist = list(stringmod)
    onetuple = False
    list1 = []
    if (" ") in stringlist:
      tupletemp = ("a","b")
      stringlist.remove(" ")
    else:
      onetuple = True
      tupletemp = ("a",)
    tuplelist = list(tupletemp)
    poplist = []
    index1 = stringlist.index(",")
    if index1 > 0:
       val = stringlist[index1-1]
       poplist.append(int(val))
       if onetuple == False:
          val2 = stringlist[index1+1]
          poplist.append(int(val2))
          tuplelist[0] = poplist[0]
          tuplelist[1] = poplist[1]
       else:
           tuplelist[0]=poplist[0]
       tuplenew = tuple(tuplelist)
       return tuplenew
    else:
        return stringmod


def find(lst, key, value):
    for i, dic in enumerate(lst):
        if dic[key] == value:
            return i
    return -1


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


def listAverage(list):
    if len(list) > 0:
        return sum(list) / len(list)
    else:
        return 0


# def collect_Metrics(metrics, model, metric):
#    container = []
#    for i in range(len(metrics)):
#        if metrics[metric][metrics['Estimator'].any() == model]:
#            container.append(metrics[metric][metrics['Estimator'] == model].values)
#    return container

def collect_Metrics(metrics, model, metric):
    container = []
    for i in range(len(metrics)):
        if (metrics[i]['model'] == model):
            # print(metrics[i]['model'], metrics[i]['metric'], metrics[i]['value'])
            container.append(metrics[i][metric])
    return container


def collect_Results(metrics, model, metric):
    container = []
    for i in range(len(metrics)):
        if (metrics[i]['model'] == model):
            container.append(metrics[i][metric])
    return container


# def collect_Results(model, metric):
#    container = []
#    if Estimator == model:
#        container.append(metric)
#    return container

# def variance(metric,mean):
#    #meanvalue = np.nanmean(metric)
#    meanvalue = mean
#    sumsquares = 0
#    for i in range(len(metric)):
#        core = abs(metric[i] - meanvalue)
#        sumsquares += np.square(core)
#    if len(metric) == 1:
#        variance = 0
#    else:
#        variance = sumsquares / ((len(metric) - 1))
#    return variance

def variance(metric):
    meanvalue = np.mean(metric)
    sumsquares = 0
    for i in range(len(metric)):
        core = abs(metric[i] - meanvalue)
        sumsquares += np.square(core)
    variance = sumsquares / ((len(metric) - 1))
    return variance


# def std_deviation(metric,mean):
#   return np.sqrt(variance(metric,mean))

def std_deviation(metric):
    return np.sqrt(variance(metric))


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
    #study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=ntrials)
    optuna_results = study.trials_dataframe()
    optuna_results['data'] = 'War-IWPC'
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

def dropColumn(IWPCparam, columnname, dfColumns, dfmod, dfIWPC):
    if columnname in dfColumns:
        if IWPCparam == "IWPC":
            dfIWPC.drop([columnname], axis=1, inplace=True)
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
    metric_columns = ['MAE', 'PW20']
    listmodels = ['WarPATH']
    mlmodels = []
    #RF = RandomForestRegressor()
    #mlmodels.append(Estimator(RF, 'RF'))
    #LR = LinearRegression()
    #mlmodels.append(Estimator(LR, 'LR'))
    #KNNR = KNeighborsRegressor()
    #RR = Ridge()
    #LAS = Lasso()
    #ELNR = ElasticNet()
    #svr= sklearn.svm.SVR()
    #GBR = GradientBoostingRegressor()
    #MLPR = MLPRegressor()
    #mlmodels.append(Estimator(MLPR, 'MLPR'))
    #DTR = DecisionTreeRegressor()
    #mlmodels.append(Estimator(GBR,'GBR'))
    #mlmodels.append(Estimator(svr,'SVREG'))
    #mlmodels.append(Estimator(LAS,'LASSO'))
    #mlmodels.append(Estimator(ELNR,"ELNET"))
    #mlmodels.append(Estimator(RR, "RIDGE"))
    #mlmodels.append(Estimator(KNNR, "KNNR"))
    #mlmodels.append(Estimator(DTR, 'DTR'))
    XGBR = XGBRegressor()
    mlmodels.append(Estimator(XGBR,'XGBR'))
    for _, est in enumerate(mlmodels):
        dfConf = pd.DataFrame()
        estimates = []
        print("Processing ML model ", est.identifier)
        estimates.append(Estimator(est.estimator, est.identifier))
        randomStates = []
        #randomStates.append(0)
        #randomStates.append(33)
        #randomStates.append(42)
        randomStates.append(66)
        randomStates.append(99)
        randomStates.append(102)
        randomStates.append(113)
        randomStates.append(143)
        for state in range(len(randomStates)):
            randomseed = randomStates[state]
            smpResults = []
            metrics = []
            timeBegin = time.time()
            print("Processing random state", randomseed)
            # filewritena       me = input("Enter file name: \n")
            # ileoutput = open(filewritename, 'w')
            # std_Dev_Summ = ({'model': 'WarPATH', 'MAE': 0, 'PW20': 0})
            # variance_Summ = ({'model': 'WarPATH', 'MAE': 0, 'PW20': 0})
            std_Dev_Summ = ({'model': 'IWPC', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0},
                            {'model': 'WarPATH', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0},
                            {'model': 'Gage', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0},
                            {'model': 'Fixed', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0})

            variance_Summ = ({'model': 'IWPC', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0},
                             {'model': 'WarPATH', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0},
                             {'model': 'Gage', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0},
                             {'model': 'Fixed', 'MAE': 0, 'PW20': 0, 'R2': 0, 'MALAR': 0, 'MLAR': 0})

            number_of_samples = 1000
            bootresults = []
            std_Dev = []
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
            impNumber = 50  # was 50 as was maximp
            maxImp = 50
            runImp = 0
            # randomseed = 0
            flagHIV = False
            # 99_42 143 33 113 102 0 66`6
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
                dfIWPC = pd.DataFrame()
                fileindex = filesImp.index(file)
                df = fileindex + 1
                suffix = str(df).zfill(3)
                if combinedata == True:
                    rootIWPC = root.replace("WarImputations", "MICESTATSMODELHIV\\")
                    IWPC_csv = rootIWPC + filesIWPC[fileindex]
                    IWPCDF = pd.read_csv(IWPC_csv, ';')
                    IWPCDF.reset_index()
                    sampleSize = int(round(trainSize))
                    dfIWPC, testset = train_test_split(IWPCDF, test_size=0.1, train_size=sampleSize,
                                                       random_state=randomseed)
                    dfIWPC["Status"] = "train"
                    dropColumn("IWPC", "Unnamed: 0", dfIWPC.columns, dfmod, dfIWPC)
                    dfIWPC.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Claire_IWPC" + randstring + suffix + ".csv",
                                  ";")
                    # dfIWPC.drop(["Unnamed: 0"], axis=1, inplace=True)
                # df = fileindex + 1
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
                dfmod.drop(["Inducer_status", "Amiodarone_status", "Smoking_status", "Indication"], axis=1,
                           inplace=True)
                dfmod["AgeYears"] = dfmod["Age_years"]
                dfmod['AgeYears'] = np.where((dfmod['AgeYears'] <= 18), 18, dfmod['AgeYears'])
                if flagHIV == True:
                    dfmod["HIVPositive"] = np.where(dfmod["HIV_status"] == "Positive", 1, 0)
                    dfmod["HIVUnknown"] = np.where(dfmod["HIV_status"] == "Unknown", 1, 0)
                dfmod["Status"] = ""
                if combinedata == True:
                    dfIWPC['AgeYears'] = np.where((dfIWPC['AgeYears'] <= 18), 18, dfIWPC['AgeYears'])
                    dropColumn("IWPC", "AgeDecades", dfIWPC.columns, dfmod, dfIWPC)
                    dropColumn("IWPC", "INR_Three", dfIWPC.columns, dfmod, dfIWPC)
                    if flagHIV == True:
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
                    # print("On imputation ", df, " and random seed ", randomseed)
                    dfIWPC.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Claire_IWPC_Formatted" + ".csv", ";")
                    data = dfmod
                    data.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Claire_Warpath" + ".csv", ";")
                    # data.index = data.index + 1
                    # print(data.shape)
                    data['Dose_mg_week'] = data['Dose_mg_week'].apply(np.sqrt)
                    impResults = []
                    models = []
                    boot = 0
                    samples = []
                    # metrics = []
                    # smpResults = []

                    if (find(models, 'model', 'WarPATH') == -1):
                        models.append({'model': 'WarPATH', 'MAE': 0, 'PW20': 0})

                    if combinedata == True:
                        dfIWPC['Dose_mg_week'] = dfIWPC['Dose_mg_week'].apply(np.sqrt)
                    # estimates = []
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

                # estimates = []
                # LR = LinearRegression()
                # estimates.append(Estimator(LR, 'LR'))
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
                        # RR =  Ridge()
                        # LAS= Lasso()
                        # ELNET = ElasticNet()
                        # estimates.append(Estimator(LAS,'LASSO'))
                        # estimates.append(Estimator(ELNET,'ELNET'))
                        # estimates.append(Estimator(RR,'RIDGE'))
                        # KNNR = KNeighborsRegressor()
                        # estimates.append(Estimator(KNNR, 'KNN'))
                        # svr = sklearn.svm.SVR()
                        # estimates.append(Estimator(svr,'SVREG'))
                        # MLPR = MLPRegressor()
                        # estimates.append(Estimator(MLPR,'MLPR'))
                        # RF = RandomForestRegressor()
                        # estimates.append(Estimator(RF, 'RF'))
                        #DTR = DecisionTreeRegressor()
                        #estimates.append(Estimator(DTR,'DTR'))
                        # LGB = lgb.LGBMRegressor()
                        # estimates.append(Estimator(LGB,'LGB'))
                        if "Unnamed: 0.1" in x_train.columns:
                            x_train.drop(["Unnamed: 0.1"], axis=1, inplace=True)
                        if "Unnamed: 0.1" in x_test.columns:
                            x_test.drop(["Unnamed: 0.1"], axis=1, inplace=True)
                        dfoptuna=pd.DataFrame()
                        for _, est in enumerate(estimates):
                            resultsdict, dfOptuna = traineval(est, x_train, y_train, x_test, y_test, squaring=squaring, df=df,
                                                    randomseed=randomseed)
                            alg = est.identifier
                            x_train.to_csv(
                                r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\training\\" + alg.replace(" ", "") + str(
                                    df) + ".csv", ";")
                            res_dict = {
                                'Estimator': [est.identifier for x in range(len(resultsdict['PW20']))],
                                'PW20': resultsdict['PW20'],
                                'MAE': resultsdict['MAE'],
                                'R2': resultsdict['R2'],
                                'Time': resultsdict['Time'],
                                'Alg': resultsdict['Alg']}

                            data["WarPATH_MAE"] = ExitSquareBracket(resultsdict['MAE'], True)
                            data["WarPATH_PW20"] = ExitSquareBracket(resultsdict['PW20'], True)
                            ml_model = resultsdict['Alg']

                            #dfOptuna = pd.read_csv(
                            #    r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\OptunaAllModelW\model_" + alg + "_" + str(
                            #        randomseed) + "_" + suffix + ".csv", ";")
                            scaler = MinMaxScaler()
                            dfOptuna['MAE'] = 0
                            dfOptuna['PW20'] = 0
                            listminmaxScaler = ['KNNR', 'MLPR', 'LASSO', 'ELNET', 'RIDGE', 'SVREG']
                            scaling = False
                            if alg in listminmaxScaler:
                              scaling = True
                            for index, row in dfOptuna.iterrows():
                               if alg == 'KNNR':
                                 n_neighbors = row['params_n_neighbors']
                                 metric = row['params_metric']
                                 weights = row['params_weights']
                                 model = KNeighborsRegressor(n_neighbors=n_neighbors, metric=metric,weights=weights)
                               elif alg == "MLPR":
                                 pactivation = row['params_activation']
                                 palpha = row['params_alpha']
                                 pbeta1 = row['params_beta_1']
                                 pbeta2 = row['params_beta_2']
                                 pepsilon = row['params_epsilon']
                                 #phiddenlayers = tupleTransform(row['params_hidden_layer_sizes'])
                                 phiddenlayers = row['params_hidden_layer_sizes']
                                 plearningrateinit = row['params_learning_rate_init']
                                 pmaxfun = row['params_max_fun']
                                 pmaxiter = row['params_max_iter']
                                 pmomentum = row['params_momentum']
                                 ppowert = row['params_power_t']
                                 psolver = row['params_solver']
                                 ptol = row['params_tol']
                                 pvalidfraction = row['params_validation_fraction']
                                 model = MLPRegressor(activation=pactivation, solver=psolver, alpha=palpha, beta_1=pbeta1, beta_2 = pbeta2,
                                                      epsilon=pepsilon, hidden_layer_sizes=phiddenlayers, learning_rate_init=plearningrateinit,
                                                      max_fun=pmaxfun, max_iter=pmaxiter, momentum=pmomentum, power_t=ppowert, tol=ptol,
                                                      validation_fraction=pvalidfraction)

                               elif alg == 'SVREG':
                                 C = row['params_C']
                                 epsilon = row['params_epsilon']
                                 gamma = row['params_gamma']
                                 kernel = row['params_kernel']
                                 model = sklearn.svm.SVR(gamma=gamma, C=C, epsilon=epsilon, kernel=kernel)
                               elif alg == "RF" or alg == "GBR" :
                                 maxdepth = row['params_max_depth']
                                 minsampleaf = row['params_min_samples_leaf']
                                 minsampsplit = row['params_min_samples_split']
                                 numberestimators = row['params_n_estimators']
                                 if alg == "RF":
                                     maxdepth = row['params_max_depth']
                                     minsampleaf = row['params_min_samples_leaf']
                                     minsampsplit = row['params_min_samples_split']
                                     numberestimators = row['params_n_estimators']
                                     minimpurity = row['params_min_impurity_decrease']
                                     model = RandomForestRegressor(max_depth = maxdepth, min_samples_leaf = minsampleaf,
                                             min_samples_split = minsampsplit, n_estimators = numberestimators,
                                             min_impurity_decrease=minimpurity)
                                 elif alg == "GBR":
                                    model = GradientBoostingRegressor(max_depth=maxdepth, min_samples_leaf=minsampleaf,
                                                                      min_samples_split=minsampsplit, n_estimators=numberestimators)
                               elif alg == "XGBR":
                                   parmboost = row['params_booster']
                                   parmsamplevel= row['params_colsample_bylevel']
                                   parmsampnode = row['params_colsample_bynode']
                                   parmsamptree = row['params_colsample_bytree']
                                   parmlearnrate = row['params_learning_rate']
                                   parmmaxdepth = row['params_max_depth']
                                   parmminchildweight = row['params_min_child_weight']
                                   parmnestimators = row['params_n_estimators']
                                   parmsubsample = row['params_subsample']
                                   model = XGBRegressor(booster = parmboost, colsample_bylevel= parmsamplevel,
                                   colsample_bynode=parmsampnode,colsample_bytree=parmsamptree,learning_rate= parmlearnrate,
                                   max_depth=parmmaxdepth,min_child_weight=parmminchildweight, n_estimators=parmnestimators,
                                   subsample = parmsubsample)
                               elif alg == "DTR":
                                   maxdepth = row['params_max_depth']
                                   minsampleaf = row['params_min_samples_leaf']
                                   minsampsplit = row['params_min_samples_split']
                                   minimpurity = row['params_min_impurity_decrease']
                                   splitter = row['params_splitter']
                                   #maxleafnodes = row['params_max_leaf_nodes'].fillna(0)
                                   #maxleafnodes = int(maxleafnodes)
                                   minweightfractionleaf = row['params_min_weight_fraction_leaf']
                                   #model = DecisionTreeRegressor(max_depth=maxdepth, min_samples_leaf=minsampleaf,
                                   #                              min_samples_split=minsampsplit,
                                   #                              max_leaf_nodes=maxleafnodes,
                                   #                              min_weight_fraction_leaf=minweightfractionleaf,
                                   #                              min_impurity_decrease=minimpurity, splitter=splitter)
                                   model=DecisionTreeRegressor(max_depth=maxdepth, min_samples_leaf=minsampleaf,
                                                                 min_samples_split=minsampsplit,splitter=splitter,
                                                                 min_weight_fraction_leaf=minweightfractionleaf,
                                                                 min_impurity_decrease=minimpurity)

                               elif alg == "LR":
                                  model = LinearRegression()
                               elif alg == "LASSO" or alg == "RIDGE" or alg == "ELNET":
                                 alpha = row['params_alpha']
                                 maxiter = row['params_max_iter']
                                 if alg == "ELNET":
                                    l1ratio = row['params_l1_ratio']
                                 if alg ==  "LASSO":
                                    model = Lasso(alpha= alpha, max_iter= maxiter)
                                 elif alg == "RIDGE":
                                    model = Ridge(alpha = alpha, max_iter = maxiter)
                                 else:
                                    model = ElasticNet(alpha=alpha, l1_ratio=l1ratio, max_iter= maxiter)
                               if scaling == True:
                                  pipeline = make_pipeline(scaler, model)
                                  fitted = pipeline.fit(x_train, y_train)
                               else:
                                 fitted = model.fit(x_train,y_train)
                               pred = fitted.predict(x_test)
                               if squaring:
                                  predicted = np.square(pred)
                                  observed = np.square(y_test)
                               MAEvalue = mean_absolute_error(observed, predicted)
                               PW20value = PercIn20(observed, predicted)
                               dfOptuna.at[index,'MAE'] = MAEvalue
                               dfOptuna.at[index,'PW20'] = PW20value
                            dfOptuna.to_csv(

                            r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\OptunaAllmodelWIWPC\model_" + alg + "_" + str(randomseed) + "_" + suffix + ".csv",";")
                            if resultsdict['MAE'] > [5]:
                               results.append(res_dict)

                        df_res = pd.DataFrame()
                        for res in results:
                            df_res = df_res.append(pd.DataFrame.from_dict(res))
                        print(f"\n\n{df_res.groupby(['Estimator']).agg(np.mean)}\n")

            # dfResults = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfResults" + ".csv", ";")

            dfResults = pd.DataFrame(results)
            dfResults["PW20"] = dfResults.apply(lambda x: ExitSquareBracket(x["PW20"], False), axis=1).astype(float)
            dfResults["MAE"] = dfResults.apply(lambda x: ExitSquareBracket(x["MAE"], False), axis=1).astype(float)
            dfResults["R2"] = dfResults.apply(lambda x: ExitSquareBracket(x["R2"], False), axis=1).astype(float)
            dfResults["Time"] = dfResults.apply(lambda x: ExitSquareBracket(x["Time"], False), axis=1).astype(float)
            dfResults["Estimator"] = dfResults.apply(lambda x: ExitSquareBracket(x["Estimator"], False), axis=1)

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
            dfResults.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATHIWPC_dfResults_" + alg + "_" + str(randomseed) + ".csv",";")
            dfSummary.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATHIWPC_dfSummary_" + alg + "_" + str(randomseed) + ".csv",";")
            print("STOP HERE")

            if False and df == impNumber:
                for i in range(len(metric_columns)):
                    current_metric = metric_columns[i]
                    metriclist = np.array(collect_Metrics(smpResults, 'WarPATH', current_metric))
                    std = np.square(std_deviation(metriclist))
                    var = variance(metriclist)
                    # std = np.sqrt(var)
                    std_Dev.append({'model': 'WarPATH', 'metric': current_metric, 'SD': std, 'VAR': var})

                for k in range(len(std_Dev)):
                    model = std_Dev[k]['model']
                    metric = std_Dev[k]['metric']
                    Wfactor = std_Dev[k]['SD']
                    Bfactor = std_Dev[k]['VAR']
                    modelpos = find(std_Dev_Summ, 'model', model)
                    std_Dev_Summ[modelpos][metric] += Wfactor
                    modelpos2 = find(variance_Summ, 'model', model)
                    variance_Summ[modelpos2][metric] += Bfactor

                for k in range(len(models)):
                    fieldname = models[k]['model']
                    for m in range(len(smpResults)):
                        if smpResults[m]['model'] == fieldname:
                            models[k]['MAE'] += smpResults[m]['MAE']
                            models[k]['PW20'] += smpResults[m]['PW20']
                Bfactor = (impNumber + 1) / impNumber

                for k in range(len(models)):
                    fieldname = models[k]['model']
                    mae_list = collect_Results(smpResults, fieldname, 'MAE')
                    mae_value = listAverage(mae_list)
                    mae_variance = variance(mae_list) * Bfactor
                    stdpos = find(std_Dev_Summ, 'model', fieldname)
                    varpos = find(variance_Summ, 'model', fieldname)
                    mae_var = variance_Summ[varpos]['MAE'] * 2
                    mae_std_dev = std_Dev_Summ[stdpos]['MAE'] / impNumber
                    mae_CI_minus = round(mae_value - 1.96 * np.sqrt(mae_std_dev + mae_variance), 4)
                    mae_CI_plus = round(mae_value + 1.96 * np.sqrt(mae_std_dev + mae_variance), 4)
                    pw20_list = collect_Results(smpResults, fieldname, 'PW20')
                    pw20_value = listAverage(pw20_list)
                    pw20_variance = variance(pw20_list) * Bfactor
                    pw20_std_dev = std_Dev_Summ[stdpos]['PW20'] / impNumber
                    pw20_var = variance_Summ[varpos]['MAE'] * 2
                    pw20_CI_minus = round(pw20_value - 1.96 * np.sqrt(pw20_std_dev + pw20_variance), 4)
                    pw20_CI_plus = round(pw20_value + 1.96 * np.sqrt(pw20_std_dev + pw20_variance), 4)
                    dfPrev = dfConf
                    timeConclude = time.time()
                    timePeriod = timeConclude - timeBegin
                    dfCurr = pd.DataFrame()
                    dfCurr['HEADER'] = ['WARPATH', alg, randomseed]
                    dfCurr["MAE"] = [round(mae_value, 6), mae_CI_minus, mae_CI_plus]
                    dfCurr["MAE_Rubin"] = [0, round(mae_std_dev, 6), round(mae_variance, 6)]
                    dfCurr["PW2O"] = [round(pw20_value, 6), pw20_CI_minus, pw20_CI_plus]
                    dfCurr["PW20_Rubin"] = [0, round(pw20_std_dev, 6), round(pw20_variance, 6)]
                    dfCurr["TIME"] = timePeriod
                    dfCurr["ML_MODEL"] = ['', ml_model, '']
                    frames = (dfPrev, dfCurr)
                    dfConf = pd.concat(frames)
                    name = "WARPATH" + alg.replace(" ", "")
                    dfConf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\CI\\" + name + ".csv", ";")
                    print(fieldname, 'MAE:', round(mae_value, 6), "StdDev:", round(mae_std_dev, 6), "B: ",
                          round(mae_variance, 4), "  CI: [", mae_CI_minus, mae_CI_plus, "]", )
                    print(fieldname, 'PW20:', round(pw20_value, 6), "StdDev:", round(pw20_std_dev, 6), "B: ",
                          round(pw20_variance, 4), " CI: [", pw20_CI_minus, pw20_CI_plus, "]")
                    print('RANDOM STATE,   ', randomseed)
                    print(dfSummary)


if __name__ == "__main__":
    main()

