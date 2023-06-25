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
from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
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


def collect_Metrics(metrics, model, metric):
    container = []
    for i in range(len(metrics)):
        if (metrics[i]['model'] == model) & (metrics[i]['metric'] == metric):
            # print(metrics[i]['model'], metrics[i]['metric'], metrics[i]['value'])
            container.append(metrics[i]['value'])
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

def confintlimit95(metric,mean):
    return 1.96 * np.sqrt(variance(metric,mean) / len(metric))

def TrainOrTest(patientID,TrainList, TestList):
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
            #lo2 = mean - confintlimit95(data,mean)
            #hi2 = mean + confintlimit95(data,mean)
            lo2 = mean
            hi2 = mean
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


def tune(objective,df,model,randomseed):
    ntrials = 50
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=ntrials)
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
            r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\OPTUNAHYPERPARAMETERS\model_" + model + "_" + str(randomseed) + "_"+ suffixpre + ".csv", ";")
        if "Unnamed: 0" in dfHyper.columns:
            dfHyper.drop(["Unnamed: 0"], axis=1, inplace=True)
        if "Unnamed: 0.1" in dfHyper.columns:
            dfHyper.drop(["Unnamed: 0.1"], axis=1, inplace=True)
    frames = (dfHyper, dfHyperCurrent)
    dfHyper = pd.concat(frames)
    dfHyper.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\OPTUNAHYPERPARAMETERS\model_" + model + "_" + str(randomseed) + "_"+  suffix + ".csv",";")
    print(f"Best score: {best_score} \nOptimized parameters: {params}")
    return params

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
    resultsdict = {'PW20': 0, 'MAE': 0, 'R2': 0, 'Time': ''}
    ml_learner = est.identifier
    #RANDOM_SEED = 66
    RANDOM_SEED = randomseed

    # 10-fold CV
    kfolds = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    model = est.estimator
    gridFit = True
    print(f'\n{est.identifier}...')
    modelID = est.identifier
    if est.identifier != "LR":  # tuning is not required for LR
        if est.identifier == "XGBR":
            start=time.time()
            def XGBR_Objective(trial):
               _booster = trial.suggest_categorical('booster',["gbtree"])
               _max_depth = trial.suggest_int('max_depth',1,4,step=1)
               _colsample_bynode = trial.suggest_categorical('colsample_bynode',[1])
               _learning_rate = trial.suggest_float('learning_rate',0.01, 0.05, step=0.01)
               _n_estimators = trial.suggest_int('n_estimators',400,800,step = 200)
               _min_child_weight = trial.suggest_int('min_child_weight',21,51,step=10)
               _subsample = trial.suggest_categorical('subsample',[0.877056, 0.957056, 0.677056, 0.757056, 0.917056])
               _colsample_bytree =trial.suggest_categorical('colsample_bytree',[0.530084,0.770084, 0.790084, 0.910084, 0.610084])
               _colsample_bylevel=trial.suggest_categorical('colsample_bylevel',[0.66163511,0.74163511, 0.94163511,0.62163511, 0.90163511])
               #_gamma = trial.suggest_float('gamma',0.000,3.308368474,step=0.002)
               XGBR_model = XGBRegressor(max_depth = _max_depth, min_child_weight=_min_child_weight,
                                         colsample_bytree=_colsample_bytree,colsample_bylevel=_colsample_bylevel,
                                         colsample_bynode= _colsample_bynode,subsample=_subsample, n_estimators=_n_estimators)
               score = cross_val_score(XGBR_model, xtrain, ytrain, cv=kfolds, scoring="neg_mean_absolute_error").mean()
               return score

            XGBR_params = tune(XGBR_Objective, df, est.identifier, randomseed)
            end = time.time()
            model = XGBRegressor(**XGBR_params)

        if est.identifier == "RF":
            start = time.time()

            def RF_Objective(trial):
                 #_min_samples_leaf = trial.suggest_categorical('min_samples_leaf', [2,3,4,5,6,7,8,9,10,11,12])
                 #_min_samples_leaf = trial.suggest_categorical('min_samples_leaf',[8,9,11,12])
                 _min_samples_leaf = trial.suggest_categorical('min_samples_leaf', [9, 11])
                 _min_impurity_decrease = trial.suggest_categorical('min_impurity_decrease', [0.0])
                 #_max_depth = trial.suggest_categorical('max_depth', [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
                 #_max_depth = trial.suggest_categorical('max_depth', [4,6,10])
                 _max_depth = trial.suggest_categorical('max_depth', [4])
                 #_n_estimators = trial.suggest_categorical('n_estimators', [50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200])
                 _n_estimators = trial.suggest_categorical('n_estimators',[60,80,110])
                 #_min_samples_split = trial.suggest_categorical("min_samples_split", [2, 3, 4, 5, 6, 7, 8, 9, 10])
                 #_min_samples_split = trial.suggest_categorical("min_samples_split",[4,5,6,7,8,9,10])
                 _min_samples_split = trial.suggest_categorical("min_samples_split", [5, 6, 7])
                 #_max_features=trial.suggest_float("max_features",0.0,50.0,step=5.0)
                 #_n_estimators = trial.suggest_categorical("n_estimators", [50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200])
                 #_max_depth = trial.suggest_categorical("max_depth", [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
                 #_min_samp_split = trial.suggest_categorical("min_samples_split", [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
                 #_min_samples_leaf = trial.suggest_categorical("min_samples_leaf", [2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0])
                 #_max_features = trial.suggest_categorical("max_features", [10,15,20,25,30,35,40,45,50])
                 RF_model = RandomForestRegressor(
                     max_depth=_max_depth,
                     min_samples_split=_min_samples_split,
                     min_impurity_decrease=_min_impurity_decrease,
                     min_samples_leaf=_min_samples_leaf,
                     n_estimators=_n_estimators

                 )

                 score = cross_val_score(RF_model, xtrain, ytrain, cv=kfolds, scoring="neg_mean_absolute_error").mean()
                 return score

            RF_params = tune(RF_Objective, df, est.identifier, randomseed)
            end = time.time()
            model = RandomForestRegressor(**RF_params)

            if False:
               _n_estimators = trial.suggest_int("n_estimators", 50, 200, step=50)
               _max_depth = trial.suggest_int("max_depth", 5, 20, step=5)
               _min_samp_split = trial.suggest_int("min_samples_split", 2, 10, step=2)
               _min_samples_leaf = trial.suggest_int("min_samples_leaf", 2 , 10, step=2)
               _max_features = trial.suggest_int("max_features", 10,50, step=10)
               rf = RandomForestRegressor(
                    max_depth=_max_depth,
                    min_samples_split=_min_samp_split,
                    min_samples_leaf=_min_samples_leaf,
                    max_features=_max_features,
                    n_estimators=_n_estimators,
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                 )
               score = cross_val_score(rf, xtrain, ytrain, cv=kfolds, scoring="neg_mean_absolute_error").mean()
               return score
        else:
            if est.identifier == 'DTR':
                # define parameter space
                start = time.time()

                def DTR_objective(trial):
                    _min_samples_leaf = trial.suggest_categorical("min_samples_leaf", [1])
                    _max_depth = trial.suggest_categorical('max_depth', [2])
                    _min_impurity_decrease = trial.suggest_categorical("min_impurity_decrease", [0.0])
                    _max_features = trial.suggest_categorical('max_features', [0.355298191, 0.55247903, 0.5616223889, 0.900168726])
                    _min_samples_split = trial.suggest_categorical("min_samples_split", [2])
                    _max_leaf_nodes = trial.suggest_categorical("max_leaf_nodes", [None,10])
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

                DTR_params = tune(DTR_objective, df, modelID, randomseed)
                end = time.time()
                model = DecisionTreeRegressor(**DTR_params)

                if False:
                    kcv = KFold(n_splits=5, shuffle=True, random_state=66)
                    gridFit = False
                    param_grid = {'max_depth': [None, 2, 3, 4, 6, 8, 10],
                                  'min_samples_leaf': [1, 2, 4, 6, 8, 10, 20, 30],
                                  'max_features': ['auto', 0.95, 0.90, 0.85, 0.80, 0.75, 0.70],
                                  'min_weight_fraction_leaf': [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.05],
                                  'splitter': ['random', 'best'],
                                  'min_impurity_decrease': [0.0, 0.0005, 0.005, 0.05, 0.10, 0.15, 0.2],
                                  'min_samples_split': [2, 3, 4, 5, 6, 8, 10],
                                  'max_leaf_nodes': [10, 15, 20, 25, 30, 35, 40, 45, 50, None]
                                   }
            elif est.identifier == "GBR":
                start = time.time()
                def GBR_Objective(trial):
                    #_learning_rate = trial.suggest_categorical("learning_rate",[0.159470245, 0.034293879, 0.055418218, 0.030973395, 0.005561334, 0.132329369, 0.03803106, 0.072507733, 0.032894225,
                    #                                                           0.0743424267, 0.023528704, 0.003805887, 0.009816521, 0.067644981, 0.075711757, 0.062691453, 0.096134727])
                    _learning_rate = trial.suggest_categorical("learning_rate", [0.023528704,0.067644981,0.030973395, 0.003805887,0.132329369, 0.03803106, 0.072507733,0.009816521])

                    _max_depth = trial.suggest_categorical("max_depth",[2,3,4,5])
                    #_n_estimators = trial.suggest_categorical("n_estimators",[229,196,119,397,543,22,45,59,955,70,502,800,494,59,645,108,25])
                    _n_estimators = trial.suggest_categorical("n_estimators",[25,45,59,119,543])
                    _min_samples_leaf = trial.suggest_categorical("min_samples_leaf",[15,21,23])
                    _min_samples_split = trial.suggest_categorical("min_samples_split",[2,3])
                    GBR_model = GradientBoostingRegressor(learning_rate=_learning_rate, max_depth=_max_depth,n_estimators=_n_estimators,
                                                          min_samples_leaf=_min_samples_leaf,min_samples_split=_min_samples_split)
                    score = cross_val_score(GBR_model, xtrain, ytrain, cv=kfolds,
                                            scoring="neg_mean_absolute_error").mean()
                    return score

                GBR_params = tune(GBR_Objective, df, modelID, randomseed)
                end = time.time()
                model = GradientBoostingRegressor(**GBR_params)

            elif est.identifier == "MLPR":


                #activation = 'identity', alpha = 0.009260944818691528,
                #beta_1 = 0.8304148442169565, beta_2 = 0.9847593650340831,
                #epsilon = 4.968151316490382e-06, learning_rate = 'adaptive',
                #learning_rate_init = 0.014389028832229495, max_fun = 27459,
                #max_iter = 240, momentum = 0.871153667593362,
                #power_t = 0.6130708398995247, random_state = 2, solver = 'lbfgs',
                #tol = 0.008075623520655316,
                #validation_fraction = 0.036450106641084085
                start = time.time()
                def MLPR_Objective(trial):
                    scaler = MinMaxScaler()
                    _mlpr_hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes",
                                                                       [(2,),(3,), (1,),(3,1,),(3,2,),(3,3,), (2, 1,),(2, 2,)])

                    #_mlpr_hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes",[(2,),(4,),(8,),(4,1,),(4,2,),(4,4,),(2,1,),(2,2,),(8,2,),(8,4,),(8,1,),(8,6,)])
                    # _mlpr_hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes",[(20,),(10,),(11,),(12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,
                   #_mlpr_learning_rate_init = trial.suggest_categorical("learning_rate_init",[ 0.0001, 0.0015,  0.002]) # WAS [0.001]
                    _mlpr_max_iter = trial.suggest_categorical("max_iter",[240,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000]) # WAS [1000,1500,2000]
                    #_mlpr_max_iter = trial.suggest_categorical("max_iter", [4500, 5000, 5500, 6000, 6500, 7000, 7500,8000])  # WAS [1000,1500,2000]
                    _mlpr_solver = trial.suggest_categorical("solver",['lbfgs'])
                    _mlpr_tol = trial.suggest_categorical("tol",[0.008075623520655316])
                    #_mlpr_activation = trial.suggest_categorical('activation',['logistic','relu'])
                    _mlpr_learning_rate_init = trial.suggest_categorical('learning_rate_init', [0.014389028832229495])
                    _mlpr_epsilon = trial.suggest_categorical('epsilon',[4.968151316490382e-06])
                    _mlpr_activation = trial.suggest_categorical('activation',['identity'])
                    _mlpr_alpha = trial.suggest_categorical('alpha',[0.009260944818691528])
                    _mlpr_beta_1 = trial.suggest_categorical('beta_1', [0.8304148442169565])
                    _mlpr_beta_2 = trial.suggest_categorical('beta_2', [0.9847593650340831])
                    _mlpr_power_t = trial.suggest_categorical('power_t', [0.6130708398995247])
                    _mlpr_max_fun = trial.suggest_categorical('max_fun', [27459])
                    _mlpr_momentum = trial.suggest_categorical('momentum',[0.871153667593362])
                    _mlpr_validation_fraction = trial.suggest_categorical('validation_fraction',[0.036450106641084085])
                    #_mlpr_learning_rate = trial.suggest_categorical("learning_rate",['adaptive'])
                    #_mlpr_activation = trial.suggest_categorical("activation",["relu"])
                    #MLPR_Model = MLPRegressor(hidden_layer_sizes = _mlpr_hidden_layer_sizes, learning_rate_init=_mlpr_learning_rate_init,
                    #                          max_iter = _mlpr_max_iter, learning_rate = _mlpr_learning_rate, activation=_mlpr_activation,
                    #                          solver=_mlpr_solver)
                    MLPR_Model = MLPRegressor(hidden_layer_sizes = _mlpr_hidden_layer_sizes,
                                              max_iter=_mlpr_max_iter,
                                              activation=_mlpr_activation,
                                              solver=_mlpr_solver,random_state = randomseed,
                                              tol=_mlpr_tol,learning_rate_init=_mlpr_learning_rate_init,
                                              epsilon=_mlpr_epsilon, alpha = _mlpr_alpha,
                                              beta_1= _mlpr_beta_1, beta_2 = _mlpr_beta_2,
                                              power_t = _mlpr_power_t,max_fun=_mlpr_max_fun,
                                              momentum= _mlpr_momentum, validation_fraction= _mlpr_validation_fraction)
                    pipeline = make_pipeline(scaler,MLPR_Model )
                    score = cross_val_score(pipeline, xtrain, ytrain, cv=kfolds, scoring="neg_mean_absolute_error").mean()
                    return score
                MLPR_params = tune(MLPR_Objective, df, modelID, randomseed)
                end = time.time()
                model = MLPRegressor(**MLPR_params)


            elif est.identifier == "KNNR":
                start = time.time()

                def KNN_Objective(trial):
                    scaler = MinMaxScaler()
                    _knn_n_neighbors = trial.suggest_int("n_neighbors", 2, 200, step=1)
                    _knn_metric = trial.suggest_categorical("metric", ['euclidean', 'manhattan', 'minkowski'])
                    _knn_weights = trial.suggest_categorical("weights", ['uniform', 'distance'])
                    KNN_Model = KNeighborsRegressor(n_neighbors=_knn_n_neighbors,metric=_knn_metric, weights=_knn_weights)
                    pipeline = make_pipeline(scaler, KNN_Model)
                    score = cross_val_score(pipeline, xtrain, ytrain, cv=kfolds, scoring="neg_mean_absolute_error").mean()
                    return score

                KNN_params = tune(KNN_Objective, df, modelID, randomseed)
                end = time.time()
                model = KNeighborsRegressor(**KNN_params)
            else:
                if est.identifier in ("LASSO"):
                    start = time.time()
                    def LASSO_Objective(trial):
                        scaler = MinMaxScaler()
                        # _alpha = trial.suggest_float('alpha',0,0.05,step=0.001)
                        _alpha = trial.suggest_categorical("alpha", [0.006644814, 0.010214848, 0.000233117, 0.012706666, 0.009199882, 0.000377741, 0.00075485, 0.0000992077, 0.000528944])
                        _max_iter = trial.suggest_categorical("max_iter", [828, 1181, 1073, 897, 1120, 1156, 880, 1142, 1140])
                        lasso_model = Lasso(alpha=_alpha, max_iter=_max_iter)
                        pipeline = make_pipeline(scaler, lasso_model)
                        score = cross_val_score(pipeline, xtrain, ytrain, cv=kfolds, scoring="neg_mean_absolute_error").mean()
                        return score
                    LASSO_params = tune(LASSO_Objective, df, modelID, randomseed)
                    end = time.time()
                    model = Lasso(**LASSO_params)
                elif est.identifier == "RIDGE":
                    start = time.time()
                    def RIDGE_Objective(trial):
                        _alpha = trial.suggest_categorical("alpha", [2.236514,0.060708, 0.204956,0.133253, 0.512693, 1.269382, 0.969999, 1.060682, 0.20964, 2.75604])
                        #_alpha = trial.suggest_float("alpha",13,16,step=0.5)
                        _max_iter = trial.suggest_categorical("max_iter", [1071,858, 1108, 1038, 930, 1220, 1214, 999, 1035,912])
                        ridge_model = Ridge(alpha=_alpha, max_iter=_max_iter)
                        scaler = MinMaxScaler()
                        pipeline = make_pipeline(scaler, ridge_model)
                        score = cross_val_score(pipeline, xtrain, ytrain, cv=kfolds,scoring="neg_mean_absolute_error").mean()
                        return score

                    RIDGE_params = tune(RIDGE_Objective, df, modelID, randomseed)
                    end = time.time()
                    model = Ridge(**RIDGE_params)

                elif est.identifier == "ELNET":
                    start = time.time()

                    def ELNET_Objective(trial):
                        scaler = MinMaxScaler()
                        # _alpha = trial.suggest_float('alpha', 0,0.05,step=0.001)
                        # _l1_ratio = trial.suggest_float('l1_ratio', 0.01, 0.99, step=0.01)
                        _alpha = trial.suggest_categorical("alpha", [0.000198094, 0.015176516, 0.0, 0.003264721, 0.00066628, 0.001536277, 0.0004508042, 0.002504531 ])
                        _l1_ratio = trial.suggest_categorical("l1_ratio", [0.023118993, 0.448966271, 0.664744323, 0.34692344, 0.550401256, 0.245397394, 0.4617206, 0.312367882, 0.236122903, 0.094718953, 0.631230102 ])
                        _max_iter = trial.suggest_categorical("max_iter",[1159, 818, 1000, 1129, 887, 1036, 876, 1149, 970, 957])
                        ELNET_model = ElasticNet(alpha=0.01, l1_ratio=_l1_ratio, max_iter=_max_iter)
                        pipeline = make_pipeline(scaler, ELNET_model)
                        score = cross_val_score(pipeline, xtrain, ytrain, cv=kfolds,
                                                scoring="neg_mean_absolute_error").mean()
                        return score

                    ELNET_params = tune(ELNET_Objective, df, modelID, randomseed)
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
                        #_gamma = trial.suggest_categorical('gamma', ['auto', 'scale'])
                        _regParam1 = trial.suggest_categorical('C', [1])
                        _regParam2 = trial.suggest_categorical('epsilon', [0.1])
                        _gamma = trial.suggest_categorical('gamma', ['scale'])
                        #_C = trial.suggest_categorical("C",  [0,0.1,0,2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0])
                        #_epsilon = trial.suggest_categorical("epsilon", [0,0.1,0,2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0])
                        _kernel = trial.suggest_categorical("kernel", ['rbf'])
                        #_coef0 = trial.suggest_float("coef0", 0.01, 1)
                        #svr = SVR(C=_C, epsilon=_epsilon, kernel='poly')
                        SVREG_model = sklearn.svm.SVR(gamma=_gamma, C=_regParam1, epsilon=_regParam2, kernel=_kernel)
                        pipeline = make_pipeline(scaler, SVREG_model)
                        score = cross_val_score(pipeline, xtrain, ytrain, cv=kfolds,scoring="neg_mean_absolute_error").mean()
                        return score

                    SVREG_params = tune(SVREG_Objective, df, modelID, randomseed)
                    end = time.time()
                    model = sklearn.svm.SVR(**SVREG_params)

                elif est.identifier == "ABRF":
                    param_grid = [{'n_estimators': [10, 50, 100, 500], 'learning_rate': [0.0001, 0.001, 0.01, 0.1]}]
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

                    GBR_params = tune(GBR_Objective, df, modelID, randomseed)
                    end = time.time()
                    model = GradientBoostingRegressor(**GBR_params)



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
                        cv=kfol0ds)
                    stack.fit(X, y)


    if est.identifier == "LR":
       start = time.time()
    fitted = model.fit(xtrain, ytrain)
    if est.identifier == "LR":
       end = time.time()
    timeElapsed = end - start
    ypred = fitted.predict(xtest)
    if squaring:
       ytest = np.square(ytest)
       ypred = np.square(ypred)
    PW20 = PercIn20(ytest, ypred)
    MAE = mean_absolute_error(ytest, ypred)
    randstring = "_" + str(randomseed) + "_"
    if ml_learner != "LR":
       suffix = str(df).zfill(3)
       dfHyper = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\OPTUNAHYPERPARAMETERS\model_" + ml_learner + randstring + suffix + ".csv",";")
       dfHyper.loc[df - 1, "mae"] = MAE
       dfHyper.loc[df - 1,"pw20"] = PW20
       dfHyper.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\OPTUNAHYPERPARAMETERS\model_" + ml_learner + randstring + suffix + ".csv", ";")
       dfHyper = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\OPTUNAHYPERPARAMETERS\model_" + ml_learner + randstring + suffix + ".csv", ";")
       if "Unnamed: 0" in dfHyper.columns:
         dfHyper.drop(["Unnamed: 0"], axis=1, inplace=True)
       if "Unnamed: 0.1" in dfHyper.columns:
         dfHyper.drop(["Unnamed: 0.1"], axis=1, inplace=True)
       dfHyper.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\OPTUNAHYPERPARAMETERS\model_" + ml_learner + randstring + suffix + ".csv",";")
    R2 = RSquared(ytest, ypred)
    resultsdict['PW20'] = [PW20]
    resultsdict['MAE'] = [MAE]
    resultsdict['R2'] = [R2]
    resultsdict['Time'] = [timeElapsed]
    return resultsdict

def main():
    mlmodels = []
    #KNNR = KNeighborsRegressor()
    RR = Ridge()
    LAS = Lasso()
    ELNET = ElasticNet()
    # svr = sklearn.svm.SVR()
    #GBR = GradientBoostingRegressor()
    #MLPR = MLPRegressor()
    #mlmodels.append(Estimator(MLPR, 'MLPR'))
    #mlmodels.append(Estimator(GBR,'GBR'))
    #mlmodels.append(Estimator(svr,'SVREG'))
    mlmodels.append(Estimator(LAS,'LASSO'))
    mlmodels.append(Estimator(ELNET, "ELNET"))
    mlmodels.append(Estimator(RR, "RIDGE"))
    #mlmodels.append(Estimator(KNNR, "KNNR"))
    for _, est in enumerate(mlmodels):
        dfConf = pd.DataFrame()
        estimates = []
        print("Processing ML model ", est.identifier)
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
            onlyIWPC = True
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
            flagHIV = False
            #99_42 143 33 113 102 0 66
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
                                runningImp = runningImp+1
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
               #indexTo1(file)
               #file.reset_index()
               dfnew = pd.read_csv(file, ";")
               dfmod = dfnew
               fileindex = filesImp.index(file)
               df=fileindex+1
               suffix = str(df).zfill(3)
               if combinedata == True:
                  rootIWPC = root.replace("WarImputations", "MICESTATSMODELHIV\\")
                  IWPC_csv = rootIWPC + filesIWPC[fileindex]
                  IWPCDF = pd.read_csv(IWPC_csv, ';')
                  IWPCDF.reset_index()
                  if onlyIWPC == True:
                    IWPCDF['AgeYears'] = np.where((IWPCDF['AgeYears'] <= 18), 18, IWPCDF['AgeYears'])
                    IWPCDF["Target_INR"] = IWPCDF.apply(lambda x: INRThree(x["Target_INR"]), axis=1)
                    IWPCDF["Target_INR"] = IWPCDF['Target_INR'].astype("float")
                    IWPCDF['Dose_mg_week'] = IWPCDF['Dose_mg_week'].apply(np.sqrt)
                    dropColumn("IWPCDF", "AgeDecades", IWPCDF.columns, dfmod, IWPCDF, IWPCDF)
                    dropColumn("IWPCDF", "INR_Three", IWPCDF.columns, dfmod, IWPCDF, IWPCDF)
                    #IWPCDF["HIVPositive"] = 0
                    #IWPCDF["HIVUnknown"] = 0
                    dfIWPConly, dfIWPC_test = train_test_split(IWPCDF, test_size=0.2, random_state=randomseed)
                    dfIWPConly.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Claire_IWPC" + randstring + suffix + ".csv",
                    ";")
                  else:
                    sampleSize = int(round(trainSize))
                    dfIWPC,testset=   train_test_split(IWPCDF, test_size=0.1, train_size=sampleSize, random_state=randomseed)
                    dfIWPC["Status"] = "train"
                    dropColumn("IWPC", "Unnamed: 0", dfIWPC.columns, dfmod, dfIWPC,IWPCDF)
                    dfIWPC.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Claire_IWPC" + randstring + suffix + ".csv", ";")
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
                      if onlyIWPC == False:
                          dfIWPC["Target_INR"] = dfIWPC.apply(lambda x: INRThree(x["Target_INR"]), axis=1)
                          dfIWPC["Target_INR"] = dfIWPC['Target_INR'].astype("float")
                  dfmod["Inducer"] = dfmod.apply(lambda x: ConvertYesNo(x["Inducer_status"]), axis=1)
                  dfmod["Amiodarone"] = dfmod.apply(lambda x: ConvertYesNo(x["Amiodarone_status"]), axis=1)
                  dfmod["Smoker"] = dfmod.apply(lambda x: ConvertYesNo(x["Smoking_status"]), axis=1)
                  dfmod["Indicationflag"] = dfmod.apply(lambda x: ConvertYesNo(x["Indication"]), axis=1)
                  dfmod.drop(["Inducer_status", "Amiodarone_status", "Smoking_status", "Indication"], axis=1, inplace=True)
                  dfmod["AgeYears"] = dfmod["Age_years"]
                  dfmod['AgeYears'] = np.where((dfmod['AgeYears'] <= 18), 18, dfmod['AgeYears'])
                  if flagHIV == True:
                     dfIWPC["HIVPositive"] = np.where(dfmod["HIV_status"] == "Positive", 1, 0)
                     dfIWPC["HIVUnknown"] = np.where(dfmod["HIV_status"] == "Unknown", 1, 0)
                  dfmod["Status"] = ""
                  if onlyIWPC == False:
                      dropColumn("WARPATH", 'HIV_status', dfmod.columns, dfmod, dfIWPC, IWPCDF)
                      dropColumn("WARPATH", "Unnamed: 0", dfmod.columns, dfmod, dfIWPC, IWPCDF)
                      dropColumn("WARPATH", "Unnamed: 0.1", dfmod.columns, dfmod, dfIWPC, IWPCDF)
                      dropColumn("WARPATH", "Unnamed: 0.2", dfmod.columns, dfmod, dfIWPC, IWPCDF)
                      dropColumn("WARPATH", "Age_years", dfmod.columns, dfmod, dfIWPC, IWPCDF)
                      dropColumn("WARPATH", ".imp", dfmod.columns, dfmod, dfIWPC, IWPCDF)
                      dropColumn("WARPATH", ".id", dfmod.columns, dfmod, dfIWPC, IWPCDF)
                      dropColumn("WARPATH", "Unnamed: 0.1.1", dfmod.columns, dfmod, dfIWPC, IWPCDF)

                  if False:
                       dfmod = dfmod.sample(frac=1)
                       dfIWPC = dfIWPC.sample(frac=1)
                       frames = [dfmod, dfIWPC]
                       dfmod = pd.concat(frames,ignore_index=True, sort=True)
                       dfmod = dfmod.sample(frac=1)
                       combfilename = "comb" + suffix
                       dfmod.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\combinedata\\" + combfilename + ".csv", ";")
                  else:
                     filename = "dfWarfarin" + suffix
                     dfmod.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\PreProcessed\\" + filename + ".csv", ";")
                  if True:
                      print("On imputation ", df)
                      if onlyIWPC == False:
                           dfIWPC.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Claire_IWPC_Formatted" + ".csv", ";")
                      data = dfmod
                      data.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Claire_Warpath" + ".csv", ";")
                      #data.index = data.index + 1
                      print(data.shape)
                      data['Dose_mg_week'] = data['Dose_mg_week'].apply(np.sqrt)
                      IWPCDF['Dose_mg_week'] = IWPCDF['Dose_mg_week'].apply(np.sqrt)
                      impResults = []
                      models = []
                      boot = 0
                      samples = []
                      metrics = []
                      smpResults = []
                      metric_columns = ['MAE', 'PW20']
                      listmodels = ['WarPATH']
                      if (find(models, 'model', 'WarPATH') == -1):
                          models.append({'model': 'WarPATH', 'MAE': 0, 'PW20': 0})


                      target_column = 'Dose_mg_week'
                      status_column = "Status"
                      test_size = 0.2
                      data.reset_index()
                      train, test = train_test_split(data, test_size=test_size, random_state=randomseed)  # was 66
                      traindf = pd.DataFrame(train)
                      traindf.reset_index()
                      traindf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Claire_train" + ".csv", ";")
                      testdf = pd.DataFrame(test)
                      if combinedata == True:
                          if onlyIWPC == False:
                            dfIWPC.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Claire_IWPC_precombine" + ".csv", ";")
                            frames = [traindf,dfIWPC]
                            traindf = pd.concat(frames)
                          if onlyIWPC == True:
                            traindf = pd.DataFrame()
                            frames = [traindf,dfIWPConly]
                            traindf = pd.concat(frames)
                            testdf = pd.DataFrame()
                            frames = [testdf,dfIWPC_test]
                            testdf = pd.concat(frames)
                          traindf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Claire_trainplusIWPC" + ".csv", ";")
                      traindf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Train" + suffix + ".csv", ";")
                      testdf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\Test" + suffix + ".csv", ";")
                      testdf['Status'] = 'test'
                      traindf['Status'] = 'train'
                      frames = (traindf, testdf)
                      combdf = pd.concat(frames)
                      combdf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\TrainPlusTest" + suffix + ".csv", ";")
                      combID = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\TrainPlusTest" + suffix + ".csv",";")
                      combID.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\TrainTestStatus" + suffix + ".csv", ";")
                      combID['NewStatus'] = combID['Status']
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
                      targ_col = list(target_column)
                      targetindex = x_cols.index(target_column)
                      y_train = train[target_column].values
                      x_train = train.drop([target_column], axis=1)
                      y_test = test[target_column].values
                      x_test = test.drop([target_column], axis=1)

                  #estimates = []
                  #LR = LinearRegression()
                  #estimates.append(Estimator(LR, 'LR'))

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
                         #GBR = GradientBoostingRegressor()
                         #estimates.append(Estimator(GBR,'GBR'))
                         #XGBR = XGBRegressor()
                         #estimates.append(Estimator(XGBR,'XGBR'))
                         #ELNET = ElasticNet()
                         #LAS = Lasso()
                         #estimates.append(Estimator(LAS,'LASSO'))
                         #estimates.append(Estimator(ELNET,'ELNET'))
                         #RR = Ridge()
                         #estimates.append(Estimator(RR,'RIDGE'))
                         #KNNR = KNeighborsRegressor()
                         #estimates.append(Estimator(KNNR, 'KNN'))
                         #svr = sklearn.svm.SVR()
                         #estimates.append(Estimator(svr,'SVREG'))
                         #MLPR = MLPRegressor()
                         #estimates.append(Estimator(MLPR,'MLPR'))
                         #RF = RandomForestRegressor()
                         #estimates.append(Estimator(RF, 'RF'))
                         #DTR = DecisionTreeRegressor()
                         #estimates.append(Estimator(DTR,'DTR'))
                         if "Unnamed: 0" in x_train.columns:
                            x_train.drop(["Unnamed: 0"], axis=1, inplace=True)
                         if "Unnamed: 0" in x_test.columns:
                            x_test.drop(["Unnamed: 0"], axis=1, inplace=True)
                         if "Unnamed: 0.1" in x_train.columns:
                            x_train.drop(["Unnamed: 0.1"], axis=1, inplace=True)
                         if "Unnamed: 0.1" in x_test.columns:
                            x_test.drop(["Unnamed: 0.1"], axis=1, inplace=True)
                         for _, est in enumerate(estimates):
                                 resultsdict = traineval(est, x_train, y_train, x_test, y_test, squaring=squaring,
                                                           df=df, randomseed=randomseed)
                                 alg = est.identifier
                                 x_train.to_csv(
                                 r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\training\\" + alg.replace(" ","") + str(df) + ".csv", ";")
                                 res_dict = {
                                       'Estimator': [est.identifier for x in range(len(resultsdict['PW20']))],
                                       'PW20': resultsdict['PW20'],
                                       'MAE': resultsdict['MAE'],
                                       'R2': resultsdict['R2'],
                                       'Time': resultsdict['Time']}
                                 data["WarPATH_MAE"] = ExitSquareBracket(resultsdict['MAE'], True)
                                 data["WarPATH_PW20"] = ExitSquareBracket(resultsdict['PW20'], True)
                                 dfKey = data[["WarPATH_MAE", "WarPATH_PW20"]]
                                 impResults.append(
                                       {'Imp': df, 'model': 'WarPATH', 'METRICS': dfKey, 'MAE': 0, 'PW20': 0})
                                 for k in range(len(impResults)):
                                     dfKey = impResults[k]['METRICS']
                                     model = impResults[k]['model']
                                     imputation = impResults[k]['Imp']
                                     model_MAE = dfKey["WarPATH_MAE"].astype('float')
                                     model_PW20 = dfKey["WarPATH_PW20"].astype('float')
                                     impResults[k]['MAE'] = model_MAE[k]
                                     impResults[k]['PW20'] = model_PW20[k]

                                 for k in range(len(impResults)):
                                     a = impResults[k]['model']
                                     b = impResults[k]['Imp']
                                     c = impResults[k]['MAE'].astype('float')
                                     d = impResults[k]['PW20'].astype('float')
                                 bootresults.append(
                                       # {'Imp': b, 'model': a, 'MAE': c, 'PW20': d})
                                     {'Imp': b, 'model': a, 'MAE': c, 'PW20': d})
                                 resultsdf = pd.DataFrame(bootresults)
                                 resultsdf.to_csv(
                                       r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPCResults_" + str(df) + ".csv", ";")
                                 boot = 0
                                 samples = []
                                 metrics = []
                                 smpResults = []

                                 dataSet = np.random.randint(10, size=(number_of_samples, 2))
                                 df_WARPATH = pd.DataFrame(data=dataSet, columns=metric_columns)
                                 boot = 0
                                 # print("bootstrapping..")
                                 while boot < number_of_samples:
                                    print("imputation ", df, " on sample ", boot)
                                    dfsample = data.sample(n=364, frac=None, replace=True)
                                    ## CHECK dfsample
                                    dfsample = dfsample.reset_index(drop=True)
                                    dfMetricfactors = dfsample[["WarPATH_MAE", "WarPATH_PW20"]]
                                    dfsample.to_csv(
                                        r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\IWPC_Imp" + str(df) + "Samp" + str(
                                             boot) + ".csv", ";")
                                    boot = boot + 1
                                    samples.append(dfMetricfactors)
                                    for m in range(len(listmodels)):
                                         model = listmodels[m]
                                         curr_MAE = float(dfMetricfactors[model + '_MAE'][m])
                                         curr_PW20 = float(dfMetricfactors[model + '_PW20'][m])
                                         smpResults.append(
                                             {'Imp': df, 'Sample': boot, 'model': model, 'MAE': curr_MAE,
                                              'PW20': curr_PW20})

                                 for m in range(len(listmodels)):
                                     modelinlist = listmodels[m]
                                     for l in range(len(metric_columns)):
                                        metric = metric_columns[l]
                                        for j in range(len(smpResults)):
                                            model = smpResults[j]['model']
                                            metric_value = smpResults[j][metric]
                                            if model == modelinlist:
                                                 metrics.append(
                                                       {'model': model, 'metric': metric, 'value': metric_value})

                                 for i in range(len(metric_columns)):
                                     current_metric = metric_columns[i]
                                     df_WARPATH[current_metric] = np.array(
                                         collect_Metrics(metrics, 'WarPATH', current_metric))
                                     # std = std_deviation(df_WARPATH[current_metric])
                                     std = np.square(std_deviation(df_WARPATH[current_metric]))
                                     var = variance(df_WARPATH[current_metric])
                                     # std = np.sqrt(var)
                                     std_Dev.append(
                                           {'model': 'WarPATH', 'metric': current_metric, 'SD': std, 'VAR': var})

                                 if resultsdict['MAE'] > [5]:
                                    results.append(res_dict)

                         df_res = pd.DataFrame()
                         for res in results:
                             df_res = df_res.append(pd.DataFrame.from_dict(res))
                         print(f"\n\n{df_res.groupby(['Estimator']).agg(np.mean)}\n")
                #dfResults = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfResults" + ".csv", ";")


            dfResults = pd.DataFrame(results)
            dfResults["PW20"] = dfResults.apply(lambda x: ExitSquareBracket(x["PW20"],False),axis=1).astype(float)
            dfResults["MAE"] = dfResults.apply(lambda x: ExitSquareBracket(x["MAE"], False), axis=1).astype(float)
            dfResults["R2"] = dfResults.apply(lambda x: ExitSquareBracket(x["R2"],False), axis=1).astype(float)
            dfResults["Time"] = dfResults.apply(lambda x: ExitSquareBracket(x["Time"],False), axis=1).astype(float)
            dfResults["Estimator"] = dfResults.apply(lambda x: ExitSquareBracket(x["Estimator"],False),axis=1)
            dfResults.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfResultsPRE" + ".csv", ";")
            if "Unnamed: 0" in dfResults.columns:
              dfResults.drop(["Unnamed: 0"], axis=1, inplace=True)
              MLPR_rows = dfResults.loc[dfResults['Estimator'] == 'MLPR']
              if len(MLPR_rows) > 0:
                  MAElist = MLPR_rows['MAE'].tolist()
                  maxlimit = iqr(MAElist) + 1.5*scoreatpercentile(MAElist, 75)
                  newMAElist = [x for x in MAElist if x <= maxlimit]
                  newMeanMae = np.mean(newMAElist)
                  for i in dfResults.index:
                    if dfResults.loc[i, 'MAE'] > maxlimit:
                      dfResults.at[i, 'MAE'] = np.nan
                      dfResults.at[i, 'PW20'] = np.nan
                      dfResults.at[i, 'R2'] = np.nan
                      dfResults.at[i, 'Time'] = np.nan

            dfSummary = dfResults.groupby('Estimator').apply(np.mean)
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
                    #metriclist = [j for j in metriclist if j != np.nan]
                    metriclistcopy = []
                    for j in metriclist:
                      if j != 9999:
                        if  math.isnan(j) == False:
                          metriclistcopy.append(j)
                    metriclist = metriclistcopy
                    current_stddev = 0
                    confinterval.append(
                          {'estimator': current_estimator, 'metric': current_metric, 'mean': current_mean,
                           '95% CI lower bound': current_mean - current_stddev,
                           '95% CI upper bound': current_mean + current_stddev})
                  # stddev.append({'metric':current_metric,'standarddev':current_stdde  v,'mean':current_mean})
            dfResults.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfResults" + ".csv", ";")

            dfSummary.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfSummary_" + str(randomseed) + ".csv", ";")

            dfConfidence = pd.DataFrame(confinterval,
                                         columns=['estimator', 'metric', 'mean', '95% CI lower bound',
                                                  '95% CI upper bound'])
            dfConfidence.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfConfidence" + ".csv", ";")
            print("STOP HERE")


            if df == impNumber:
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
                   for m in range(len(bootresults)):
                        if bootresults[m]['model'] == fieldname:
                            models[k]['MAE'] += bootresults[m]['MAE']
                            models[k]['PW20'] += bootresults[m]['PW20']
                 Bfactor = (impNumber + 1) / impNumber

                 for k in range(len(models)):
                   fieldname = models[k]['model']
                   mae_value = models[k]['MAE'] / impNumber
                   mae_list = collect_Results(bootresults, fieldname, 'MAE')
                   mae_variance = variance(mae_list) * Bfactor
                   stdpos = find(std_Dev_Summ, 'model', fieldname)
                   varpos = find(variance_Summ, 'model', fieldname)
                   mae_var = variance_Summ[varpos]['MAE'] * 2
                   mae_std_dev = std_Dev_Summ[stdpos]['MAE'] / impNumber
                   mae_CI_minus = round(mae_value - 1.96 * np.sqrt(mae_std_dev + mae_variance), 4)
                   mae_CI_plus = round(mae_value + 1.96 * np.sqrt(mae_std_dev + mae_variance), 4)
                   pw20_value = models[k]['PW20'] / impNumber
                   pw20_list = collect_Results(bootresults, fieldname, 'PW20')
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
                   dfCurr["PW2O"] = [round(pw20_value, 6), pw20_CI_minus, pw20_CI_plus]
                   dfCurr["TIME"] = timePeriod
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

