import sklearn
import csv
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from warfit_learn import datasets, preprocessing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.svm import LinearSVR, SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost
from xgboost import XGBClassifier # for extreme gradient boosting model
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
from sklearn.ensemble import BaggingClassifier,  VotingRegressor
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
#import tensorflow as tf
#from tensorflow import keras
from numpy import loadtxt
#from keras.models import Sequential
#from keras.layers import Dense
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
            df=pd.DataFrame(data,columns=[metric])
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

def PercIn20(true, predicted):
    patients_in_20 = 0
    for i in range(len(true)):
        if abs(true[i] - predicted[i]) < 0.2 * true[i]:
            patients_in_20 += 1
    return 100 * patients_in_20 / len(true)

def INRThree(targetINR):
    if (targetINR >= 2.5).all() & (targetINR <= 3.5).all():
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

def traineval(estimator: Estimator,xtrain,ytrain,xtest,ytest,squaring):
    for _, est in enumerate(estimator):
          print(f'\n{est.identifier}...')
    if squaring:
        ytest = np.square(ytest)
    estimator.fit(xtrain, ytrain)
    predicts = estimator.predict(xtest)
    if squaring:
        predicts = np.square(predicts)
    return (
        mean_absolute_error(ytest, predicts),
        PercIn20((ytest, predicts),
        RSquared(ytest, predicts)))

def main():
    # Assume that the number of cores per socket in the machine is denoted as NUM_PARALLEL_EXEC_UNITS
    #  when NUM_PARALLEL_EXEC_UNITS=0 the system chooses appropriate settings

    #config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
    #                        inter_op_parallelism_threads=2,
    #                        allow_soft_placement=True,
    #                        device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})

    #session = tf.Session(config=config)
    combinedata = False
    dftemplate = pd.DataFrame()
    dfWarPath = pd.DataFrame()
    impNumber = 100
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    pd.set_option('expand_frame_repr', False)
    pd.set_option("display.max_rows", False)
    df = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MiceRWarPATHData.csv", ";")
    if combinedata == True:
      filesIWPC = []
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
    fixedtraintest = False
    while fixedtraintest==False:
      for root, dirs, files in os.walk(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations"):
        for file in files:
            if file.endswith('.csv') and 'TEST' not in file and 'TRAIN' not in file:
                #filesImp.append(file)
                if not fixedtraintest:
                    filedf = pd.read_csv(root + '\\' + file, ";")
                    trainID,testID = train_test_split(filedf,test_size=0.2)
                    fixedtraintest = True
    #for imp in range(impNumber):
    patients_train = []
    # collect training set data
    for row in trainID.itertuples():
       currentid = row[4]
       patients_train.append(currentid)
    dftrain = df[df['.id'].isin(patients_train)]
    dftrain.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\Training\train" + suffix + ".csv",";")
    for imp in range(impNumber):
       counter = imp+1
       dftrainimp = dftrain.loc[df[".imp"]==counter]
       suffix = str(counter).zfill(3)
       dftrainimp.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\Training\train_" + suffix + ".csv", ";")
    patients_test = []
    #collect testing set data
    for row in testID.itertuples():
       currentid = row[4]
       patients_test.append(currentid)
    dftest = df[df['.id'].isin(patients_test)]
    dftest.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\Testing\test" + suffix + ".csv",";")
    for imp in range(impNumber):
        counter = imp+1
        dftestimp = dftest.loc[df[".imp"]==counter]
        suffix = str(counter).zfill(3)
        dftestimp.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\Testing\test_" + suffix + ".csv", ";")
    #filesTrain=[]
    #filesTest=[]

    #for root, dirs, files in os.walk(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\Training\train"):
    counter = 0
    for root, dirs, files in os.walk(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations"):
        for file in files:
            if file.endswith('.csv') and ("train_" not in file and "test_" not in file and "TRAIN" not in file and "TEST" not in file) and "ImpWarPATH" in file:
                filedf = pd.read_csv(root + '\\' + file, ";")
                if "Status" not in filedf.columns:
                   filedf["Status"]=""
                   counter = counter + 1
                for row in filedf.itertuples():
                    checkID = row[4]
                    rowindex = filedf.loc[filedf[".id"] == checkID].index.tolist()[0]  # OR row[0]
                    if checkID in patients_train:
                        filedf.loc[rowindex, 'Status'] = 'train'
                    elif checkID in patients_test:
                        filedf.loc[rowindex, 'Status'] = 'test'
                suffix = str(counter).zfill(3)
                filedf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\ImpWarPATHSPLIT_" + suffix + ".csv", ";")
                filesImp.append(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\ImpWarPATHSPLIT_" + suffix + ".csv")

if __name__ == "__main__":
    main()

