import random
import sklearn
import csv
import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from matplotlib import pyplot as plt
from warfit_learn import datasets, preprocessing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import LinearSVR, SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier # for extreme gradient boosting model
from warfit_learn.estimators import Estimator
from warfit_learn.evaluation import evaluate_estimators
from warfit_learn.metrics.scoring import confidence_interval
from scipy.stats import norm
from mlxtend.regressor import StackingCVRegressor
import warnings
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
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, \
    MinMaxScaler, FunctionTransformer, Normalizer
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoLarsCV, Lasso, ElasticNet, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import BaggingClassifier
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

def format_summary(df_res):
    df_summary = df_res.groupby(['Estimator']).mean()
    df_summary.reset_index(inplace=True)
    for alg in df_res['Estimator'].unique():
        for metric in ['PW20', 'MAE']:
            lo, hi = confidence_interval(df_res[metric][df_res['Estimator'] == alg].values, )
            mean = df_res[metric][df_res['Estimator'] == alg].mean()
            for v in [mean, lo, hi]:
                if not (-10000 < v < 10000):
                    print('nan applied: ', alg, metric, lo, hi, mean)
                    mean, lo, hi = np.nan, np.nan, np.nan
                conf = f"{mean:.2f}({lo:.2f}-{hi:.2f})"
                print(alg, metric, lo, hi, mean, conf)
                df_summary[metric][df_summary['Estimator'] == alg] = conf
    return df_summary

def collect_Metrics(metrics, model, metric):
    container = []
    for i in range(len(metrics)):
        if (metrics[i]['model'] == model) & (metrics[i]['metric'] == metric):
            container.append(metrics[i]['value'])
    return container

def collect_Results(metrics, model, metric):
    container = []
    for i in range(len(metrics)):
        if (metrics[i]['model'] == model):
            container.append(metrics[i][metric])
    return container

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

def ConvertHIV(row,matching):
    row = int(row)
    for i in range(len(matching)):
      if row == matching[i]:
        return 1
    else:
        return 0


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


def main():
    # Assume that the number of cores per socket in the machine is denoted as NUM_PARALLEL_EXEC_UNITS
    #  when NUM_PARALLEL_EXEC_UNITS=0 the system chooses appropriate settings

    #config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
    #                        inter_op_parallelism_threads=2,
    #                        allow_soft_placement=True,
    #                        device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})

    #session = tf.Session(config=config)

    dftemplatetrain = pd.DataFrame()
    dftemplatetest = pd.DataFrame()
    impNumber = 100
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    pd.set_option('expand_frame_repr', False)
    pd.set_option("display.max_rows", False)
    dfmax = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MiceRWarPATHData.csv", ";")
    df = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MiceRWarPATHData.csv", ";")

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
    for root, dirs, files in os.walk(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations"):
        for file in files:
            if file.endswith('.csv') and "TEST" not in file and "TRAIN" not in file:
                filesImp.append(file)

    for file in filesImp:
        dfnew = pd.read_csv(root + '\\' + file, ";")

        fileindex = filesImp.index(file)
        rootIWPC = root.replace("WarImputations","MICESTATSMODEL\\")
        rootnewIWPC = root.replace("WarImputations","MICESTATSMODELHIV\\")
        #rootnewIWPC = root.replace("WarImputations","MICESTATS25\\")
        #rootnewIWPC = root.replace("WarImputations", "MICESTATS50\\")
        #rootnewIWPC = root.replace("WarImputations", "MICESTATS75\\")
        #rootnewIWPC = root.replace("WarImputations", "MICESTATS150\\")
        #rootnewIWPC = root.replace("WarImputations", "MICESTATS125\\")
        IWPC_csv = rootIWPC + filesIWPC[fileindex]
        dfIWPC = pd.read_csv(IWPC_csv,';')
        df = fileindex + 1
        print("adding HIV columns to ",dfIWPC, "imputation ",df)
        dfmod = dfnew
        dfmodcount = len(dfmod.index)
        dfmodcount25percent = int(len(dfmod.index)/4)
        dfmodcount50percent = int(len(dfmod.index)/2)
        dfmodcount75percent = int(dfmodcount25percent*3)
        dfmodcount125percent = int(len(dfmod.index)*5/4)
        dfmodcount150percent = int(len(dfmod.index)*3/2)
        dfmod.drop(['Gender', 'Country_recruitment'], axis=1, inplace=True)
        dfIWPC.drop(['AgeDecades'], axis=1, inplace=True)
        dfIWPC.drop(['INR_Three'], axis=1, inplace=True)
        # dfmod["INR_Three"] = np.where(dfmod["Target_INR"] == "Three", 1, 0)
        dfmod["Target_INR"] = np.where(dfmod["Target_INR"] == "Three", 3.0,
                                       np.where(dfmod["Target_INR"] == "aTwo_point_five", 2.5, 2.0))
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
        dfmod_countHIVPositive = np.sum(dfmod["HIVPositive"])
        #fIWPCcountHIVPositive = int(dfmod_countHIVPositive * dfIWPCcount / dfmodcount)
        #IWPC_HIVPositive = random.sample(range(dfIWPCcount - 1), dfIWPCcountHIVPositive)
        #IWPC_HIVPositive.sort()
        #dfIWPC["HIVPositive"] = dfIWPC.apply(lambda x: ConvertHIV(x["Unnamed: 0"], IWPC_HIVPositive), axis=1)
        dfmod["HIVUnknown"] = np.where(dfmod["HIV_status"] == "Unknown", 1, 0)
        dfmod_countHIVUnknown = np.sum(dfmod["HIVUnknown"])
        #dfIWPCcountHIVUnknown = int(dfmod_countHIVUnknown * dfIWPCcount / dfmodcount)
        #IWPC_HIVUnknown = random.sample(range(dfIWPCcount - 1), dfIWPCcountHIVUnknown)
        #IWPC_HIVUnknown.sort()
        #dfIWPC["HIVUnknown"] = dfIWPC.apply(lambda x: ConvertHIV(x["Unnamed: 0"], IWPC_HIVUnknown), axis=1)
        #dfIWPC.drop(["Unnamed: 0"], axis=1, inplace=True)
        #dfIWPC.to_csv(rootnewIWPC + filesIWPC[fileindex], ";")
        dfmod.drop(["HIV_status"], axis=1, inplace=True)
        # dfmod['BSA'] = dfmod.apply(lambda x: BSA(x["Height_cm"], x["Weight_kg"]), axis=1)
        # dfmod.drop(["Height_cm"], axis = 1, inplace = True)
        # dfmod.drop(["Weight_kg"], axis=1, inplace=True)
        dfmod.drop(["Unnamed: 0"], axis=1, inplace=True)
        #dfmod.drop(["Unnamed: 0.1"], axis=1, inplace=True)
        dfmod.drop(["Age_years"], axis=1, inplace=True)
        dfmod.drop([".imp"], axis=1, inplace=True)
        dfmod.drop([".id"], axis=1, inplace=True)
        data = dfmod
        test_size = 0.2
        target_column = 'Dose_mg_week'
        train, test = train_test_split(data, test_size=test_size)
        traindf = pd.DataFrame(train)
        testdf = pd.DataFrame(test)
        y_train = train[target_column].values
        y_train = pd.DataFrame(y_train)
        x_train = train.drop([target_column], axis=1).values
        x_train = pd.DataFrame(x_train)
        y_test = test[target_column].values  # must isolate this and x_test to prevent data leakage.
        y_test = pd.DataFrame(y_test)
        x_test = test.drop([target_column], axis=1).values
        x_test = pd.DataFrame(x_test)
        suffix = str(df).zfill(3)
        testdf.drop(["Unnamed: 0.1"], axis=1, inplace=True)
        traindf.drop(["Unnamed: 0.1"], axis=1, inplace=True)

        dftemplatetest = dftemplatetest.append(testdf)
        #dftemplatetrain = dftemplatetrain.append(traindf)
        #dftemplatetest.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\ImpWarPATH_TEST20_" + suffix + ".csv",
        #              ";")

        traindf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\ImpWarPATH_TRAIN80_" + suffix + ".csv",
                   ";")
        testdf.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\ImpWarPATH_TEST20_" + suffix + ".csv",
                       ";")
        #frames = [dftemplatetest, traindf]
        #dftemplatetest.compare(traindf, align_axis=0)
        #dfcompare = pd.concat(frames)
        #print(dfcompare.duplicated())
        #duplicate = dfcompare[dfcompare.duplicated()]
        #duplicate = dftemplatetest[dftemplatetest.duplicated()]
        #print(duplicate)

        #if len(duplicate) >= 1:
        #    print("problem")

    if False:
        data = dfmod
        data['Dose_mg_week'] = data['Dose_mg_week'].apply(np.sqrt)
        LR = LinearRegression()
        RR = Ridge(alpha=0.02, solver="lsqr", fit_intercept=True)
        LAS = Lasso(alpha=0.002)
        KNN = KNeighborsRegressor(weights="uniform", p=1, n_neighbors=14, algorithm="brute")
        RF = RandomForestRegressor(max_features='sqrt', bootstrap=True, n_estimators=500, max_depth=10,
                                   min_samples_split=2, min_samples_leaf=5)
        XGBR = XGBRegressor(learning_rate=0.01, colsample_bytree=0.3, max_depth=3, n_estimators=500,
                            objective='reg:squarederror')
        estimates = []
        estimates.append(Estimator(LR, 'LR'))
        estimates.append(Estimator(RR, 'RR'))
        estimates.append(Estimator(LAS, 'LASSO'))
        estimates.append(Estimator(XGBR, 'XGBR'))
        estimates.append(Estimator(RF, 'RF'))
        estimates.append(Estimator(KNN, 'KNN'))

        warpath_results = evaluate_estimators(estimates,
                                              data,
                                              target_column='Dose_mg_week'
                                              , scale=True
                                              , test_size=0.2
                                              , squaring=True
                                              , technique='mccv'
                                              , parallelism=0.8
                                              )
        print(warpath_results)
        summary = warpath_results.groupby('Estimator').apply(np.mean)
        print(summary)
        dftemplate = dftemplate.append(summary)
        warpath_formatted = format_summary(warpath_results)
        df_final = pd.concat([warpath_formatted], axis=1, keys=['WARPATH'])
        print(df_final)






if __name__ == "__main__":
    main()

