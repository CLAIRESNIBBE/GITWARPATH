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
from sklearn.model_selection import GridSearchCV
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

def traineval(est: Estimator, grid, xtrain, ytrain, xtest, ytest, squaring):
    results = {'PW20': [], 'MAE': [], 'R2': []}
    res=[]
    print(f'\n{est.identifier}...')
    mae_scorer = make_scorer(MAEScore)
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    gridinitial = grid[est.identifier]
    ytest_numpy = np.array(ytest)
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
    # filesTrain=[]
    # filesTest=[]

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
            #unnamed_column = "Unnamed: 0.1.1"
            train = dfmod.loc[dfmod["Status"] == "train"]
            test = dfmod.loc[dfmod["Status"] == "test"]
            # train, test = train_test_split(data, test_size=test_size)
            squaring = True
            train = train.drop([status_column], axis=1)
            test = test.drop([status_column], axis=1)
            x_cols = list(train.columns)
            #_cols_notarg = x_cols.remove(target_column)
            targ_col = list(target_column)
            #train = scaler.fit_transform(train)
            #test = scaler.transform(test)
            #train = pd.DataFrame(train, columns = x_cols)
            #test = pd.DataFrame(test, columns = x_cols)
            targetindex = x_cols.index(target_column)
            y_train = train[target_column].values
            x_train = train.drop([target_column], axis=1)
            y_test = test[target_column].values
            x_test = test.drop([target_column], axis=1)
            LR = LinearRegression()
            estimates.append(Estimator(LR, 'LR'))

            # RIDGE
            grid = dict()
            ridge_alphas = np.linspace(0, 0.02, 11)
            ridge_solvers = []
            ridge_solvers.append('svd')
            ridge_solvers.append('cholesky')
            ridge_solvers.append('lsqr')
            ridge_solvers.append('sag')
            RR = Ridge(alpha=0.02, solver="lsqr")
            estimates.append(Estimator(RR, 'RR'))

            # LASSO
            lasso_alphas = np.linspace(0, 0.02, 11)
            LAS = Lasso(alpha=0.002)
            estimates.append(Estimator(LAS, 'LAS'))

            # EL
            EL = ElasticNet(alpha=0.01, l1_ratio=0.01)
            ratios = np.arange(0, 1, 0.01)
            alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
            estimates.append(Estimator(EL, 'EL'))

            KNN = KNeighborsRegressor()
            k_values = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
            param_grid = dict(n_neighbors=k_values)
            estimates.append(Estimator(KNN, 'KNN'))

            RF=RandomForestRegressor(n_estimators=1000)
            # n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=10)]
            # max_features = ['auto', 'sqrt']
            # max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
            # min_samples_split = [2, 5, 10]
            # min_samples_leaf = [1, 2, 4]
            # bootstrap = [True, False]
            # random_grid = {'n_estimators': n_estimators}
            estimates.append(Estimator(RF,'RF'))

            grid = {'RR': {'alpha': ridge_alphas, 'solver': ridge_solvers},
                    'RF': {'n_estimators':1000},
                    'LAS': {'alpha': lasso_alphas},
                    'EL': {'alpha': alphas, 'l1_ratio': ratios},
                    'LR': {},
                    'KNN': param_grid
                    }

            results2 = []
            for _, est in enumerate(estimates):
                results2 = traineval(est, grid, x_train, y_train, x_test, y_test, squaring=squaring)
                res_dict = {
                    'Estimator': [est.identifier for x in range(len(results2['PW20']))],
                    'PW20': results2['PW20'],
                    'MAE': results2['MAE'],
                    'R2': results2['R2']}
                results.append(res_dict)

                # rog = {k: [np.mean(res_dict[k])]
                #         for k in list(res_dict.keys())[1:]}
                # print(tabulate(prog, headers=prog.keys()))
                # results.append(res_dict)

            df_res = pd.DataFrame()
            for res in results:
                df_res = df_res.append(pd.DataFrame.from_dict(res))
            print(f"\n\n{df_res.groupby(['Estimator']).agg(np.mean)}\n")
            if False:
                ab_EL = AdaBoostRegressor(EL, n_estimators=400, random_state=7)
                SGD = SGDRegressor(penalty="l2")
                KNN = KNeighborsRegressor(weights="uniform", p=1, n_neighbors=14, algorithm="brute")
                NN = MLPRegressor(hidden_layer_sizes=(100,), activation="relu", random_state=1, max_iter=2000)
                # NN = MLPRegressor(hidden_layer_sizes=(100,), activation='logistic', solver='lbfgs', max_iter=1000)
                # SV = SVR(kernel='linear', cache_size=1000)
                # SVReg = SVR(epsilon=1.5, kernel='sigmoid',C=2.0)

                # SVR = LinearSVR(C=9.59, epsilon=0.42, fit_intercept=True)
                # DTR = DecisionTreeRegressor(criterion="friedman_mse",max_depth=11, max_features='sqrt', max_leaf_nodes=40, min_impurity_decrease=0.8,min_samples_leaf=7,min_weight_fraction_leaf=0.1,splitter='best')
                # DTR = DecisionTreeRegressor(max_depth=4)
                # ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
                # BRT = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100)
                # XGBR = XGBRegressor(learning_rate=0.01, colsample_bytree=0.3, max_depth=3, n_estimators=500,
                #                   objective='reg:squarederror')
                # RandomForestRegressor(max_depth=40, max_features=2, min_samples_leaf=4, min_samples_split=12)

                # RF = RandomForestRegressor(max_features='sqrt', bootstrap=True, n_estimators=500, max_depth=10,
                #                       min_samples_split=2, min_samples_leaf=5)
                # ab_RF = AdaBoostRegressor(RF, n_estimators=400, random_state=7)
                # CUBE = Cubist()
                # XGB = XGBRegressor(max_depth=10, n_estimators=1000, min_child_weight=5, subsample=0.6,
                #               alpha=0.1, eta=0.1, seed=42)
                # n_trees = 100
                # mdepth = 6
                # gamma = 1
                # lam = 1
                # XG = XGBClassifier(
                # learning_rate=0.1,
                # n_estimators=1000,
                # max_depth=5,
                # min_child_weight=1,
                # gamma=0,
                # subsample=0.8,
                # colsample_bytree=0.8,
                # objective='multi:softmax',
                # nthread=4,
                # scale_pos_weight=1,
                # seed=27,
                # num_class=3,
                # )
                # BAG = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
                # XG = XGBClassifier(use_label_encoder=False,
                #                      booster='gbtree',  # boosting algorithm to use, default gbtree, othera: gblinear, dart
                #                      n_estimators=n_trees,  # number of trees, default = 100
                #                      eta=0.3,  # this is learning rate, default = 0.3
                #                      max_depth=mdepth,  # maximum depth of the tree, default = 6
                #                      gamma=gamma,
                #                      # used for pruning, if gain < gamma the branch will be pruned, default = 0
                #                      reg_lambda=lam,  # regularization parameter, defautl = 1
                #                      # min_child_weight=0 # this refers to Cover which is also responsible for pruning if not set to 0
                #                      )
                estimates.append(Estimator(LR, 'LR'))
                estimates.append(Estimator(BRT, 'BRT'))
                # estimates.append(Estimator(GBT, 'GBT'))
                estimates.append(Estimator(RF, 'RF'))
                estimates.append(Estimator(CUBE, 'Cubist'))
                estimates.append(Estimator(XG, 'XGB'))
                estimates.append(Estimator(NN, 'NN'))
                estimates.append(Estimator(RR, 'RR'))
                # estimates.append(Estimator(SV, 'SV'))
                estimates.append(Estimator(EL, 'EL'))
                # models = list()
                # models.append(('KNN', KNeighborsRegressor(weights="uniform", p=1, n_neighbors= 14,algorithm = "brute")))
                # models.append(('DTR', DecisionTreeRegressor(max_depth=4)))
                # models.append(('SVR', SVR(epsilon=1.5, kernel='sigmoid',C=2.0)))
                # scores = evaluate_models(models, x_train, x_test, y_train, y_test)
                # ensemble1 = VotingRegressor(estimators = models,weights = scores)
                estimates.append(
                    Estimator(LinearSVR(epsilon=0.0, tol=0.0001, C=1.0, loss='epsilon_insensitive'), 'SVR'))
                estimates.append(
                    Estimator(StackingCVRegressor(regressors=[SVR, KNN, BRT], meta_regressor=SVR, cv=5, ),
                              'Stacked_SVR'))
                estimates.append(
                    Estimator(StackingCVRegressor(regressors=[XGB, SVR, NN], meta_regressor=SVR, cv=5, ), 'Stacked_SV'))
                # estimates.append(Estimator(BAG, 'Bag'))
                estimates.append(Estimator(LAS, 'Lasso'))
                tpot2 = make_pipeline(
                    StackingEstimator(
                        estimator=LinearSVR(
                            C=1.0,
                            dual=True,
                            epsilon=0.01,
                            loss="epsilon_insensitive",
                            tol=0.001, )),
                    StackingEstimator(
                        estimator=ElasticNetCV(l1_ratio=0.6000000000000001, tol=0.01, cv=5)),
                    RobustScaler(),
                    StackingEstimator(estimator=RidgeCV()),
                    ExtraTreesRegressor(
                        bootstrap=True,
                        max_features=1.0,
                        min_samples_leaf=20,
                        min_samples_split=2,
                        n_estimators=100, )
                )
                tpot10 = make_pipeline(
                    StackingEstimator(estimator=ExtraTreesRegressor(
                        bootstrap=True, max_features=0.05,
                        min_samples_leaf=18, min_samples_split=10,
                        n_estimators=100)),
                    MaxAbsScaler(),
                    StackingEstimator(estimator=ExtraTreesRegressor(
                        bootstrap=True, max_features=0.05,
                        min_samples_leaf=18, min_samples_split=10, n_estimators=100)),
                    LassoLarsCV(normalize=True, cv=3)
                )
                tpot17 = make_pipeline(
                    make_union(
                        FunctionTransformer(copy, validate=True),
                        MaxAbsScaler()
                    ),
                    StackingEstimator(estimator=RidgeCV()),
                    ZeroCount(),
                    GradientBoostingRegressor(alpha=0.9, learning_rate=0.1, loss="lad",
                                              max_depth=3, max_features=0.9000000000000001,
                                              n_estimators=100, subsample=0.55)
                )
                estimates.append(Estimator(tpot2, 'TPOT2'))
                estimates.append(Estimator(tpot10, 'TPOT10'))
                estimates.append(Estimator(tpot17, 'TPOT17'))
                estimates = []
                #
                # estimates.append(Estimator(DTR,'DTR'))
                # estimates.append(Estimator(ab_regressor,'ABDTR'))
                # estimates.append(Estimator(LR, 'LR'))
                # estimates.append(Estimator(ab_RR,'ABRR'))
                # estimates.append(Estimator(ab_RF, 'ABRF'))
                # estimates.append(Estimator(ab_EL, 'ABEL'))
                # estimates.append(Estimator(ab_LAS, 'ABLasso'))
                # estimates.append(Estimator(XGBR, 'XGBR'))
                # estimates.append(Estimator(RR, 'RR'))
                # estimates.append(Estimator(RF, 'RF'))
                # estimates.append(Estimator(EL, 'EL'))
                # estimates.append(Estimator(LAS, 'Lasso'))
                # estimates.append(Estimator(SGD, 'SGD'))
                # estimates.append(Estimator(SVReg, 'SVR'))
                # estimates.append(Estimator(NN, 'NN'))
                # estimates.append(Estimator(KNN,"KNN"))
                # estimates.append(Estimator(ensemble1,'Ensemble1')) #KNN,DTR,SVR
                warpath_results = evaluate_estimators(estimates,
                                                      data,
                                                      target_column='Dose_mg_week'
                                                      , scale=True
                                                      , test_size=0.1
                                                      , squaring=True
                                                      , technique='mccv'
                                                      , parallelism=0.8
                                                      )
                print(warpath_results)
                summary = warpath_results.groupby('Estimator').apply(np.mean)
                print(summary)
                dftemplate = dftemplate.append(summary)
                warpath_formatted = format_summary(warpath_results)
                dfWarPath = dfWarPath.append(warpath_results)
                df_final = pd.concat([warpath_formatted], axis=1, keys=['WARPATH'])
                print(df_final)
                suffix = str(df).zfill(3)
                df_final.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_" + suffix + ".csv", ";")
                if False:
                    model.fit(X, y, epochs=150, batch_size=10, verbose=0)
                    # make class predictions with the model
                    predictions = (model.predict(X) > 0.5).astype(int)
                    # summarize the first 50 cases
                    for i in range(50):
                        print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
    if False:
        dftemplate.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dftemplate" + ".csv", ";")
        dfSummary = dftemplate.groupby('Estimator').apply(np.mean)
        dfSummary.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfSummary" + ".csv", ";")
        dfWarPath_formatted = format_summary(dfWarPath)
        dfWarPath_formatted.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfWarPath" + ".csv", ";")
        print('the end')

    dfResults = pd.DataFrame(results)
  #  dfResults['MAE'] = dfResults['MAE'].str.strip('[]').astype(float)
  #  dfResults['PW20'] = dfResults['PW20'].str.strip('[]').astype(float)
  #  dfResults['R2'] = dfResults['R2'].str.strip('[]').astype(float)

    dfResults.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfResults" + ".csv", ";")
   # print(dfResults.groupby('Estimator').mean())
    print("STOP HERE")


if __name__ == "__main__":
    main()

