import copy
import sklearn
import sklearn_evaluation as sklearneval
import csv
import os
import plotly.graph_objects as go
import os.path
import pandas as pd
import numpy as np
import scikitplot as skplot
from tabulate import tabulate
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from IPython.display import display
from sklearn_evaluation.plot import grid_search
from yellowbrick.features import ParallelCoordinates
from yellowbrick.regressor import AlphaSelection, PredictionError, ResidualsPlot
from warfit_learn import datasets, preprocessing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
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
from xgboost import plot_tree
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
#from cubist import Cubist

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


from imports import *
import logging

# a function  to create and save logs in the log files
def log(path, file):
    """[Create a log file to record the experiment's logs]

    Arguments:
        path {string} -- path to the directory
        file {string} -- file name

    Returns:
        [func] -- [logger that record logs]

    Author:

    """

    # check if the file exist
    log_file = os.path.join(path, file)

    if not os.path.isfile(log_file):
        open(log_file, "w+").close()

    console_logging_format = "%(levelname)s %(message)s"
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()

    # create a file handler for output file
    handler = logging.FileHandler(log_file)

    # set the logging level for log file
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger


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

def grid_searchnew(params, reg, x_train, y_train, x_test, y_test):
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

def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')


def GridSearch_table_plot(grid_clf, param_name,
                          num_results=15,
                          negative=True,
                          graph=True,
                          display_all_params=True):
    '''Display grid search results

    Arguments
    ---------

    grid_clf           the estimator resulting from a grid search
                       for example: grid_clf = GridSearchCV( ...

    param_name         a string with the name of the parameter being tested
    num_results        an integer indicating the number of results to display
                       Default: 15

    negative           boolean: should the sign of the score be reversed?
                       scoring = 'neg_log_loss', for instance
                       Default: True

    graph              boolean: should a graph be produced?
                       non-numeric parameters (True/False, None) don't graph well
                       Default: True

    display_all_params boolean: should we print out all of the parameters, not just the ones searched for?
                       Default: True

    Usage
    -----

    GridSearch_table_plot(grid_clf, "min_samples_leaf")

                          '''
    from matplotlib import pyplot as plt
    from IPython.display import display
    import pandas as pd
    clf = grid_clf.best_estimator_
    clf_params = grid_clf.best_params_
    if negative:
        clf_score = -grid_clf.best_score_
    else:
        clf_score = grid_clf.best_score_
    clf_stdev = grid_clf.cv_results_['std_test_score'][grid_clf.best_index_]
    cv_results = grid_clf.cv_results_

    print("best parameters: {}".format(clf_params))
    print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))
    if display_all_params:
        import pprint
        pprint.pprint(clf.get_params())

    # pick out the best results
    # =========================
    scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')

    best_row = scores_df.iloc[0, :]
    if negative:
        best_mean = -best_row['mean_test_score']
    else:
        best_mean = best_row['mean_test_score']
    best_stdev = best_row['std_test_score']
    best_param = best_row['param_' + param_name]

    # display the top 'num_results' results
    # =====================================
    display(pd.DataFrame(cv_results) \
            .sort_values(by='rank_test_score').head(num_results))

    # plot the results
    # ================
    scores_df = scores_df.sort_values(by='param_' + param_name)

    if negative:
        means = -scores_df['mean_test_score']
    else:
        means = scores_df['mean_test_score']
    stds = scores_df['std_test_score']
    params = scores_df['param_' + param_name]
    # plot
    if graph:
        plt.figure(figsize=(8, 8))
        plt.errorbar(params, means, yerr=stds)

        plt.axhline(y=best_mean + best_stdev, color='red')
        plt.axhline(y=best_mean - best_stdev, color='red')
        plt.plot(best_param, best_mean, 'or')

        plt.title(param_name + " vs Score\nBest Score {:0.5f}".format(clf_score))
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.show()
    # Calling Method
#plot_grid_search(pipe_grid.cv_results_, n_estimators, max_features, 'N Estimators', 'Max Features')

def GridSearch_Contour_plot(grid_clf, paramlist, param1, param2, Contour, Surface):
    paramdf = pd.DataFrame(grid_clf.cv_results_["params"])
    meantestdf = pd.DataFrame(grid_clf.cv_results_["mean_test_score"], columns=["Accuracy"])
    grid_res = pd.concat([paramdf, meantestdf], axis=1)
    grid_contour = grid_res.groupby(paramlist).mean()
    grid_reset = grid_contour.reset_index()
    grid_reset.columns = [param1, param2, 'Accuracy']
    grid_pivot = grid_reset.pivot(param2, param1)
    x = grid_pivot.columns.levels[1].values
    y = grid_pivot.index.values
    z = grid_pivot.values
    # X and Y axes labels
    layout = go.Layout(
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text=param1)
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text=param2)
        ))
    if Contour == True:
        fig = go.Figure(data=[go.Contour(z=z, x=x, y=y)], layout=layout)

        fig.update_layout(title='Hyperparameter tuning - 2D Contour Plot', autosize=False,
                          width=500, height=450,
                          margin=dict(l=65, r=50, b=65, t=90))
        fig.show()
    else:
        fig = go.Figure(data=[go.Surface(z=z, y=y, x=x)], layout=layout)

        fig.update_layout(title='Hyperparameter tuning - 3D Surface Plot',
                          scene=dict(
                              xaxis_title=param1,
                              yaxis_title=param2,
                              zaxis_title='Accuracy'),
                          autosize=False,
                          width=800, height=500,
                          margin=dict(l=65, r=50, b=65, t=90))
        fig.show()


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
    #sklearn.externals.joblib.load('gridsearch.pkl')
    scaler = MinMaxScaler()
    dftemplate = pd.DataFrame()
    dfWarPath = pd.DataFrame()
    impNumber = 3
    maxImp = 3
    runImp = 0
    n_bootstraps = 10

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
            if runImp < maxImp and file.endswith('.csv') and (
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
                runImp = runImp + 1
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
        for boot in range(n_bootstraps):
            dfsample = dfmod.sample(n=364, frac=None, replace=True)
            dfsample = dfsample.reset_index(drop=True)
            boot = boot + 1
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
            # define model
            space = {'kernel': {'linear': {'C': [0, 100]},
                                'rbf': {'gamma': ['scale'], 'C': [1, 100]},
                                'poly': {'degree': [2, 5], 'C': [1000, 20000], 'coef0': [0, 1]}
                                }
                     }



            if False:
                model = model = sklearn.svm.SVR()
                # define model evaluation method
                cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
                pipeline_scaled = Pipeline([('scale', MinMaxScaler()), ('alg', model)])
                # define grid
                grid = dict()
                grid['alg__kernel'] = ['rbf']
                grid['alg__gamma'] = ['scale','auto']
                grid['alg__C'] = [ 1, 100]
                grid2 = dict()
                grid2['alg__kernel'] = ['linear']
                grid2['alg__C'] = [0,100]
                grid3 = dict()
                grid3['alg__kernel'] = ['poly']
                grid3['alg__degree'] = [2,5]
                grid3['alg__C'] = [1000,20000]
                grid3['alg__coef0'] = [0,1]
            # define search
                search = GridSearchCV(estimator=pipeline_scaled, param_grid=grid, scoring='neg_mean_absolute_error', cv=cv,
                                  n_jobs=-1, verbose = 3)
                result = search.fit(x_train, y_train)

                search = GridSearchCV(estimator=pipeline_scaled, param_grid=grid2, scoring='neg_mean_absolute_error', cv=cv,
                                  n_jobs=-1, verbose=3)
                result2 = search.fit(x_train, y_train)

                search = GridSearchCV(estimator=pipeline_scaled, param_grid=grid3, scoring='neg_mean_absolute_error', cv=cv,
                                  n_jobs=-1, verbose=3)
                result3 = search.fit(x_train, y_train)
            # Use the random grid to search for best hyperparameters
            # First create the base model to tune
            if True:
                RFR = RandomForestRegressor(max_depth=90, max_features='sqrt', min_samples_leaf=60,
                                           min_samples_split=12, n_estimators=300)
                #grid = {'n_estimators': [10, 50, 100, 500],'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0]}
                #grid = {'n_estimators': [10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 500], 'learning_rate': [0.0001, 0.001, 0.005, 0.01, 0.015,0.1, 1.0]}
                #grid['n_estimators'] = [10, 50, 100, 500]
                #grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
                # define the evaluation procedure
                kfold = KFold(n_splits=10, shuffle=True, random_state=2)
                # define the grid search procedure
                grid = {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
                    'loss': ['linear', 'square', 'exponential']
                }
                grid2 = {'n_estimators':[100,150,200,500,1000],
                         'learning_rate':[0.1, 0.15, 0.2, 0.25, 0.3],
                         'loss':['linear']}
                Ridge1 = Ridge(alpha=0.5)
                grid_search = GridSearchCV(estimator=AdaBoostRegressor(Ridge1), param_grid=grid2, n_jobs=-1, scoring="neg_mean_absolute_error",cv=kfold, verbose=3)
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
                max_features_range = np.arange(1, 6, 1)
                n_estimators_range = np.arange(10, 210, 10)
                param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)
                kfold = KFold(n_splits=5, shuffle=True, random_state=2)
                grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kfold, scoring="neg_mean_absolute_error",n_jobs=-1, verbose=3)
                grid_result = grid.fit(x_train, y_train)

                GridSearch_table_plot(grid_result, "n_estimators")
                # summarize the best score and configuration
                print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
                # summarize all scores that were evaluated
                #means = grid_result.cv_results_['mean_test_score']
                #stds = grid_result.cv_results_['std_test_score']
                #params = grid_result.cv_results_['params']
                #for mean, stdev, param in zip(means, stds, params):
                #print("%f (%f) with: %r" % (mean, stdev, param))
                #print("end of tuning for RandomForestRegressor")
                kfold = KFold(n_splits=10, shuffle=True, random_state=2)
                #n_estimators = 400, max_depth = 3, learning_rate = 0.01, colsample_bytree = 0.5
                #param_grid = {'max_depth': [3],
                #          'n_estimators': [400],
                #          'learning_rate': [0.01],
                #          'colsample_bytree': [0.5],
                #          'subsample':[0.6,0.7,0.8,0.9,1.0]}
                param_grid = {  'max_depth': [3,4,5,6,7,8,9,10],
                              'n_estimators':[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                              'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                              'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                               'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]  }
                model = XGBRegressor()
                kfold = KFold(n_splits=10, shuffle=True, random_state=2)
                grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold, scoring="neg_mean_absolute_error", verbose=2)
                grid_result = grid.fit(x_train, y_train)
                print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
                # summarize all scores that were evaluated
                means = grid_result.cv_results_['mean_test_score']
                stds = grid_result.cv_results_['std_test_score']
                params = grid_result.cv_results_['params']
                for mean, stdev, param in zip(means, stds, params):
                     print("%f (%f) with: %r" % (mean, stdev, param))



            #param_grid = {
            #    'alg__hidden_layer_sizes': [(5, 3,), (10, 5,), (25, 10,)],
            #    'alg__max_iter': [1000, 1500, 2000,2500,3000,3500, 4000]}

            #param_grid2 = {
            #    'alg__hidden_layer_sizes': [(5,3,),(10,5,),(25,10,)],
            #    'alg__max_iter': [1000,2000,4000],
            #    'alg__activation': [ 'relu'],
            #    'alg__alpha': [0.0001],
            #    'alg__learning_rate': [ 'adaptive'],
            #    'alg__learning_rate_init':[0.002],
            #}

            #hidden_layer_sizes = [(5, 3,), (10, 5,), (25, 10,)]
            #max_iter = [ 1000, 2000,3000, 4000]
            mlp_reg = MLPRegressor()
            kfold = KFold(n_splits=10, shuffle=True, random_state=2)
            pipeline_scaled = Pipeline([('scale',MinMaxScaler()),('alg',mlp_reg)])
            layers = []
            layers = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
            iters =  [1000, 1500, 1600, 1700, 1800, 1900,2000,2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900,3000,3100,3200,3300,3400,3500]
            rates = [0.001,0.002,0.0021,0.0022,0.0023,0.0024,0.0025,0.0026,0.0027,0.0028,0.0029,0.003,0.0031,0.0032,0.0033,0.0034,0.0035,0.0036,0.0037,0.0038,0.0039,0.004,0.0045,0.005,0.0055,0.006]
            #layers2 = [3,5,10,20]
            results = []
            for x in iters:
                print('iter is ',x)
                #for y in layers2:
                    #if x>y:
                param_grid = {
                        'alg__hidden_layer_sizes': [(15,3,)],
                        'alg__max_iter': [x],
                        'alg__learning_rate_init': [0.001,0.002,0.0021,0.0022,0.0023,0.0024,0.0025,0.0026,0.0027,0.0028,0.0029,0.003,0.0031,0.0032,0.0033,0.0034,0.0035,0.0036,0.0037,0.0038,0.0039,0.004,0.0045,0.005,0.0055,0.006],
                        'alg__learning_rate':['adaptive']
                        }
                grid = GridSearchCV(estimator = pipeline_scaled, param_grid = param_grid, n_jobs=-1, cv = kfold, scoring="neg_mean_absolute_error",verbose=3)
                grid_result = grid.fit(x_train,y_train)
                currentvalues = {'best_score':grid_result.best_score_, 'best_params':grid_result.best_params_}
                results.append(currentvalues)
            for j in range(len(gridresults)):
                print(results[j])


            grid = GridSearchCV(estimator=pipeline_scaled, param_grid=param_grid, n_jobs=-1, cv=kfold, verbose=3)
            print(pipeline_scaled.get_params().keys())
            grid_result=grid.fit(x_train,y_train)
            param1 = 'alg__hidden_layer_sizes'
            param2 = 'alg__max_iter'
            GridSearch_Contour_plot(grid_result, [param1, param2], param1, param2, True, False)
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            GridSearch_Contour_plot(grid_result, [param1, param2], param1, param2, False, True)



            hidden_layer_sizes = [(5,3,),(8,5),(10,5,),(25,10,)]
            max_iter = [800, 1000,2000,4000]
            GridSearch_table_plot(grid_result, "hidden_layer_sizes")
            #plot_grid_search(grid_result.cv_results_, hidden_layer_sizes, max_iter, 'Hidden Layer Sizes', 'Max Iter')
            #plot_grid_search(grid_result.cv_results_, hidden_layer_sizes, max_iter, 'Hidden Layer Sizes', 'Max Iter')
            changes = []
            changes.append('hidden_layer_sizes')
            changes.append('max_iter')
            #sklearneval.plot.grid_search(grid_result.cv_results_, change='hidden_layer_sizes',kind='bar',cmap='Reds')
            #sklearneval.plot.grid_search(grid_result.cv_results_, changes, kind='bar', cmap='Reds')
            #plt.show()
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
            sklearn.plot.grid_search(grid_result.cv_results, change = 'n_estimators', kind='bar')

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

            model = XGBRegressor( )

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

