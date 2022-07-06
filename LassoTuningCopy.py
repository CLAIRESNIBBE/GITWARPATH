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
from random import sample
from random import choices
from matplotlib import pyplot as plt
from IPython.display import display
from sklearn_evaluation.plot import grid_search
from yellowbrick.features import ParallelCoordinates
from yellowbrick.regressor import AlphaSelection, PredictionError, ResidualsPlot
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

# from cubist import Cubist

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
    if stringvar.find('[') >= 0 and stringvar.find(']') >= 0:
        var1 = stringvar.replace('[', '')
        var2 = var1.replace(']', '')
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


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy


def traineval(est: Estimator, xtrain, ytrain, xtest, ytest, squaring):
    resultsdict = {'PW20': 0, 'MAE': 0, 'R2': 0}
    print(f'\n{est.identifier}...')
    mae_scorer = make_scorer(MAEScore)
    kcv = KFold(n_splits=10, random_state=1, shuffle=True)
    # ridinitial = grid[est.identifier]
    ytest_numpy = np.array(ytest)
    model = est.estimator
    fitted = model.fit(xtrain, ytrain)
    predict = fitted.predict(xtest)
    # search = GridSearchCV(est.estimator, gridinitial, scoring='neg_mean_absolute_error',cv=cv)
    # gridresult= search.fit(xtrain, ytrain)
    # redicts = search.best_estimator_.predict(xtest)
    if squaring:
        ytest = np.square(ytest)
        predict = np.square(predict)
    PW20 = PercIn20(ytest, predict)
    MAE = mean_absolute_error(ytest, predict)
    R2 = RSquared(ytest, predict)
    results2 = cross_val_score(model, xtrain, ytrain, cv=kcv, scoring='neg_mean_absolute_error')
    # print("Accuracy: %.3f%% (%.3f%%)" % (results2.mean() * 100.0, results2.std() * 100.0))
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
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2), len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2), len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1, 1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx, :], '-o', label=name_param_2 + ': ' + str(val))

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


# plot_grid_search(pipe_grid.cv_results_, n_estimators, max_features, 'N Estimators', 'Max Features')

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
    dfImputedList = []
    combinedata = False
    # sklearn.externals.joblib.load('gridsearch.pkl')
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
            IWPCDF = pd.read_csv(IWPC_csv, ';')
            sampleSize = int(round(trainSize) * 0.25)
            dfIWPC = IWPCDF.sample(n=sampleSize)
            dfIWPC["Status"] = "train"
            dfIWPC.drop(["Unnamed: 0"], axis=1, inplace=True)
        df = fileindex + 1
        dfmod = dfnew
        dfmod.drop(['Gender', 'Country_recruitment'], axis=1, inplace=True)
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
        dfImputedList.append(dfmod)

    print("Ready to bootstrap")
    matrix = np.zeros((364,n_bootstraps))
    print(matrix)
    patientlist = list(np.arange(1, 364 + 1))
    patient_sample = choices(patientlist,k=364)
    for boot in range(n_bootstraps):
      patient_sample= choices(patientlist,k=364)
      row = 0
      for j in range(len(patient_sample)):
         matrix[row,boot]=patient_sample[j]
         row = row + 1

    print(matrix)

    dfMatrix = pd.DataFrame(matrix)
    for boot in range(n_bootstraps):
        patientlist = dfMatrix[boot]
        for imp in range(impNumber):
            suffix = str(imp + 1).zfill(3)
            filename = "dfWarfarin" + suffix
            dfImp = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\PreProcessed\\" + filename + ".csv")
            dfImparray = dfImp.to_numpy()
            bootstrap=[]
            for i in range(len(patientlist)):
                for j in range(len(dfImparray)):
                    if (j == patientlist[i]):
                       bootstrap.append(dfImparray[j])
            bootcurrent = bootstrap
            if (imp==0):
                trainboot = pd.DataFrame(bootcurrent)
            else:
                bootcurrent = pd.DataFrame(bootcurrent)
                frames = (trainboot,bootcurrent)
                trainboot = pd.concat(frames)

    trainboot.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\ImputedforTuning" + ".csv", ";")




    #['C:\\Users\\Claire\\GIT_REPO_1\\CSCthesisPY\\WarImputations\\Split\\ImpWarPATHSPLIT_001.csv',
    # 'C:\\Users\\Claire\\GIT_REPO_1\\CSCthesisPY\\WarImputations\\Split\\ImpWarPATHSPLIT_002.csv',
    # 'C:\\Users\\Claire\\GIT_REPO_1\\CSCthesisPY\\WarImputations\\Split\\ImpWarPATHSPLIT_003.csv']

    for boot in range(n_bootstraps):
        for imp in range(impNumber):
            suffix = str(imp+1).zfill(3)
            filename = "dfWarfarin"+suffix
            dbfile = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\PreProcessed\\" + filename + ".csv")
            databoot = matrix[imp][boot]
            dbdataboot = pd.DataFrame(databoot)
            if imp==1:
                trainboot = databoot
            else:
                frames=[trainboot,databoot]
                trainboot=pd.concat(frames)





if __name__ == "__main__":
    main()

