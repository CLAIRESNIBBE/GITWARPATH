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


def main():
   randomseed = 113
   dfResults = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfResultsPRE.csv", ";")
   if "Unnamed: 0" in dfResults.columns:
      dfResults.drop(["Unnamed: 0"], axis=1, inplace=True)
   dfSummary = dfResults.groupby('Estimator').apply(np.mean)
   dfSummary.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfSummary_" + str(randomseed) + ".csv", ";")

if __name__ == "__main__":
   main()