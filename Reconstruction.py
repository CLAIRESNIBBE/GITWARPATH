import pandas as pd
import numpy as np
import sklearn
import xgboost
from warfit_learn.estimators import Estimator
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoLarsCV, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler, FunctionTransformer, Normalizer
from xgboost import XGBRegressor
def ExitSquareBracket(variable):
    stringvar = str(variable)
    if stringvar.find('[') >= 0 and stringvar.find(']') >= 0:
        var1 = stringvar.replace('[', '')
        var2 = var1.replace(']', '')
        var2 = var2.replace("'", "")
        return var2
    else:
        return stringvar


def variance(metric):
    meanvalue = np.mean(metric)
    sumsquares = 0
    for i in range(len(metric)):
        core = abs(metric[i] - meanvalue)
        sumsquares += np.square(core)
    variance = sumsquares / ((len(metric) - 1))
    return variance


def confintlimit95(metric):
    return 1.96 * np.sqrt(variance(metric ) / len(metric))

metric_columns = ['MAE', 'PW20', 'R2', 'Time']
estimates = []
#RF = RandomForestRegressor(max_depth=120, max_features=3, min_samples_leaf=4,
#                                       min_samples_split=12, n_estimators=100)

LR = LinearRegression()
estimates.append(Estimator(LR,'LR'))
MLPR = MLPRegressor()
estimates.append(Estimator(MLPR, 'MLPR'))


#estimates.append(Estimator(pipeline_KNNR_scaled, 'KNN'))

dfResults = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfResults.csv",";")

#dfSummary = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfSummary" + ".csv", ";")
dfSummary = dfResults.groupby('Estimator').apply(np.mean)
stddev = []
confinterval = []
for i in range(len(metric_columns)):
    for _, est in enumerate(estimates):
        current_estimator = est.identifier
        current_metric = metric_columns[i]
        current_mean = dfSummary.loc[current_estimator][current_metric]
        metric_values = np.where(dfResults['Estimator'] == current_estimator, dfResults[current_metric], 9999)
        metriclist = np.array(metric_values)
        metriclist = [j for j in metriclist if j != 9999]
        current_stddev = confintlimit95(metriclist)
        confinterval.append({'estimator': current_estimator, 'metric': current_metric, 'mean': current_mean,
                             '95% CI lower bound': current_mean - current_stddev,
                             '95% CI upper bound': current_mean + current_stddev})
dfConfidence = pd.DataFrame(confinterval,
              columns=['estimator', 'metric', 'mean', '95% CI lower bound', '95% CI upper bound'])
dfConfidence.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfConfidence" + ".csv", ";")
#dfResults.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfResults" + ".csv", ";")
#dfSummary.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfSummary" + ".csv", ";")
print("STOP HERE")
