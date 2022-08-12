import numpy as np
import pandas as pd
import csv
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from matplotlib import pyplot as plt
import seaborn as sns
import scipy as sp
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression, Ridge

dat = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\PreProcessed\dfWarfarin001.csv",";")
xvalues=dat.drop("Dose_mg_week",axis=1, inplace = True)
model = LinearRegression()
modelfit = model.fit()