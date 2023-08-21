import sklearn
import time
import csv
import os
import os.path
import pandas as pd
from scipy.stats import norm, iqr, scoreatpercentile
import numpy as np


dfResults = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\MAE_MLPR" + ".csv", ";")
#fResults.drop(["Unnamed: 0"], axis=1, inplace=True)
#dfSummary = dfResults.groupby('Estimator').apply(np.nanmean)
MLPR_rows = dfResults.loc[dfResults['Estimator']=='MLPR']
if len(MLPR_rows) > 0:
  MAElist = MLPR_rows['MAE'].tolist()
  maxlimit = iqr(MAElist)+1.5*scoreatpercentile(MAElist,75)
  newMAElist = [x for x in MAElist if x <= maxlimit]
  newMeanMae = np.mean(newMAElist)
  for i in dfResults.index:
    if dfResults.loc[i,'MAE'] > maxlimit:
      dfResults.at[i,'MAE'] = np.nan
      dfResults.at[i,'PW20'] = np.nan
      dfResults.at[i,'R2'] =np.nan
      dfResults.at[i,'Time'] = np.nan
  #dfResults.loc[dfResults['Estimator']=='MLPR']['MAE']=np.where(dfResults.loc[dfResults['Estimator']=='MLPR']['MAE'] > maxlimit,np.nan,dfResults.loc[dfResults['Estimator']=='MLPR']['MAE'])
  dfSummary = dfResults.groupby('Estimator').apply(np.mean)
  print('DONE PART 1')



