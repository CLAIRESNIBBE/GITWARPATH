import numpy as np
import pandas as pd
import csv
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from matplotlib import pyplot as plt
import seaborn as sns

#pd.set_option("display.max_rows", None, "display.max_columns", None)
#pd.set_option('expand_frame_repr', False)
#pd.set_option("display.max_rows", True)
dat = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\ExampleFile.csv",";")
dat['Dose_mg_week']=dat['Dose_mg_week'].apply(np.sqrt)
results = smf.ols('Dose_mg_week ~ Target_INR + Weight_kg + AgeYears + HIVPositive + HIVUnknown',data = dat).fit()
print(results.summary())
print("Residuals")
residuals = results.resid
print(results.resid)
fittedvals = results.fittedvalues
print("Fitted Values")
print(results.fittedvalues)
fig, ax = plt.subplots(figsize=(6,2.5))
_ = ax.scatter(residuals, fittedvals)
#sms.linear_harvey_collier(results)
#define figure size
fig = plt.figure(figsize=(12,8))
#produce regression plots
fig = sm.graphics.plot_regress_exog(results, 'Weight_kg', fig=fig)

