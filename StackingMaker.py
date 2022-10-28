import sklearn
import time
import csv
import os
import os.path
import pandas as pd
from scipy.stats import norm, iqr, scoreatpercentile
import numpy as np

df= pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\OPTUNAHYPERPARAMETERS\model_ELNET050" + ".csv", ";")
df.drop(["mae"], axis = 1, inplace=True)
df.set_index("ID", drop=True, inplace=True)
dictionary = df.to_dict(orient="index")
print("done")