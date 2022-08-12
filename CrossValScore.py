from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from numpy import mean
from numpy import absolute
from numpy import sqrt
import pandas as pd
import numpy as np

def MAEScore(true, predicted):
    return mean_absolute_error(true,predicted)

def PercIn20(true, predicted):
    patients_in_20 = 0
    for i in range(len(true)):
        if abs(true[i] - predicted[i]) < 0.2 * true[i]:
            patients_in_20 += 1
    return 100 * patients_in_20 / len(true)

def RSquared(trueval, predval):
    true_mean = np.mean(trueval)
    topsum = 0
    lowersum = 0
    for i in range(len(trueval)):
        topsum += np.square((predval[i] - true_mean))
        lowersum += np.square(trueval[i] - true_mean)
    return topsum / lowersum * 100


df = pd.DataFrame({'y': [6, 8, 12, 14, 14, 15, 17, 22, 24, 23],
                   'x1': [2, 5, 4, 3, 4, 6, 7, 5, 8, 9],
                   'x2': [14, 12, 12, 13, 7, 8, 7, 4, 6, 5]})
#define predictor and response variables
X = df[['x1', 'x2']]
y = df['y']

#define cross-validation method to use
cv = KFold(n_splits=10, random_state=1, shuffle=True)

#build multiple linear regression model
model = LinearRegression()
#use k-fold CV to evaluate model
maescorer = make_scorer(MAEScore)
pw20scorer = make_scorer(PercIn20)
custom_scorer = {'mae': make_scorer(MAEScore),
                 'pw20': make_scorer(PercIn20),
                 'rsquared': make_scorer(RSquared),
                }
scores = cross_validate(model, X, y, cv = cv, scoring = maescorer,verbose=2)
print(scores)
#view mean absolute error
#print(mean(absolute(scores)))

