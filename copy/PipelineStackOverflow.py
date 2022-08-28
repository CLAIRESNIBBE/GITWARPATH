import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

data = [[100, 1, 50],[500 , 3, 25],[1000 , 10, 100]]
df = pd.DataFrame(data,columns=['Cost','People', 'Supplies'])

X = df[['People', 'Supplies']]
y = df[['Cost']]

#Split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

#Pipeline
pipeline_scaled = Pipeline([('scale', RobustScaler()),
                     ('alg', Lasso(random_state=0))])

pipeline_unscaled = Pipeline([('alg', Lasso(random_state=0))])

clf1 = pipeline_scaled.fit(X_train,y_train)
clf2 = pipeline_unscaled.fit(X_train,y_train)

#Pipeline predict example
example = [[10,100]]
print('Pipe Scaled: ', clf1.predict(example))
print('Pipe Unscaled: ',clf2.predict(example))

#------------------------------------------------

rs = RobustScaler()
reg = Lasso(random_state=0)
# Scale the taining data
X_train_scaled = rs.fit_transform(X_train)
reg.fit(X_train_scaled, y_train)
# Scale the example
example_scaled = rs.transform(example)
# Predict using the scaled data
print('----------------------')
print('Reg Scaled: ', reg.predict(example_scaled))
print('Scaled Coefficients:',reg.coef_)

#------------------------------------------------
reg.fit(X_train, y_train)
print('Reg Unscaled: ', reg.predict(example))
print('Unscaled Coefficients:',reg.coef_)