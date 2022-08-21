import sklearn
import csv
import os
import os.path
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer, KNNImputer

def main():
    df = pd.read_csv( r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\warpathmissing.csv", ";")
    dfcopy = df
    imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
    imputer2 = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
    iterimputer = IterativeImputer()
    knnimputer = KNNImputer()
    df_weight = pd.DataFrame(df['Weight_kg'])
    print("Sum of missing values in Weight ", df_weight.isnull().sum())
    imputer = imputer.fit(df_weight)
    dfweight = imputer.transform(df_weight)
    dfweight2 = pd.DataFrame(dfweight)
    print(dfweight2)
    print("Sum of missing values in Weight ", dfweight2.isnull().sum())

    df_height = pd.DataFrame(df['Height_cm'])
    print("Sum of missing values in Height ", df_height.isnull().sum())
    imputer = imputer.fit(df_height)
    dfheight = imputer.transform(df_height)
    dfheight2 = pd.DataFrame(dfheight)
    print(dfheight2)
    print("Sum of missing values in Height ", dfheight2.isnull().sum())

    df_smoker = pd.DataFrame(df['Smoking_status'])
    print("Sum of missing values in Smoker_status ", df_smoker.isnull().sum())
    imputer = imputer2.fit(df_smoker)
    dfsmoker = imputer2.transform(df_smoker)
    dfsmoker2 = pd.DataFrame(dfsmoker)
    print(dfsmoker2)
    print("Sum of missing values in Smoker_status ",dfsmoker2.isnull().sum())

    df['Weight_kg']=dfweight2[0]
    df['Height_cm']=dfheight2[0]
    df['Smoking_status']=dfsmoker2[0]

    df['Weight_kg'] = dfweight2[0]
    df['Height_cm'] = dfheight2[0]
    df['Smoking_status'] = dfsmoker2[0]
    filename="Warpath_SimpleMeanImputed"
    df.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\\" + filename + ".csv", ";")

    df = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\warpathmissing.csv", ";")
    imputer = SimpleImputer(strategy='median', missing_values=np.nan)
    imputer2 = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
    df_weight = pd.DataFrame(df['Weight_kg'])
    print("Sum of missing values in Weight ", df_weight.isnull().sum())
    imputer = imputer.fit(df_weight)
    dfweight = imputer.transform(df_weight)
    dfweight2 = pd.DataFrame(dfweight)
    print(dfweight2)
    print("Sum of missing values in Weight ", dfweight2.isnull().sum())

    df_height = pd.DataFrame(df['Height_cm'])
    print("Sum of missing values in Height ", df_height.isnull().sum())
    imputer = imputer.fit(df_height)
    dfheight = imputer.transform(df_height)
    dfheight2 = pd.DataFrame(dfheight)
    print(dfheight2)
    print("Sum of missing values in Height ", dfheight2.isnull().sum())

    df_smoker = pd.DataFrame(df['Smoking_status'])
    print("Sum of missing values in Smoker_status ", df_smoker.isnull().sum())
    imputer = imputer2.fit(df_smoker)
    dfsmoker = imputer2.transform(df_smoker)
    dfsmoker2 = pd.DataFrame(dfsmoker)
    print(dfsmoker2)
    print("Sum of missing values in Smoker_status ", dfsmoker2.isnull().sum())

    df['Weight_kg'] = dfweight2[0]
    df['Height_cm'] = dfheight2[0]
    df['Smoking_status'] = dfsmoker2[0]
    filename = "Warpath_SimpleMedianImputed"
    df.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\\" + filename + ".csv", ";")

    #filename2 = "WarpathIterativeImputed"
    filename3 = "WarpathKNNImputed"
    #iterimputer = IterativeImputer()
    knnimputer = KNNImputer()
    target_column = 'Dose_mg_week'
    X = dfcopy.drop([target_column], axis=1)
    df_smoker = pd.DataFrame(X['Smoking_status'])
    imputer = imputer2.fit(df_smoker)
    dfsmoker = imputer.transform(df_smoker)
    dfsmoker = pd.DataFrame(dfsmoker)
    X.drop(['Country_recruitment', 'Gender', 'Target_INR', 'Indication', 'Amiodarone_status', 'HIV_status',
            'Inducer_status', 'Smoking_status'], axis=1, inplace=True)
    #df2 = iterimputer.fit_transform(X)
    df3 = knnimputer.fit_transform(X)
    #df2 = pd.DataFrame(df2)
    df3 = pd.DataFrame(df3)
    #df['Age_years'] = df2[0]
    #df['Weight_kg'] = df2[1]
    #df['Height_cm'] = df2[2]
    #df['Smoking_status'] = dfsmoker[0]
    #print(df)
    #filename2 = "WarpathIterativeImputed"
    #df.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\\" + filename2 + ".csv", ";")

    #df3 = pd.DataFrame(df3)
    df['Age_years'] = df3[0]
    df['Weight_kg'] = df3[1]
    df['Height_cm'] = df3[2]
    df['Smoking_status'] = dfsmoker[0]
    print(df)
    filename3 = "WarpathKNNImputed"
    df.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\\" + filename3 + ".csv", ";")








if __name__ == "__main__":
    main()