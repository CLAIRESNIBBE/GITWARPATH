import pandas as pd
from sklearn.model_selection import train_test_split
from warfit_learn import evaluation
from warfit_learn.evaluation import evaluate_estimators
from warfit_learn.evaluation import _train_eval

def main():
    suffix = str(100).zfill(3)
    dftest = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\ImpWarPATH_TEST20_" + suffix + ".csv",";")
    dftrain = pd.read_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\ImpWarPATH_TRAIN80_" + suffix + ".csv",";")
    dftest.drop(["Unnamed: 0"], axis=1, inplace=True)
    dftrain.drop(["Unnamed: 0"], axis=1, inplace=True)
    dftest.drop(["Unnamed: 0.1"], axis=1, inplace=True)
    dftrain.drop(["Unnamed: 0.1"], axis=1, inplace=True)
    frames = [dftrain, dftest]
    dfcompare = pd.concat(frames)
    duplicate = dfcompare[dfcompare.duplicated()]
    print(" no of duplicates ", len(duplicate))
    dfcompare.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\ImpWarPATH_COMBINED_" + suffix + ".csv",
                          ";")
    if True:
        test_size = 0.2
        random_state = 42
        target_column = 'Dose_mg_week'
        resamples = 100
        squaring = True
        replace = False
        results = {'PW20': [], 'MAE': [], 'R2': []}

        res = []
        for i in range(resamples):
            r = _train_eval(
                estimator,
                data,
                target_column=target_column,
                test_size=test_size,
                squaring=squaring,
                replace=replace)
            res.append(r)

        results['PW20'] = [r[0] for r in res]
        results['MAE'] = [r[1] for r in res]
        results['R2'] = [r[2] for r in res]


        for i in range(resamples):
            train, test = train_test_split(data, random_state, test_size)
        y_train = train[target_column].values
        y_train = pd.DataFrame(y_train)
        x_train = train.drop([target_column], axis=1).values
        x_train = pd.DataFrame(x_train)
        y_test = test[target_column].values  # must isolate this and x_test to prevent data leakage.
        y_test = pd.DataFrame(y_test)
        x_test = test.drop([target_column], axis=1).values
        x_test = pd.DataFrame(x_test)
        suffix = str(df).zfill(3)
        y_test.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\ImpWarPATH_YTEST20_" + suffix + ".csv", ";")
        x_test.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\ImpWarPATH_XTEST20_" + suffix + ".csv", ";")
        y_train.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\ImpWarPATH_YTRAIN80_" + suffix + ".csv",";")
        x_train.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WarImputations\ImpWarPATH_XTRAIN80_" + suffix + ".csv",";")

        LR = LinearRegression()
        RR = Ridge(alpha=0.02, solver="lsqr", fit_intercept=True)
        LAS = Lasso(alpha=0.002)
        KNN = KNeighborsRegressor(weights="uniform", p=1, n_neighbors=14, algorithm="brute")
        RF = RandomForestRegressor(max_features='sqrt', bootstrap=True, n_estimators=500, max_depth=10,
                                   min_samples_split=2, min_samples_leaf=5)
        XGBR = XGBRegressor(learning_rate=0.01, colsample_bytree=0.3, max_depth=3, n_estimators=500,
                            objective='reg:squarederror')
        estimates = []
        estimates.append(Estimator(LR, 'LR'))
        estimates.append(Estimator(RR, 'RR'))
        estimates.append(Estimator(LAS, 'LASSO'))
        estimates.append(Estimator(XGBR, 'XGBR'))
        estimates.append(Estimator(RF, 'RF'))
        estimates.append(Estimator(KNN, 'KNN'))

        warpath_results = evaluate_estimators(estimates,
                                              data,
                                              target_column='Dose_mg_week'
                                              , scale=True
                                              , test_size=0.2
                                              , squaring=True
                                              , technique='mccv'
                                              , parallelism=0.8
                                              )
        print(warpath_results)
        summary = warpath_results.groupby('Estimator').apply(np.mean)
        print(summary)
        dftemplate = dftemplate.append(summary)
        warpath_formatted = format_summary(warpath_results)
        df_final = pd.concat([warpath_formatted], axis=1, keys=['WARPATH'])
        #print(df_final)

if __name__ == "__main__":
    main()