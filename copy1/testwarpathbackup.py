if False:
    for file in filesImp:
        dfnew = pd.read_csv(root + '\\' + file, ";")
        fileindex = filesImp.index(file)
        rootIWPC = root.replace("WarImputations", "MICESTATSMODELHIV\\")
        # rootIWPC = root.replace("WarImputations","MICESTATS125\\")
        IWPC_csv = rootIWPC + filesIWPC[fileindex]
    dfIWPC = pd.read_csv(IWPC_csv, ';')
    df = fileindex + 1
    dfmod = dfnew
    dfmod.drop(['Gender', 'Country_recruitment'], axis=1, inplace=True)
    # dfIWPC.drop(['AgeDecades'], axis =1, inplace = True)
    # dfIWPC.drop(['INR_Three'], axis=1, inplace=True)
    # dfmod["INR_Three"] = np.where(dfmod["Target_INR"] == "Three", 1, 0)
    dfmod["Target_INR"] = np.where(dfmod["Target_INR"] == "Three", 3.0,
                                   np.where(dfmod["Target_INR"] == "aTwo_point_five", 2.5, 2.0))
    dfmod["Target_INR"] = dfmod['Target_INR'].astype("float")
    dfmod["Inducer"] = dfmod.apply(lambda x: ConvertYesNo(x["Inducer_status"]), axis=1)
    dfmod["Amiodarone"] = dfmod.apply(lambda x: ConvertYesNo(x["Amiodarone_status"]), axis=1)
    dfmod["Smoker"] = dfmod.apply(lambda x: ConvertYesNo(x["Smoking_status"]), axis=1)
    dfmod["Indicationflag"] = dfmod.apply(lambda x: ConvertYesNo(x["Indication"]), axis=1)
    dfmod.drop(["Inducer_status", "Amiodarone_status", "Smoking_status", "Indication"], axis=1, inplace=True)
    # dfmod["AgeDecades"] = np.floor(dfmod["Age_years"] * 0.1).astype("int")
    dfmod["AgeYears"] = dfmod["Age_years"]
    dfmod['AgeYears'] = np.where((dfmod['AgeYears'] <= 18), 18, dfmod['AgeYears'])
    dfmod["HIVPositive"] = np.where(dfmod["HIV_status"] == "Positive", 1, 0)
    dfmod["HIVUnknown"] = np.where(dfmod["HIV_status"] == "Unknown", 1, 0)
    # dfIWPC["HIVPositive"]=0
    # dfIWPC["HIVUnknown"] = 0
    dfmod.drop(["HIV_status"], axis=1, inplace=True)
    # dfmod['BSA'] = dfmod.apply(lambda x: BSA(x["Height_cm"], x["Weight_kg"]), axis=1)
    # dfmod.drop(["Height_cm"], axis = 1, inplace = True)
    # dfmod.drop(["Weight_kg"], axis=1, inplace=True)
    dfmod.drop(["Unnamed: 0"], axis=1, inplace=True)
    dfIWPC.drop(["Unnamed: 0"], axis=1, inplace=True)
    dfmod.drop(["Unnamed: 0.1"], axis=1, inplace=True)
    dfmod.drop(["Age_years"], axis=1, inplace=True)
    dfmod.drop([".imp"], axis=1, inplace=True)
    # dfmod.drop([".id"], axis=1, inplace=True)
    # combinedata = True
    if combinedata == True:
        dfmod = dfmod.sample(frac=1)
        dfIWPC = dfIWPC.sample(frac=1)
        frames = [dfmod, dfIWPC]
        dfmod = pd.concat(frames)
        dfmod = dfmod.sample(frac=1)
        suffix = str(df).zfill(3)
        combfilename = "comb" + suffix
        dfmod.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\combinedata\\" + combfilename + ".csv", ";")
    if True:
        print("On imputation ", df)
        data = dfmod
        print(data.shape)
        test_size = 0.2
        data['Dose_mg_week'] = data['Dose_mg_week'].apply(np.sqrt)
        estimates = []
        target_column = 'Dose_mg_week'
        train = dfmod.loc[dfmod["Status"] == train]
        test = dfmod.loc[dfmod["Status"] == test]
        # train, test = train_test_split(data, test_size=test_size)
        y_train = train[target_column].values
        x_train = train.drop([target_column], axis=1).values
        y_test = test[target_column].values
        x_test = test.drop([target_column], axis=1).values
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        y_train = scaler.fit_transform(y_train)
        LR = LinearRegression()
        RR = Ridge(alpha=0.02, solver="lsqr", fit_intercept=True)
        LAS = Lasso(alpha=0.002)
        EL = ElasticNet(alpha=0.01, l1_ratio=0.01)
        estimates.append(Estimator(LR, 'LR'))
        results = {'PW20': [], 'MAE': [], 'R2': []}
        res = []
        resamples = 1
        for i in range(resamples):
            r = traineval(
                estimates,
                squaring=squaring,
            )
            res.append(r)

        results['PW20'] = [r[0] for r in res]
        results['MAE'] = [r[1] for r in res]
        results['R2'] = [r[2] for r in res]

        ab_EL = AdaBoostRegressor(EL, n_estimators=400, random_state=7)
        SGD = SGDRegressor(penalty="l2")
        KNN = KNeighborsRegressor(weights="uniform", p=1, n_neighbors=14, algorithm="brute")
        NN = MLPRegressor(hidden_layer_sizes=(100,), activation="relu", random_state=1, max_iter=2000)
        # NN = MLPRegressor(hidden_layer_sizes=(100,), activation='logistic', solver='lbfgs', max_iter=1000)
        # SV = SVR(kernel='linear', cache_size=1000)
        # SVReg = SVR(epsilon=1.5, kernel='sigmoid',C=2.0)

        SVR = LinearSVR(C=9.59, epsilon=0.42, fit_intercept=True)
        # DTR = DecisionTreeRegressor(criterion="friedman_mse",max_depth=11, max_features='sqrt', max_leaf_nodes=40, min_impurity_decrease=0.8,min_samples_leaf=7,min_weight_fraction_leaf=0.1,splitter='best')
        DTR = DecisionTreeRegressor(max_depth=4)
        ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
        BRT = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100)
        XGBR = XGBRegressor(learning_rate=0.01, colsample_bytree=0.3, max_depth=3, n_estimators=500,
                            objective='reg:squarederror')
        RandomForestRegressor(max_depth=110, max_features=2, min_samples_leaf=4, min_samples_split=12)

        RF = RandomForestRegressor(max_features='sqrt', bootstrap=True, n_estimators=500, max_depth=10,
                                   min_samples_split=2, min_samples_leaf=5)
        ab_RF = AdaBoostRegressor(RF, n_estimators=400, random_state=7)
        CUBE = Cubist()
        XGB = XGBRegressor(max_depth=10, n_estimators=1000, min_child_weight=5, subsample=0.6,
                           alpha=0.1, eta=0.1, seed=42)
        n_trees = 100
        mdepth = 6
        gamma = 1
        lam = 1
        XG = XGBClassifier(
            learning_rate=0.1,
            n_estimators=1000,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softmax',
            nthread=4,
            scale_pos_weight=1,
            seed=27,
            num_class=3,
        )
        # BAG = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
        # XG = XGBClassifier(use_label_encoder=False,
        #                      booster='gbtree',  # boosting algorithm to use, default gbtree, othera: gblinear, dart
        #                      n_estimators=n_trees,  # number of trees, default = 100
        #                      eta=0.3,  # this is learning rate, default = 0.3
        #                      max_depth=mdepth,  # maximum depth of the tree, default = 6
        #                      gamma=gamma,
        #                      # used for pruning, if gain < gamma the branch will be pruned, default = 0
        #                      reg_lambda=lam,  # regularization parameter, defautl = 1
        #                      # min_child_weight=0 # this refers to Cover which is also responsible for pruning if not set to 0
        #                      )
        estimates.append(Estimator(LR, 'LR'))
        estimates.append(Estimator(BRT, 'BRT'))
        # estimates.append(Estimator(GBT, 'GBT'))
        estimates.append(Estimator(RF, 'RF'))
        estimates.append(Estimator(CUBE, 'Cubist'))
        estimates.append(Estimator(XG, 'XGB'))
        estimates.append(Estimator(NN, 'NN'))
        estimates.append(Estimator(RR, 'RR'))
        # estimates.append(Estimator(SV, 'SV'))
        estimates.append(Estimator(EL, 'EL'))
        # models = list()
        # models.append(('KNN', KNeighborsRegressor(weights="uniform", p=1, n_neighbors= 14,algorithm = "brute")))
        # models.append(('DTR', DecisionTreeRegressor(max_depth=4)))
        # models.append(('SVR', SVR(epsilon=1.5, kernel='sigmoid',C=2.0)))
        # scores = evaluate_models(models, x_train, x_test, y_train, y_test)
        # ensemble1 = VotingRegressor(estimators = models,weights = scores)
        estimates.append(Estimator(LinearSVR(epsilon=0.0, tol=0.0001, C=1.0, loss='epsilon_insensitive'), 'SVR'))
        estimates.append(
            Estimator(StackingCVRegressor(regressors=[SVR, KNN, BRT], meta_regressor=SVR, cv=5, ), 'Stacked_SVR'))
        estimates.append(
            Estimator(StackingCVRegressor(regressors=[XGB, SVR, NN], meta_regressor=SVR, cv=5, ), 'Stacked_SV'))
        # estimates.append(Estimator(BAG, 'Bag'))
        estimates.append(Estimator(LAS, 'Lasso'))
        tpot2 = make_pipeline(
            StackingEstimator(
                estimator=LinearSVR(
                    C=1.0,
                    dual=True,
                    epsilon=0.01,
                    loss="epsilon_insensitive",
                    tol=0.001, )),
            StackingEstimator(
                estimator=ElasticNetCV(l1_ratio=0.6000000000000001, tol=0.01, cv=5)),
            RobustScaler(),
            StackingEstimator(estimator=RidgeCV()),
            ExtraTreesRegressor(
                bootstrap=True,
                max_features=1.0,
                min_samples_leaf=20,
                min_samples_split=2,
                n_estimators=100, )
        )
        tpot10 = make_pipeline(
            StackingEstimator(estimator=ExtraTreesRegressor(
                bootstrap=True, max_features=0.05,
                min_samples_leaf=18, min_samples_split=10,
                n_estimators=100)),
            MaxAbsScaler(),
            StackingEstimator(estimator=ExtraTreesRegressor(
                bootstrap=True, max_features=0.05,
                min_samples_leaf=18, min_samples_split=10, n_estimators=100)),
            LassoLarsCV(normalize=True, cv=3)
        )
        tpot17 = make_pipeline(
            make_union(
                FunctionTransformer(copy, validate=True),
                MaxAbsScaler()
            ),
            StackingEstimator(estimator=RidgeCV()),
            ZeroCount(),
            GradientBoostingRegressor(alpha=0.9, learning_rate=0.1, loss="lad",
                                      max_depth=3, max_features=0.9000000000000001,
                                      n_estimators=100, subsample=0.55)
        )
        estimates.append(Estimator(tpot2, 'TPOT2'))
        estimates.append(Estimator(tpot10, 'TPOT10'))
        estimates.append(Estimator(tpot17, 'TPOT17'))
        estimates = []
        #
        # estimates.append(Estimator(DTR,'DTR'))
        # estimates.append(Estimator(ab_regressor,'ABDTR'))
        estimates.append(Estimator(LR, 'LR'))
        # estimates.append(Estimator(ab_RR,'ABRR'))
        # estimates.append(Estimator(ab_RF, 'ABRF'))
        # estimates.append(Estimator(ab_EL, 'ABEL'))
        # estimates.append(Estimator(ab_LAS, 'ABLasso'))
        # estimates.append(Estimator(XGBR, 'XGBR'))
        estimates.append(Estimator(RR, 'RR'))
        # estimates.append(Estimator(RF, 'RF'))
        # estimates.append(Estimator(EL, 'EL'))
        # estimates.append(Estimator(LAS, 'Lasso'))
        # estimates.append(Estimator(SGD, 'SGD'))
        # estimates.append(Estimator(SVReg, 'SVR'))
        # estimates.append(Estimator(NN, 'NN'))
        # estimates.append(Estimator(KNN,"KNN"))
        # estimates.append(Estimator(ensemble1,'Ensemble1')) #KNN,DTR,SVR

        warpath_results = evaluate_estimators(estimates,
                                              data,
                                              target_column='Dose_mg_week'
                                              , scale=True
                                              , test_size=0.1
                                              , squaring=True
                                              , technique='mccv'
                                              , parallelism=0.8
                                              )
        print(warpath_results)
        summary = warpath_results.groupby('Estimator').apply(np.mean)
        print(summary)
        dftemplate = dftemplate.append(summary)
        warpath_formatted = format_summary(warpath_results)
        dfWarPath = dfWarPath.append(warpath_results)
        df_final = pd.concat([warpath_formatted], axis=1, keys=['WARPATH'])
        print(df_final)
        suffix = str(df).zfill(3)
        df_final.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_" + suffix + ".csv", ";")
        if False:
            model.fit(X, y, epochs=150, batch_size=10, verbose=0)
            # make class predictions with the model
            predictions = (model.predict(X) > 0.5).astype(int)
            # summarize the first 50 cases
            for i in range(50):
                print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))

dftemplate.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dftemplate" + ".csv", ";")
dfSummary = dftemplate.groupby('Estimator').apply(np.mean)
dfSummary.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfSummary" + ".csv", ";")
dfWarPath_formatted = format_summary(dfWarPath)
dfWarPath_formatted.to_csv(r"C:\Users\Claire\GIT_REPO_1\CSCthesisPY\WARPATH_dfWarPath" + ".csv", ";")
print('the end')