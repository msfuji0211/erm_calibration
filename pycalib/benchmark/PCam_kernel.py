"""Calibration experiment on the PCam benchmark dataset."""

if __name__ == "__main__":

    import os
    import numpy as np
    import time
    import pandas as pd

    import xgboost as xgb
    # Install latest version of scikit-garden from github to enable partial_fit(X, y):
    # (https://github.com/scikit-garden/scikit-garden)
    #from skgarden import MondrianForestClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import AdaBoostClassifier
    import pycalib.benchmark
    import pycalib.calibration_methods as calm
    from tqdm.auto import tqdm

    ###############################
    #   Generate calibration data
    ###############################

    # Setup
    #classify_images = False
    classify_images = True
    file = os.path.dirname(os.path.realpath(__file__))
    if os.path.basename(os.path.normpath(file)) == "pycalib":
        file += "/datasets/pcam/"
    else:
        file = os.path.split(os.path.split(file)[0])[0] + "/pycalib/datasets/pcam/"
    output_folder = "clf_output"
    data_folder = "data"
    clf_output_dir = os.path.join(file, "clf_output")
    print(file)

    # Classify PCam validation data with selected classifiers
    random_state = 1

    clf_dict = {
        "XGBoost": xgb.XGBClassifier(booster="gbtree",
                                     n_estimators=100,
                                     random_state=random_state,
                                     n_jobs=-1),
        "random_forest": RandomForestClassifier(n_estimators=100,
                                                criterion="gini",
                                                min_samples_split=2,
                                                bootstrap=True,
                                                n_jobs=-1,
                                                random_state=random_state),
        "1layer_NN": MLPClassifier(hidden_layer_sizes=(100,),
                                   activation="relu",
                                   solver="adam",
                                   random_state=random_state)
    }

    if classify_images:
        for clf_name, clf in tqdm(clf_dict.items()):
            pycalib.benchmark.PCamData.classify_val_data(file, clf_name=clf_name, classifier=clf,
                                                         data_folder=data_folder, output_folder=output_folder)

    ###############################
    #   Benchmark
    ###############################

    # Initialization
    run_dir = os.path.join(file, "calibration")

    # Classifiers
    classifier_names = list(clf_dict.keys())

    #n_samples = [10, 50, 100, 250, 500, 1000, 3000, 5000]
    n_samples = [50, 100, 250, 500, 1000, 3000, 5000]
    for ns in tqdm(n_samples):
        train_size = ns
        #test_size = 9000 - ns
        test_size = 5000

        # Calibration methods
        cal_methods = {
            "Uncal": calm.NoCalibration(),
            "KRR_gauss": calm.KernelRidgeCalibration(kernel='rbf'),
            "KRR_laplace": calm.KernelRidgeCalibration(kernel='laplace'),
            "KRR_gauss_regopt": calm.KernelRidgeCalibration(kernel='rbf', alpha=((ns ** 0.5)**-1)),
            "KRR_laplace_regopt": calm.KernelRidgeCalibration(kernel='laplace', alpha=((ns ** (1/3))**-1)),
            "KLR_gauss": calm.KernelLogisticRegressionCalibration(kernel='rbf'),
            "KLR_laplace": calm.KernelLogisticRegressionCalibration(kernel='laplace'),
        }


        # Create benchmark object
        pcam_benchmark = pycalib.benchmark.PCamData(run_dir=run_dir, clf_output_dir=clf_output_dir,
                                                    classifier_names=classifier_names,
                                                    cal_methods=list(cal_methods.values()),
                                                    cal_method_names=list(cal_methods.keys()),
                                                    n_splits=10, test_size=test_size,
                                                    train_size=train_size, random_state=random_state)

        # Run
        pcam_benchmark.run_kernel(n_jobs=1)
    
    
    train_size = 5000
    test_size = 5000
    reg_candidates = np.logspace(-4, 0, num=10)  # 例: 1e-4 ～ 1e2 の候補
    
    for reg in tqdm(reg_candidates):

        # Calibration methods
        cal_methods = {
            "Uncal": calm.NoCalibration(),
            "KRR_gauss": calm.KernelRidgeCalibration(kernel='rbf', alpha=reg),
            "KRR_laplace": calm.KernelRidgeCalibration(kernel='laplace', alpha=reg),
            "KLR_gauss": calm.KernelLogisticRegressionCalibration(kernel='rbf', alpha=reg),
            "KLR_laplace": calm.KernelLogisticRegressionCalibration(kernel='laplace', alpha=reg),
        }

        # Create benchmark object
        pcam_benchmark = pycalib.benchmark.PCamData(run_dir=run_dir, clf_output_dir=clf_output_dir,
                                                    classifier_names=classifier_names,
                                                    cal_methods=list(cal_methods.values()),
                                                    cal_method_names=list(cal_methods.keys()),
                                                    n_splits=10, test_size=test_size,
                                                    train_size=train_size, random_state=random_state)
        
        ## TODO: We want to implement the experiments that uses all training dataset as a calibration data.
        ## NOW: We can set the ratio of the training dataset for preparing the calibraiton dataset. Thus, when we set "train_size=1.", the above is achieved.

        # Run
        pcam_benchmark.run_kernel(n_jobs=1, reg_sensitive=True)
        scores = pd.concat([scores, pcam_benchmark.results], axis=0)
    scores.to_csv(path_or_buf=os.path.join(pcam_benchmark.run_dir, "cv_scores_total_reg_{}.csv".format(time.strftime("%Y%m%d-%Hh%Mm%Ss"))))
    
