"""Calibration experiment on the (binarized) KITTI benchmark dataset."""
if __name__ == "__main__":

    import os
    import numpy as np
    import pandas as pd
    import time

    import xgboost as xgb
    # Install latest version of scikit-garden from github to enable partial_fit(X, y):
    # (https://github.com/scikit-garden/scikit-garden)
    #from skgarden import MondrianForestClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    import pycalib.benchmark
    import pycalib.calibration_methods as calm
    from tqdm.auto import tqdm

    ###############################
    #   Generate calibration data
    ###############################

    # Setup
    #classify_images = True
    classify_images = False
    file = os.path.dirname(os.path.realpath(__file__))
    if os.path.basename(os.path.normpath(file)) == "pycalib":
        file += "/datasets/kitti/"
    else:
        file = os.path.split(os.path.split(file)[0])[0] + "/pycalib/datasets/kitti/"
    output_folder = "clf_output"
    print(file)

    # Classify KITTI validation data with selected classifiers
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
        for clf_name, clf in clf_dict.items():
            pycalib.benchmark.KITTIBinaryData.classify_val_data(file, clf_name=clf_name, classifier=clf,
                                                                data_folder="kitti_features", output_folder=output_folder)

    ###############################
    #   Benchmark
    ###############################

    ## Initialization
    clf_output_dir = os.path.join(file, output_folder)
    run_dir = os.path.join(file, "calibration")

    ## Classifiers
    classifier_names = list(clf_dict.keys())

    #n_samples = [10, 50, 100, 250, 500, 1000, 3000, 5000]
    n_samples = [50, 100, 250, 500, 1000, 3000, 5000]
    for ns in tqdm(n_samples):
        train_size = ns
        #test_size = 8000 - ns
        test_size = 4000

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
        kitti_benchmark = pycalib.benchmark.KITTIBinaryData(run_dir=run_dir, clf_output_dir=clf_output_dir,
                                                            classifier_names=classifier_names,
                                                            cal_methods=list(cal_methods.values()),
                                                            cal_method_names=list(cal_methods.keys()),
                                                            n_splits=10, test_size=test_size,
                                                            train_size=train_size, random_state=random_state)
        
        ## TODO: We want to implement the experiments that uses all training dataset as a calibration data.
        ## NOW: We can set the ratio of the training dataset for preparing the calibraiton dataset. Thus, when we set "train_size=1.", the above is achieved.

        # Run
        kitti_benchmark.run_kernel(n_jobs=1)
    
    train_size = 5000
    test_size = 4000
    #reg_candidates = np.logspace(-4, 0, num=10)  # e.g., candidates from 1e-4 to 1e0
    reg_candidates = np.array([0.1, 1.])
    scores = pd.DataFrame()
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
        kitti_benchmark = pycalib.benchmark.KITTIBinaryData(run_dir=run_dir, clf_output_dir=clf_output_dir,
                                                            classifier_names=classifier_names,
                                                            cal_methods=list(cal_methods.values()),
                                                            cal_method_names=list(cal_methods.keys()),
                                                            n_splits=10, test_size=test_size,
                                                            train_size=train_size, random_state=random_state)
        
        ## TODO: We want to implement the experiments that uses all training dataset as a calibration data.
        ## NOW: We can set the ratio of the training dataset for preparing the calibraiton dataset. Thus, when we set "train_size=1.", the above is achieved.

        # Run
        kitti_benchmark.run_kernel(n_jobs=1, reg_sensitive=True)
        scores = pd.concat([scores, kitti_benchmark.results], axis=0)
    scores.to_csv(path_or_buf=os.path.join(kitti_benchmark.run_dir, "cv_scores_total_reg_{}.csv".format(time.strftime("%Y%m%d-%Hh%Mm%Ss"))))
    