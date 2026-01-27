import os
import numpy as np
import torch
import pandas as pd
import argparse

from data import generate_data
from utils import select_sigma_from_data, select_sigma_from_vector
from kernels import gaussian_kernel, laplace_kernel
from model import PredictionModel, kernel_ridge_regression, kernel_logistic_regression, kernel_logistic_regression_gd
from metrics import classification_error, kernel_calibration_error, compute_LinECE, expected_calibration_error

# ------------------------------
# Experiment
# ------------------------------
def run_experiment(n_train=10000, n_test=10000, reg=1e-2, input_dim=1):
    X_all, y_all = generate_data(n_train + n_test, input_dim=input_dim)
    
    # data split (d-dimensional case, so unsqueeze is not needed)
    X_train = X_all[:n_train]   # shape: [n_train, d]
    y_train = y_all[:n_train]
    X_test = X_all[n_train:]    # shape: [n_test, d]
    y_test = y_all[n_train:]
    
    # model-specific sigma is selected from training data by median heuristic
    model_sigma = select_sigma_from_data(X_train)
    print(f"selected model-specific sigma: {model_sigma:.4f}")
    
    num_bins_test = int(n_test ** (1/3))
    num_bins_train = int(n_train ** (1/3))
    
    
    kernels = [("Gaussian", gaussian_kernel), ("Laplace", laplace_kernel)]
    
    results = []
    for kernel_name, kernel_func in kernels:
        print("\n==============================")
        print(f"model-specific kernel: {kernel_name}")
        print("==============================")
        
        # --- Kernel Ridge Regression (squared loss) ---
        alpha_krr = kernel_ridge_regression(X_train, y_train, kernel_func, model_sigma, reg)
        
        # evaluation on test data
        K_test_train = kernel_func(X_test, X_train, model_sigma)
        y_pred_reg_test = K_test_train @ alpha_krr
        prob_krr_test = torch.clamp(y_pred_reg_test, 0, 1)
        
        err_krr_test = classification_error(y_test, prob_krr_test)
        kce_sigma_krr_test = select_sigma_from_vector(prob_krr_test)
        kce_krr_gaussian_test = kernel_calibration_error(y_test, prob_krr_test, sigma=kce_sigma_krr_test, kernel_type='gaussian')
        kce_krr_laplace_test  = kernel_calibration_error(y_test, prob_krr_test, sigma=kce_sigma_krr_test, kernel_type='laplace')
        sce_krr_test = compute_LinECE(y_test, prob_krr_test)
        ece_krr_test = expected_calibration_error(y_test, prob_krr_test, num_bins_test)
        
        # evaluation on training data
        K_train_train = kernel_func(X_train, X_train, model_sigma)
        y_pred_reg_train = K_train_train @ alpha_krr
        prob_krr_train = torch.clamp(y_pred_reg_train, 0, 1)
        
        err_krr_train = classification_error(y_train, prob_krr_train)
        kce_sigma_krr_train = select_sigma_from_vector(prob_krr_train)
        kce_krr_gaussian_train = kernel_calibration_error(y_train, prob_krr_train, sigma=kce_sigma_krr_train, kernel_type='gaussian')
        kce_krr_laplace_train  = kernel_calibration_error(y_train, prob_krr_train, sigma=kce_sigma_krr_train, kernel_type='laplace')
        sce_krr_train = compute_LinECE(y_train, prob_krr_train)
        ece_krr_train = expected_calibration_error(y_train, prob_krr_train, num_bins_train)
        
        print("\n[Kernel Ridge Regression (squared loss)]")
        print("---- Test Data ----")
        print(f"Classification error: {err_krr_test:.4f}")
        print(f"KCE (Gaussian, sigma={kce_sigma_krr_test:.4f}): {kce_krr_gaussian_test.item():.4f}")
        print(f"KCE (Laplace,  sigma={kce_sigma_krr_test:.4f}): {kce_krr_laplace_test.item():.4f}")
        print(f"Smooth CE (SCE): {float(sce_krr_test):.4f}")
        print(f"Expected Calibration Error (binning): {float(ece_krr_test):.4f}")
        
        print("---- Train Data ----")
        print(f"Classification error: {err_krr_train:.4f}")
        print(f"KCE (Gaussian, sigma={kce_sigma_krr_train:.4f}): {kce_krr_gaussian_train.item():.4f}")
        print(f"KCE (Laplace,  sigma={kce_sigma_krr_train:.4f}): {kce_krr_laplace_train.item():.4f}")
        print(f"Smooth CE (SCE): {float(sce_krr_train):.4f}")
        print(f"Expected Calibration Error (binning): {float(ece_krr_train):.4f}")

        results.append({
            'model': 'KRR',
            'kernel': kernel_name,
            'train_size': n_train,
            'error_test': err_krr_test,
            'kce_gaussian_test': kce_krr_gaussian_test.item(),
            'kce_laplace_test': kce_krr_laplace_test.item(),
            'sce_test': float(sce_krr_test),
            'ece_test': float(ece_krr_test),
            'error_train': err_krr_train,
            'kce_gaussian_train': kce_krr_gaussian_train.item(),
            'kce_laplace_train': kce_krr_laplace_train.item(),
            'sce_train': float(sce_krr_train),
            'ece_train': float(ece_krr_train),
        })
        
        # --- Kernel Ridge Regression (squared loss) with theoretical reg ---
        if kernel_name == "Gaussian":
            reg_n = 1/(n_train ** 0.5)
        elif kernel_name == "Laplace":
            reg_n = 1/(n_train ** (1/3))
        else:
            raise ValueError("Unimplemented kernel. Use 'Gaussian' or 'Laplace'.")
            
        alpha_krr = kernel_ridge_regression(X_train, y_train, kernel_func, model_sigma, reg_n)
        
        # evaluation on test data
        K_test_train = kernel_func(X_test, X_train, model_sigma)
        y_pred_reg_test = K_test_train @ alpha_krr
        prob_krr_test = torch.clamp(y_pred_reg_test, 0, 1)
        
        err_krr_test = classification_error(y_test, prob_krr_test)
        kce_sigma_krr_test = select_sigma_from_vector(prob_krr_test)
        kce_krr_gaussian_test = kernel_calibration_error(y_test, prob_krr_test, sigma=kce_sigma_krr_test, kernel_type='gaussian')
        kce_krr_laplace_test  = kernel_calibration_error(y_test, prob_krr_test, sigma=kce_sigma_krr_test, kernel_type='laplace')
        sce_krr_test = compute_LinECE(y_test, prob_krr_test)
        ece_krr_test = expected_calibration_error(y_test, prob_krr_test, num_bins_test)
        
        # evaluation on training data
        K_train_train = kernel_func(X_train, X_train, model_sigma)
        y_pred_reg_train = K_train_train @ alpha_krr
        prob_krr_train = torch.clamp(y_pred_reg_train, 0, 1)
        
        err_krr_train = classification_error(y_train, prob_krr_train)
        kce_sigma_krr_train = select_sigma_from_vector(prob_krr_train)
        kce_krr_gaussian_train = kernel_calibration_error(y_train, prob_krr_train, sigma=kce_sigma_krr_train, kernel_type='gaussian')
        kce_krr_laplace_train  = kernel_calibration_error(y_train, prob_krr_train, sigma=kce_sigma_krr_train, kernel_type='laplace')
        sce_krr_train = compute_LinECE(y_train, prob_krr_train)
        ece_krr_train = expected_calibration_error(y_train, prob_krr_train, num_bins_train)
        
        print("\n[Kernel Ridge Regression (squared loss)]")
        print("---- Test Data ----")
        print(f"Classification error: {err_krr_test:.4f}")
        print(f"KCE (Gaussian, sigma={kce_sigma_krr_test:.4f}): {kce_krr_gaussian_test.item():.4f}")
        print(f"KCE (Laplace,  sigma={kce_sigma_krr_test:.4f}): {kce_krr_laplace_test.item():.4f}")
        print(f"Smooth CE (SCE): {float(sce_krr_test):.4f}")
        print(f"Expected Calibration Error (binning): {float(ece_krr_test):.4f}")
        
        print("---- Train Data ----")
        print(f"Classification error: {err_krr_train:.4f}")
        print(f"KCE (Gaussian, sigma={kce_sigma_krr_train:.4f}): {kce_krr_gaussian_train.item():.4f}")
        print(f"KCE (Laplace,  sigma={kce_sigma_krr_train:.4f}): {kce_krr_laplace_train.item():.4f}")
        print(f"Smooth CE (SCE): {float(sce_krr_train):.4f}")
        print(f"Expected Calibration Error (binning): {float(ece_krr_train):.4f}")

        results.append({
            'model': 'KRR_reg',
            'kernel': kernel_name,
            'train_size': n_train,
            'error_test': err_krr_test,
            'kce_gaussian_test': kce_krr_gaussian_test.item(),
            'kce_laplace_test': kce_krr_laplace_test.item(),
            'sce_test': float(sce_krr_test),
            'ece_test': float(ece_krr_test),
            'error_train': err_krr_train,
            'kce_gaussian_train': kce_krr_gaussian_train.item(),
            'kce_laplace_train': kce_krr_laplace_train.item(),
            'sce_train': float(sce_krr_train),
            'ece_train': float(ece_krr_train),
        })
        
        # --- Kernel Logistic Regression (GD method) ---
        print("\n[Kernel Logistic Regression (GD method)]")
        # Increase max_iter significantly to ensure convergence for small reg values
        alpha_klr = kernel_logistic_regression(X_train, y_train, kernel_func, model_sigma, reg,
                                               max_iter=1000, lr=0.01, tol=1e-6, method='gd')
        
        # evaluation on test data
        K_test_train = kernel_func(X_test, X_train, model_sigma)
        f_test = K_test_train @ alpha_klr
        prob_klr_test = torch.sigmoid(f_test)
        
        err_klr_test = classification_error(y_test, prob_klr_test)
        kce_sigma_klr_test = select_sigma_from_vector(prob_klr_test)
        kce_klr_gaussian_test = kernel_calibration_error(y_test, prob_klr_test, sigma=kce_sigma_klr_test, kernel_type='gaussian')
        kce_klr_laplace_test  = kernel_calibration_error(y_test, prob_klr_test, sigma=kce_sigma_klr_test, kernel_type='laplace')
        sce_klr_test = compute_LinECE(y_test, prob_klr_test)
        ece_klr_test = expected_calibration_error(y_test, prob_klr_test, num_bins_test)
        
        # evaluation on training data
        K_train_train = kernel_func(X_train, X_train, model_sigma)
        f_train = K_train_train @ alpha_klr
        prob_klr_train = torch.sigmoid(f_train)
        
        err_klr_train = classification_error(y_train, prob_klr_train)
        kce_sigma_klr_train = select_sigma_from_vector(prob_klr_train)
        kce_klr_gaussian_train = kernel_calibration_error(y_train, prob_klr_train, sigma=kce_sigma_klr_train, kernel_type='gaussian')
        kce_klr_laplace_train  = kernel_calibration_error(y_train, prob_klr_train, sigma=kce_sigma_klr_train, kernel_type='laplace')
        sce_klr_train = compute_LinECE(y_train, prob_klr_train)
        ece_klr_train = expected_calibration_error(y_train, prob_klr_train, num_bins_train)
        
        print("---- Test Data ----")
        print(f"Classification error: {err_klr_test:.4f}")
        print(f"KCE (Gaussian, sigma={kce_sigma_klr_test:.4f}): {kce_klr_gaussian_test.item():.4f}")
        print(f"KCE (Laplace,  sigma={kce_sigma_klr_test:.4f}): {kce_klr_laplace_test.item():.4f}")
        print(f"Smooth CE (SCE): {float(sce_klr_test):.4f}")
        print(f"Expected Calibration Error (binning): {float(ece_klr_test):.4f}")
        
        print("---- Train Data ----")
        print(f"Classification error: {err_klr_train:.4f}")
        print(f"KCE (Gaussian, sigma={kce_sigma_klr_train:.4f}): {kce_klr_gaussian_train.item():.4f}")
        print(f"KCE (Laplace,  sigma={kce_sigma_klr_train:.4f}): {kce_klr_laplace_train.item():.4f}")
        print(f"Smooth CE (SCE): {float(sce_klr_train):.4f}")
        print(f"Expected Calibration Error (binning): {float(ece_klr_train):.4f}")
        
        results.append({
            'model': 'KLR',
            'kernel': kernel_name,
            'train_size': n_train,
            'error_test': err_klr_test,
            'kce_gaussian_test': kce_klr_gaussian_test.item(),
            'kce_laplace_test': kce_klr_laplace_test.item(),
            'sce_test': float(sce_klr_test),
            'ece_test': float(ece_klr_test),
            'error_train': err_klr_train,
            'kce_gaussian_train': kce_klr_gaussian_train.item(),
            'kce_laplace_train': kce_klr_laplace_train.item(),
            'sce_train': float(sce_klr_train),
            'ece_train': float(ece_klr_train),
        })
    
    return results

def run_experiment_log_scale(n_low=100, n_high=10000, n_test=10000, num=10, input_dim=1):
    train_sizes = np.logspace(np.log10(n_low), np.log10(n_high), num=num).astype(int)
    all_results = []
    
    for n_train in train_sizes:
        print(f"\n----- Training size: {n_train} -----")
        results = run_experiment(n_train=n_train, n_test=n_test, reg=1e-2, input_dim=input_dim)
        all_results.extend(results)
    
    return all_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-e', default='n_asymptotic',
                        help='specifies the experimental settings (the change of behaviors w.r.t. the size of training dataset or the value of the regularization parameter).')
    parser.add_argument('--sample_size_high', type=int, default=10000)
    parser.add_argument('--sample_size_low', type=int, default=100)
    parser.add_argument('--sample_size_test', type=int, default=10000)
    parser.add_argument('--num_candidates_sample', type=int, default=10)
    parser.add_argument('--num_candidates_reg', type=int, default=10)
    parser.add_argument('--input_dim', type=int, default=1)
    
    
    parser.set_defaults(parse=True)
    args = parser.parse_args()
    print(args)
    
    # definition of seeds and experiment execution
    seeds = [123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627, 282930]
    
    all_results = []
    
    if args.exp_name in ['n_asymptotic']:
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            # run run_experiment_log_scale() for each seed
            results = run_experiment_log_scale(n_low=args.sample_size_low, n_high=args.sample_size_high, 
                                               n_test=args.sample_size_test, num=args.num_candidates_sample,
                                               input_dim=args.input_dim)
            # add seed information
            for r in results:
                r['seed'] = seed
            all_results.extend(results)

        # aggregate results into a DataFrame
        df_all = pd.DataFrame(all_results)

        # calculate mean and standard deviation for each train_size, model, kernel
        grouped = df_all.groupby(['train_size', 'model', 'kernel']).agg({
            'error_test': ['mean', 'std'],
            'kce_gaussian_test': ['mean', 'std'],
            'kce_laplace_test': ['mean', 'std'],
            'ece_test': ['mean', 'std'],
            'sce_test': ['mean', 'std'],
            'error_train': ['mean', 'std'],
            'kce_gaussian_train': ['mean', 'std'],
            'kce_laplace_train': ['mean', 'std'],
            'ece_train': ['mean', 'std'],
            'sce_train': ['mean', 'std']
        }).reset_index()

        # create a folder for saving results if it does not exist
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        # save df_all as a CSV file
        df_all.to_csv(os.path.join(results_dir, "experiment_results_n.csv"), index=False)
        # save grouped results if needed
        grouped.to_csv(os.path.join(results_dir, "experiment_results_grouped_n.csv"), index=False)
    
    elif args.exp_name in ['reg_sensitivity']:
        #reg_candidates = np.logspace(-4, 2, num=args.num_candidates_reg)  # example: 1e-4 ～ 1e2
        #reg_candidates = np.logspace(-6, 2, num=args.num_candidates_reg)  # example: 1e-4 ～ 1e2
        reg_candidates = np.logspace(-6, 3, num=args.num_candidates_reg)  # example: 1e-4 ～ 1e3
        n_train_fixed = args.sample_size_high
        n_test_fixed = args.sample_size_test
        for reg in reg_candidates:
            print('reg_size:', reg)
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                results = run_experiment(n_train=n_train_fixed, n_test=n_test_fixed, reg=reg,
                                         input_dim=args.input_dim)
                # add seed and reg information
                for r in results:
                    r['reg'] = reg
                    r['seed'] = seed
                all_results.extend(results)
        
        # aggregate results into a DataFrame
        df_all = pd.DataFrame(all_results)
        grouped = df_all.groupby(['reg', 'model', 'kernel']).agg({
            'error_test': ['mean', 'std'],
            'kce_gaussian_test': ['mean', 'std'],
            'kce_laplace_test': ['mean', 'std'],
            'ece_test': ['mean', 'std'],
            'sce_test': ['mean', 'std'],
            'error_train': ['mean', 'std'],
            'kce_gaussian_train': ['mean', 'std'],
            'kce_laplace_train': ['mean', 'std'],
            'ece_train': ['mean', 'std'],
            'sce_train': ['mean', 'std']
        }).reset_index()
        
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        df_all.to_csv(os.path.join(results_dir, "experiment_results_reg.csv"), index=False)
        grouped.to_csv(os.path.join(results_dir, "experiment_results_grouped_reg.csv"), index=False)

if __name__ == '__main__':
    main()