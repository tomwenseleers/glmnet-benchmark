import numpy as np
from scipy.sparse import csc_matrix
from scipy.io import mmread
import os
import time
from glum import GeneralizedLinearRegressorCV
from glum import GeneralizedLinearRegressor

def get_data(n, p, data_prefix):
    # Create file paths
    X_file = os.path.join(data_prefix, f"{n}_{p}_X.mtx")
    y_file = os.path.join(data_prefix, f"{n}_{p}_y.csv")
    beta_true_file = os.path.join(data_prefix, f"{n}_{p}_beta_true.csv")
    
    # Read the data
    X = csc_matrix(mmread(X_file))  # Read .mtx file and convert to CSC format
    y = np.genfromtxt(y_file, delimiter=',', skip_header=1)
    beta_true = np.genfromtxt(beta_true_file, delimiter=',', skip_header=1)
    
    return X, y, beta_true

def timer_glr(X, y, glr):
    start = time.time()
    glr_fit = glr.fit(X, y)
    end = time.time()
    return glr_fit, end-start

# load simulated data (spike train convoluted with gaussian point spread function
# with 1000 / 10000 coefficients being nonzero (all positive) and Gaussian noise
# added
n = 10000
p = 10000
X_sparse, y, beta_true = get_data(n, p, 'data')

# determine optimal alpha using cross validation
glrcv = GeneralizedLinearRegressorCV(l1_ratio=1, # lasso
          family='normal',
          fit_intercept=False,
          # gradient_tol=1e-7,
          scale_predictors=False,
          lower_bounds = np.zeros(p), # nonnegativity constraints
          min_alpha_ratio=0.01 if n < p else 1e-4,
          solver='irls-cd',
          cv=10, # use 10-fold CV as in glmnet
          max_iter=10000,
          n_alphas=100 # as in glmnet
          )
glrcv_fit, elapsed = timer_glr(X_sparse, y, glrcv)
print("Coef:\n", glrcv_fit.coef_)
print("Intercept: ", glrcv_fit.intercept_)
print("N_iter: ", glrcv_fit.n_iter_)
print("Elapsed: ", elapsed)
# now refit on complete dataset using optimal alpha
glr = GeneralizedLinearRegressor(l1_ratio=1, # lasso
          family='normal',
          fit_intercept=False,
          # gradient_tol=1e-7,
          scale_predictors=False,
          lower_bounds = np.zeros(p), # nonnegativity constraints
          min_alpha_ratio=0.01 if n < p else 1e-4,
          solver='irls-cd',
          max_iter=10000,
          alpha=glrcv_fit.alpha_
          #n_alphas=100,
          #alpha_search=True
          )
glr_fit, elapsed = timer_glr(X_sparse, y, glr)
print("Coef:\n", glr_fit.coef_)
print("Intercept: ", glr_fit.intercept_)
print("N_iter: ", glr_fit.n_iter_)
print("Elapsed: ", elapsed)
