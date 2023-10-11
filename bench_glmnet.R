# SPARSE & DENSE EXAMPLE WITH n=p=10000 & nonnegativity constraints ####

library(glmnet)
library(microbenchmark)
library(Matrix)
library(L0glm) # my own experimental GLM package to fit L0 penalized GLMs
library(export)

setwd("~/Github/glmnet-benchmark")

data_prefix = 'data'
n = 10000L
p = 10000L

# simulate data (blurred spike train with Gaussian noise)
# data = simul(method = 1, n = n, p = p, seed = 2) # X simulated as a banded sparse matrix with time-shifted Gaussian peak shape function
# writeMM(data$X, file=paste0(data_prefix, '/', n, '_', p, '_X.mtx')) # write sparse dgCMatrix as Matrix Market format
# write.csv(data$y, file=paste0(data_prefix, '/', n, '_', p, '_y.csv'), row.names=F)
# write.csv(as.vector(data$beta_true), file=paste0(data_prefix, '/', n, '_', p, '_beta_true.csv'), row.names=F)
# sum(beta_true!=0) # 1000 nonzero coefficients out of 10000

# NOTE:
# abess:generate_data() might be a good generic simulation function for
# benchmarking
# https://github.com/abess-team/abess/blob/master/R-package/R/generate.data.R
# make_glm_data in 
# https://github.com/abess-team/abess/blob/master/python/abess/datasets.py


get_data = function(n, p) {
  list(X=as(as(readMM(paste0(data_prefix, '/', n, '_', p, '_X.mtx')), "CsparseMatrix"), "dgCMatrix"), 
       y=read.csv(paste0(data_prefix, '/', n, '_', p, '_y.csv'), header=T),
       beta_true=read.csv(paste0(data_prefix, '/', n, '_', p, '_beta_true.csv'), header=T))
}

timer_glmnet = function(X, y) {
  time.out = microbenchmark({  cvfit <- cv.glmnet(X, y, family='gaussian', 
                                                  tol=1e-14, standardize=F, 
                                                  alpha=1, # LASSO
                                                  lower.limits=0, # nonnegativity constraints
                                                  intercept=FALSE, # no intercept
                                                  nlambda = 100) 
  fit <- glmnet(X, y, family='gaussian', 
                tol=1e-14, standardize=F, 
                alpha=1, # LASSO
                lower.limits=0, # nonnegativity constraints
                intercept=FALSE, # no intercept
                nlambda = 100) }, 
  times=1L, 
  unit='s')
  coefs = coef(fit, s=cvfit$lambda.min)
  
  list(fit=fit, coefs=coefs, elapsed=summary(time.out)$mean)
}


dat = get_data(n, p)
X_dense = as.matrix(dat$X)
X_sparse = dat$X
y = dat$y[,1]
coefs_true = dat$beta_true[,1]

# TIMINGS WHEN FIT AS SPARSE SYSTEM WITH GLMNET LASSO OR USING L0 PENALIZED GLM ####

## timings for glmnet when fit as sparse system ####
out_sparse = timer_glmnet(X_sparse, y)
R = cor(as.vector(out_sparse$coefs)[-1], coefs_true)
print("Correlation between estimated & true coefs:\n")
print(round(R, 4)) 
print(paste("N_iter:", out_sparse$fit$npasses)) 
print(paste("Elapsed:", out_sparse$elapsed))
# [1] "Correlation between estimated & true coefs:\n"
# > print(round(R, 4)) 
# [1] 0.9092
# > print(paste("N_iter:", out_sparse$fit$npasses)) 
# [1] "N_iter: 762"
# > print(paste("Elapsed:", out_dense$elapsed))
# [1] "Elapsed: 0.9451085"

plot(x=as.vector(out_sparse$coefs)[-1], y=coefs_true, pch=16, col='steelblue',
     xlab="estimated coefficients", ylab="true coefficients",
     main=paste0("glmnet nonnegative LASSO regression\n(n=", n,", p=", p,", R=", round(R,4), ")"))
dir.create("./plots/")
graph2png(file="./plots/glmnet_LASSO_true_vs_estimated_coefs.png", width=7, height=5)

table(as.vector(out_sparse$coefs)[-1]!=0, 
      coefs_true!=0, dnn=c("estimated beta nonzero", "true beta nonzero"))
#                 true beta nonzero
# estimated beta nonzero FALSE TRUE
#                  FALSE  7929  195
#                  TRUE   1071  805


## timings for L0glm when fit as sparse system using eigen sparse Cholesky LLT solver ####
# (L0 penalized GLM with L0 penalty approximated via
# iterative adaptive ridge regression procedure & using 
# eigen dense or sparse Cholesky LLT solver)

system.time(L0glm_sparse_chol <- L0glm.fit(X_sparse, y, 
                                      family = gaussian(identity), 
                                      lambda = "gic", # options "aic", "bic", "gic", "ebic", "mbic"
                                      nonnegative = TRUE,
                                      solver = "eigen",
                                      method = 7L) # Cholesky LLT
) # 0.18s, i.e. 4x faster than sparse glmnet LASSO fit

R = cor(as.vector(L0glm_sparse_chol$coefficients), coefs_true)
print("Correlation between estimated & true coefs:\n")
print(round(R, 4)) # 0.9929 - much better solution than LASSO

plot(x=as.vector(L0glm_sparse_chol$coefficients), y=coefs_true, pch=16, col='steelblue',
     xlab="estimated coefficients", ylab="true coefficients",
     main=paste0("L0glm regression\n(n=", n,", p=", p, ", R=", round(R,4), ")"))
graph2png(file="./plots/L0glm_true_vs_estimated_coefs.png", width=7, height=5)

table(as.vector(L0glm_sparse_chol$coefficients)!=0, 
      coefs_true!=0, dnn=c("estimated beta nonzero", "true beta nonzero"));
# better true positive rate & much lower false positive rate than LASSO
#                 true beta nonzero
# estimated beta nonzero FALSE TRUE
#                  FALSE  8919   45
#                  TRUE     81  955


## timings for L0glm when fit as sparse system using sparse osqp quadratic programming solver ####

system.time(L0glm_sparse_osqp <- L0glm.fit(X_sparse, y, 
                                           family = gaussian(identity), 
                                           lambda = "ebic", # options "aic", "bic", "gic", "ebic", "mbic"
                                           nonnegative = TRUE,
                                           solver = "osqp") # osqp solver
) # 11.2s, i.e. 13x slower than sparse glmnet LASSO fit

R = cor(as.vector(L0glm_sparse_osqp$coefficients), coefs_true)
print("Correlation between estimated & true coefs:\n")
print(round(R, 4)) # 0.9899 - much better solution than LASSO


## timings for L0glm when fit as sparse system using bigGlm glmnet solver ####

system.time(L0glm_sparse_bigglm <- L0glm.fit(X_sparse, y, 
                                             family = gaussian(identity), 
                                             lambda = "ebic", # options "aic", "bic", "gic", "ebic", "mbic"
                                             nonnegative = TRUE,
                                             solver = "glmnet") # bigGlm glmnet solver
) # 5.11s, i.e. 6x slower than sparse glmnet LASSO fit

R = cor(as.vector(L0glm_sparse_bigglm$coefficients), coefs_true)
print("Correlation between estimated & true coefs:\n")
print(round(R, 4)) # 0.9399 - much better solution than LASSO



# TIMINGS WHEN FIT AS DENSE SYSTEM WITH GLMNET LASSO OR USING L0 PENALIZED GLM ####

# solutions identical as above just slower

## timings for glmnet when fit as dense system ####
out_dense = timer_glmnet(X_dense, y)
R = cor(as.vector(out_dense$coefs)[-1], coefs_true)
print("Correlation between estimated & true coefs:\n")
print(round(R, 4)) 
print(paste("N_iter:", out_dense$fit$npasses)) 
print(paste("Elapsed:", out_dense$elapsed))
# [1] "Correlation between estimated & true coefs:\n"
# > print(round(R, 4)) 
# [1] 0.9063
# > print(paste("N_iter:", out_dense$fit$npasses)) 
# [1] "N_iter: 762"
# > print(paste("Elapsed:", out_dense$elapsed))
# [1] "Elapsed: 86.74697"


# timings for L0glm when fit as dense system using eigen dense Cholesky LLT solver ####
# (L0 penalized GLM with L0 penalty approximated via
# iterative adaptive ridge regression procedure & using 
# eigen dense or sparse Cholesky LLT solver)

system.time(L0glm_dense_chol <- L0glm.fit(X_dense, y, 
                        family = gaussian(identity), 
                        lambda = "ebic", # options "aic", "bic", "gic", "ebic", "mbic"
                        nonnegative = TRUE,
                        solver = "eigen",
                        method = 7L) # Cholesky LLT
) # 7s, i.e. 12x faster than dense glmnet LASSO fit

R = cor(as.vector(L0glm_dense_chol$coefficients), coefs_true)
print("Correlation between estimated & true coefs:\n")
print(round(R, 4)) # 0.9928 - much better solution than LASSO

table(as.vector(L0glm_dense_chol$coefficients)!=0, 
      coefs_true!=0, dnn=c("estimated beta nonzero", "true beta nonzero"))
#                 true beta nonzero
# estimated beta nonzero FALSE TRUE
#                  FALSE  8919   45
#                  TRUE     81  955


# timings for L0glm when fit as dense system using dense osqp quadratic programming solver ####

system.time(L0glm_dense_osqp <- L0glm.fit(X_dense, y, 
                                          family = gaussian(identity), 
                                          lambda = "ebic", # options "aic", "bic", "gic", "ebic", "mbic"
                                          nonnegative = TRUE,
                                          solver = "osqp") # osqp solver
) # 18.98s, i.e. 4.6x faster than dense glmnet LASSO fit

R = cor(as.vector(L0glm_dense_osqp$coefficients), coefs_true)
print("Correlation between estimated & true coefs:\n")
print(round(R, 4)) # 0.9899 - much better solution than LASSO


# timings for L0glm when fit as dense system using bigGlm glmnet solver ####

system.time(L0glm_dense_bigglm <- L0glm.fit(X_dense, y, 
                                          family = gaussian(identity), 
                                          lambda = "ebic", # options "aic", "bic", "gic", "ebic", "mbic"
                                          nonnegative = TRUE,
                                          solver = "glmnet") # bigGlm glmnet solver
) # 12.56s, i.e. 6.9x faster than dense glmnet LASSO fit

R = cor(as.vector(L0glm_dense_bigglm$coefficients), coefs_true)
print("Correlation between estimated & true coefs:\n")
print(round(R, 4)) # 0.9399 - much better solution than LASSO


sessionInfo()
# R version 4.3.1 (2023-06-16 ucrt)
# Platform: x86_64-w64-mingw32/x64 (64-bit)
# Running under: Windows 11 x64 (build 22621)
# 
# Matrix products: default
# 
# 
# locale:
# [1] LC_COLLATE=English_United States.utf8  LC_CTYPE=English_United States.utf8   
# [3] LC_MONETARY=English_United States.utf8 LC_NUMERIC=C                          
# [5] LC_TIME=English_United States.utf8    
# 
# time zone: Europe/Paris
# tzcode source: internal
# 
# attached base packages:
#   [1] stats     graphics  grDevices utils     datasets  methods   base     
# 
# other attached packages:
# [1] export_0.3.0          abess_0.4.8           dplyr_1.1.3           L0glm_0.2.0          
# [5] microbenchmark_1.4.10 glmnet_4.1-8          Matrix_1.6-1.1  
