# Simulation data for ML/DL Missing Data Imputation; 3 Missing Mechanism * 3 Missing Porportion
# Alzheimer’s Disease Simulation Framework; CSFratio–MoCA
# 12/14/2025

# Missing Data Generators

### function to randomly set a proportion of target_cov values to NA (MCAR CASE)
# @target_cov: target missing variable
# @ missing_rate: specify the missing proportion of target_cov
# return: vector of target variable with target proportion missing
gen_mcar <- function(target_cov, missing_rate) {
  n <- length(target_cov)
  idx <- sample(1:n, size = floor(missing_rate * length(target_cov)))
  target_cov_mcar <- target_cov
  target_cov_mcar[idx] <- NA
  return(target_cov_mcar)
}


### function that set missingness of target variable depends on other variable (MAR)
# @target_missing: target missing proportion
# @target_cov: target missing variable
# @X_gen_missing: predictors related to missing generation
## return: vector of target variable with target proportion missing
gen_mar <- function(target_missing,
                    target_cov,
                    X_gen_missing,
                    init = NULL) {
  
  for (col in names(X_gen_missing)) {
    if (is.factor(X_gen_missing[[col]])) {
      if (nlevels(X_gen_missing[[col]]) == 2) {
        X_gen_missing[[col]] <- as.numeric(X_gen_missing[[col]]) - 1
      } else {
        stop(paste("Column", col, "has more than 2 levels. Cannot encoding."))
      }
    }
  }
  
  X <- as.matrix(X_gen_missing)
  p <- ncol(X_gen_missing)
  n <- nrow(X_gen_missing)
  
  if (is.null(init)) {
    init <- rep(0, p + 1)
  }
  
  sq_loss <- function(para) {
    gamma0 <- para[1]
    gammas <- para[-1]
    
    link_pred <- gamma0 + X %*% gammas
    p_obs <- 1 / (1 + exp(-link_pred))
    expected_missing <- mean(1 - p_obs)
    
    return((expected_missing - target_missing)^2)
  }
  
  opt_result <- optim(par = init, fn = sq_loss, method = "BFGS")
  gamma_est <- opt_result$par
  
  link_pred_gamma <- gamma_est[1] + X %*% gamma_est[-1]
  
  p_obs_gamma <- 1 / (1 + exp(-link_pred_gamma))
  
  X_obs_ind <- rbinom(n, 1, p_obs_gamma)
  
  target_cov_missing <- ifelse(X_obs_ind == 1, target_cov, NA)
  
  return(target_cov_missing)
}


### function that set missingness of target variable depend on itself (MNAR)
# @target_missing: target missing proportion
# @target_cov: target missing variable
# return: vector of target variable with target proportion missing
### function that set missingness of target variable depend on itself (MNAR)
# @target_missing: target missing proportion
# @target_cov: target missing variable (complete vector)
# return: vector of target variable with target proportion missing
gen_mnar <- function(target_missing, target_cov, init = NULL) {
  cov_df <- data.frame(id = seq_along(target_cov), cov = target_cov, miss = 0)
  
  n <- nrow(cov_df)
  cov <- cov_df$cov
  
  if (is.null(init)) {
    init <- rep(0, 2)
  }
  
  sq_loss <- function(para) {
    gamma0 <- para[1]
    gamma1 <- para[2]
    logit_pmiss <- gamma0 + gamma1 * cov
    p_miss <- 1 / (1 + exp(-logit_pmiss))
    expected_missing <- mean(p_miss)
    (expected_missing - target_missing)^2
  }
  
  opt_result <- optim(par = init, fn = sq_loss, method = "BFGS")
  gamma_est <- opt_result$par
  
  logit_pmiss <- gamma_est[1] + gamma_est[2] * cov
  p_miss <- 1 / (1 + exp(-logit_pmiss))
  
  cov_df$miss <- rbinom(n, 1, p_miss)
  
  target_mnar <- ifelse(cov_df$miss == 1, NA, cov_df$cov)
  return(target_mnar)
}




# Data Generators

### function to build reproducible seed
make_seeds <- function(nsims, nscenarios, root_seed = 123) {
  expand.grid(
    sim = 1:nsims,
    scenario = 1:nscenarios
  ) |>
    transform(seed = root_seed + sim * 100 + scenario)
}


### function to simulate an AD sample dataset
# @n: sample size
generate_population <- function(n = 300) {
  beta0 <- 30
  beta1 <- 18
  beta2 <- -0.3
  beta3 <- -1.5
  sigma_sd <- 1.5
  target_r <- 0.5
  
  Sigma <- matrix(c(1, target_r, target_r, 1), 2, 2)
  L <- chol(Sigma)
  Z <- matrix(rnorm(n * 2), ncol = 2) %*% L
  z_age <- Z[,1]; z_csf <- Z[,2]
  
  age <- 72 + 7 * scale(z_age)
  u <- pnorm(z_csf)
  csf <- rgamma(u, shape = 9, scale = 0.008)
  extreme_csf <- runif(30, min = quantile(csf, 0.75), max = quantile(csf, 0.95))
  csf <- c(csf[1:270], extreme_csf)
  
  sex <- factor(rbinom(n, 1, 0.6))
  
  moca <- beta0 + beta1 * csf + beta2 * age + beta3 * (as.numeric(sex)-1) +
    rnorm(n, 0, sigma_sd)
  
  data.frame(moca, csf, age, sex)
}


### function to generate missingness, save full & missing in a .rds file, save seed in a seperate .rds file. 
#@ nsims: number of simulations.
#@ missing_types: define the type of missing (only accept MCAR/MAR/MNAR)
#@ missing_levels: define the missing proportion
generate_pop_and_missing_save <- function(nsims, missing_types, missing_levels,
                                          root_seed = 123, file = "missdata.rds") {
  
  nscenarios <- length(missing_types) * length(missing_levels)
  seed_table <- make_seeds(nsims, nscenarios, root_seed)
  
  all_miss_data <- list()
  
  scen_idx <- 1
  for (type in missing_types) {
    for (rate in missing_levels) {
      
      for (i in 1:nsims) {
        
        this_seed <- seed_table$seed[seed_table$sim == i & seed_table$scenario == scen_idx]
        set.seed(this_seed)
        
        full <- generate_population()
        
        miss_cov <- switch(type,
        MCAR = gen_mcar(full[["csf"]], rate),
        MAR  = gen_mar(rate, full[["csf"]], full[c("age", "sex")]),
        MNAR = gen_mnar(rate, full[["csf"]])
        )
        miss <- full
        miss[["csf"]] <- miss_cov
        
        all_miss_data[[paste0("sim", i, "_", type, "_", rate)]] <- list(
          full = full,
          miss = miss
        )
      }
      scen_idx <- scen_idx + 1
    }
  }
  
  saveRDS(all_miss_data, file)
  saveRDS(seed_table, sub("\\.rds$", "_seeds.rds", file))
  return(list(data = all_miss_data, seeds = seed_table))
}


# Parameters
n_sim <- 500
missing_levels <- c(0.25, 0.5, 0.75)
missing_types  <- c("MCAR", "MAR", "MNAR")
methods_vec <- c("CCA", "MICE", "KNN", "SVM", "RF", "VAE", "MLP")
# methods_vec <- c("CCA", "GAN")
beta_name <- c("Intercept", "csf", "age", "sex1")
outcome <- "moca"
covariates <- c("csf", "age", "sex")
len_method <- length(methods_vec)
dict <- "~/Desktop/cleaned_code/testdata/missdata.rds"

# Generate and save simulated population and missing data
sim_pop <- generate_pop_and_missing_save(
  nsims          = n_sim,
  missing_types  = missing_types,
  missing_levels = missing_levels,
  root_seed      = 123,
  file           = dict
)
