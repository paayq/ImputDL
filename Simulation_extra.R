# Simulation data for ML/DL Missing Data Imputation; Extra MNAR Case
# Alzheimer’s Disease Simulation Framework; CSFratio–MoCA
# 12/14/2025

# Missing Data Generator

### Function to generate an MNAR where missingness of the target covariate depends on itself (tail heaviness)
# and the outcome. The realized missing proportion is random.
# @target_cov: target missing variable
# @outcome: outcome variable
# return: a list contain vector of target variable and a proportion missing value
gen_MNAR_extra <- function(target_cov, outcome) {
  
  n <- length(target_cov)
  
  # Tail score
  s <- ((target_cov - mean(target_cov)) / sd(target_cov))^4
  
  # Logistic mapping
  alpha <- -1
  beta  <- 5
  p_base <- plogis(alpha + beta * s)

  p_with_outcome <- p_base + 0.3 * (outcome > 26) + 0.3 * (outcome < 11)
  p_jittered <- p_with_outcome + runif(n, -0.08, 0.08)

  p_miss <- pmin(0.95, pmax(0.05, p_jittered))
  
  miss_indicator <- runif(n) < p_miss
  
  # Apply missingness
  target_cov_mnar <- target_cov
  target_cov_mnar[miss_indicator] <- NA
  
  return(list(
    cov = target_cov_mnar,
    miss_rate = round(mean(miss_indicator), 3)
  ))
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
  sigma_sd <- 4
  target_r <- 0.5
  
  n_outliers <- 60
  outlier_multiplier <- 2.5
  
  Sigma <- matrix(c(1, target_r, target_r, 1), 2, 2)
  L <- chol(Sigma)
  Z <- matrix(rnorm(n * 2), ncol = 2) %*% L
  z_age <- Z[, 1]; z_csf <- Z[, 2]
  
  age <- 72 + 7 * scale(z_age)
  u <- pnorm(as.numeric(scale(z_csf)))
  csf <- qgamma(u, shape = 7, scale = 0.05)
  
  idx_out <- sample(1:n, n_outliers)
  csf[idx_out] <- csf[idx_out] * outlier_multiplier
  
  alpha <- 0.4
  gamma <- 1.2
  p_female <- plogis(alpha + gamma * scale(z_csf))
  sex <- rbinom(n, 1, p_female)
  sex <- factor(sex, levels = c(0, 1), labels = c("Male", "Female"))
  
  moca <- beta0 + beta1 * csf + beta2 * age + beta3 * (as.numeric(sex) - 1) +
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
      miss_rate <- NA
      
      for (i in 1:nsims) {
        this_seed <- seed_table$seed[seed_table$sim == i & seed_table$scenario == scen_idx]
        set.seed(this_seed)
        
        full <- generate_population()
        
        MNAR <- gen_MNAR_extra(full[["csf"]], full[["moca"]])
        miss_cov <- MNAR$cov
        miss_rate[i] <- MNAR$miss_rate
        
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
  list(data = all_miss_data, seeds = seed_table, miss_rate = miss_rate)
}


# Parameters
n_sim <- 500
missing_levels <- c(0.58)
missing_types <- c("MNAR")
methods_vec <- c("CCA", "MICE", "KNN", "SVM", "RF", "VAE", "MLP")
# methods_vec <- c("CCA", "GAN")
beta_name <- c("Intercept", "csf", "age", "sex1")
outcome <- "moca"
covariates <- c("csf", "age", "sex")
len_method <- length(methods_vec)
dict <- "~/Desktop/cleaned_code/testdata/missdata_extra.rds"


# Generate and save simulated population and missing data
sim_pop <- generate_pop_and_missing_save(
  nsims          = n_sim,
  missing_types  = missing_types,
  missing_levels = missing_levels,
  root_seed      = 123,
  file           = dict
)
