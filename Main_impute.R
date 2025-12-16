# Main Function to perform imputation on Alzheimer’s Disease simulated data
# Use data from Simulation.R as an example; run CCA/MICE/KNN/SVM/RF/VAE/MLP
# To run GAN, start a new session to reset the version of 'tensorflow' called
# 12/14/2025

library(dplyr)
library(tidyr)

source("~/Documents/ML_Classic_Imputation.R") 
source("~/Documents/MLP_imputation.R") 
# source("~/Documents/GAN_imputation.R") 
source("~/Documents/VAE_imputation.R") 

# Parameters (Should be the same as the Simulation file used)
n_sim <- 50
missing_levels <- c(0.25, 0.5, 0.75)
missing_types  <- c("MCAR", "MAR", "MNAR")
methods_vec <- c("CCA", "MICE", "KNN", "SVM", "RF", "VAE", "MLP")
# methods_vec <- c("CCA", "GAN")
beta_name <- c("Intercept", "csf", "age", "sexMale")
outcome <- "moca"
covariates <- c("csf", "age", "sex")
len_method <- length(methods_vec)
dict <- "~/Desktop/cleaned_code/testdata/missdata.rds"

true_intercept <- 30
true_csf       <- 18
true_age       <- -0.3
true_sex       <- -1.5


# Main regression formula
if (length(covariates) == 0) {
  formula_main <- as.formula(paste(outcome, "~ 1"))
} else {
  formula_main <- as.formula(paste(outcome, "~", paste(covariates, collapse = " + ")))
}

# Read in local data
missdata_file <- "~/Desktop/cleaned_code/testdata/missdata.rds"
seed_file     <- "~/Desktop/cleaned_code/testdata/missdata_seeds.rds"

all_miss_data <- readRDS(missdata_file)
seed_table    <- readRDS(seed_file)

# Scenario Setup
scenario_keys <- vector("character", length = length(missing_types) * length(missing_levels))
scen_idx <- 1
for (type in missing_types) {
  for (rate in missing_levels) {
    scenario_keys[scen_idx] <- paste(type, rate, sep = "_")
    scen_idx <- scen_idx + 1
  }
}
n_scenarios <- length(scenario_keys)

# Simulation containers
coef_results <- list()
se_results   <- list()
mean_runtimes <- matrix(NA, nrow = n_scenarios, ncol = len_method,
                        dimnames = list(NULL, methods_vec))

# Hyperparameters tuning
method_args <- list(
  CCA  = list(),
  MICE = list(m = 30, maxit = 40),
  KNN  = list(k = 5),
  SVM  = list(),
  RF   = list(m = 15, maxit = 10, ntree = 50),
  VAE  = list(epochs = 100, latent_dim = 8, hidden_dim = 32, lr = 0.001),
  MLP  = list(epochs = 100, batch = 32, hidden_dim = 32)
  # GAN  = list(batch_size = 16, hint_rate = 1.0, alpha = 100, iterations = 3000)
)


beta_term <- function(x) if (x == "Intercept") "(Intercept)" else x

get_imputed_df <- function(x) {
  if (is.data.frame(x)) x else x$imputed_data
}

call_method <- function(m, miss, this_seed, outcome, covariates, method_args) {
  if (m %in% c("VAE", "MLP", "GAN")) {
    do.call(m, c(list(missData = miss, outcome = outcome, covariates = covariates,
                      seed = this_seed, lm.fit = FALSE), method_args[[m]]))
  } else if (m %in% c("CCA", "KNN", "SVM", "MICE", "RF")) {
    do.call(m, c(list(missData = miss, outcome = outcome, covariates = covariates),
                 method_args[[m]]))
  } else {
    stop("Unknown method: ", m)
  }
}



scenario_index <- 1

for (type in missing_types) {
  for (rate in missing_levels) {
    
    time_results <- lapply(methods_vec, function(x) numeric(n_sim))
    names(time_results) <- methods_vec
    
    beta_list <- lapply(seq_along(beta_name), function(k)
      matrix(NA, nrow = n_sim, ncol = len_method,
             dimnames = list(NULL, methods_vec)))
    se_list <- lapply(seq_along(beta_name), function(k)
      matrix(NA, nrow = n_sim, ncol = len_method,
             dimnames = list(NULL, methods_vec)))
    
    scenario_key <- paste(type, rate, sep = "_")
    scen_number  <- which(scenario_keys == scenario_key)
    if (length(scen_number) != 1) stop("Scenario not found in scenario_keys")
    
    cat("\n>>> Running scenario:", scenario_key, "\n")
    
    for (i in seq_len(n_sim)) {
      data_name <- paste0("sim", i, "_", type, "_", rate)
      if (!data_name %in% names(all_miss_data)) {
        stop("Data for ", data_name, " not found in RDS. Check naming.")
      }
      
      this_seed <- seed_table$seed[seed_table$sim == i & seed_table$scenario == scen_number]
      if (length(this_seed) != 1) stop("Seed lookup returned not exactly 1 value.")
      set.seed(this_seed)
      
      miss_entry <- all_miss_data[[data_name]]
      miss <- miss_entry$miss
      
      for (m in methods_vec) {
        
        t0 <- system.time({
          out_obj <- call_method(m, miss, this_seed, outcome, covariates, method_args)
        })["elapsed"]
        time_results[[m]][i] <- t0
        
        if (m %in% c("MICE", "RF")) {
          sm <- summary(out_obj)
          term_vec <- if ("term" %in% names(sm)) sm$term else rownames(sm)
          
          for (k in seq_along(beta_name)) {
            b <- beta_term(beta_name[k])
            idx <- match(b, term_vec)
            beta_list[[k]][i, m] <- if (!is.na(idx)) sm$estimate[idx] else NA
            se_list[[k]][i, m]   <- if (!is.na(idx)) sm$std.error[idx] else NA
          }
          
        } else {
          imputed_df <- get_imputed_df(out_obj)
          if (is.null(imputed_df)) {
            stop("Method ", m, " did not return an imputed data.frame (or list(imputed_data=...)).")
          }
          
          fit_lm <- lm(formula_main, data = imputed_df)
          est <- coef(fit_lm)
          se  <- coef(summary(fit_lm))[, "Std. Error"]
          
          for (k in seq_along(beta_name)) {
            b <- beta_term(beta_name[k])
            beta_list[[k]][i, m] <- if (b %in% names(est)) est[[b]] else NA
            se_list[[k]][i, m]   <- if (b %in% names(se))  se[[b]]  else NA
          }
        }
      }
    }
    
    for (j in seq_along(beta_name)) {
      key <- paste(beta_name[j], type, rate, sep = "_")
      coef_results[[key]] <- beta_list[[j]]
      se_results[[key]]   <- se_list[[j]]
    }
    
    mean_runtimes[scenario_index, ] <- sapply(time_results, mean, na.rm = TRUE)
    scenario_index <- scenario_index + 1
  }
}


# Results evaluation

# Convert results to long format
list_to_long_df <- function(results_list, miss_rate = NULL) {
  bind_rows(
    lapply(names(results_list), function(x) {
      type_rate <- strsplit(x, "_")[[1]]
      type <- paste(type_rate[2:(length(type_rate) - 1)], collapse = "_")
      rate <- as.numeric(type_rate[length(type_rate)])
      mat <- results_list[[x]]
      data.frame(
        type = type,
        rate = factor(rate),
        #rate = factor(miss_rate),
        method = rep(colnames(mat), each = nrow(mat)),
        sim_id = rep(seq_len(nrow(mat)), times = ncol(mat)),
        estimate = as.vector(mat),
        stringsAsFactors = FALSE
      )
    }),
    .id = "scenario"
  )
}

# miss_rate_val <- mean(sim_pop[["miss_rate"]])
intercept_df <- list_to_long_df(coef_results[grep("^Intercept", names(coef_results))])
csf_df       <- list_to_long_df(coef_results[grep("^csf", names(coef_results))])
age_df       <- list_to_long_df(coef_results[grep("^age", names(coef_results))])
sex_df       <- list_to_long_df(coef_results[grep("^sex", names(coef_results))])

intercept_SEdf <- list_to_long_df(se_results[grep("^Intercept", names(se_results))])
csf_SEdf       <- list_to_long_df(se_results[grep("^csf", names(se_results))])
age_SEdf       <- list_to_long_df(se_results[grep("^age", names(se_results))])
sex_SEdf       <- list_to_long_df(se_results[grep("^sex", names(se_results))])


# Define the evaluation table
eval_table <- function(est_df, se_df, true_val) {
  df <- est_df %>%
    rename(estimate = estimate) %>%
    left_join(
      se_df %>%
        rename(SE = estimate) %>%
        dplyr::select(scenario, type, rate, method, sim_id, SE),
      by = c("scenario", "type", "rate", "method", "sim_id")
    )
  
  df %>%
    group_by(type, rate, method) %>%
    summarise(
      perc_bias = mean((estimate - true_val) / true_val) * 100,
      SE = mean(SE),
      ESD = sd(estimate),
      coverage_prob = mean((estimate - 1.96 * SE <= true_val) &
                             (estimate + 1.96 * SE >= true_val)),
      .groups = "drop"
    ) %>%
    arrange(type, rate, method)
}


tab_intercept <- eval_table(intercept_df, intercept_SEdf, true_intercept)
tab_csf       <- eval_table(csf_df,       csf_SEdf,       true_csf)
tab_age       <- eval_table(age_df,       age_SEdf,       true_age)
tab_sex       <- eval_table(sex_df,       sex_SEdf,       true_sex)


summary_all <- bind_rows(
  tab_intercept %>% mutate(param = "Intercept"),
  tab_csf       %>% mutate(param = "csf"),
  tab_age       %>% mutate(param = "age"),
  tab_sex       %>% mutate(param = "sex")
) %>%
  relocate(param, .before = type)

print(summary_all, n=nrow(summary_all))
