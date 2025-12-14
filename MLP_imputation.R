# MLP-based imputation algorithm
# 12/04/2025

# Multilayer Perceptron (MLP) Imputation
# @missData: the dataframe with missing data (can take-in continuous & factor variables only)
# @outcome: name of the outcome variable, used for lm.fit (character)
# @covariates: vector of covariate names, used for lm.fit (character)
# @epochs: number of training epochs 
# @batch: batch size
# @hidden_dim: number of neurons in hidden layer
# @lr: learning rate
# @lm.fit: if TRUE, fit lm and return estimation matrix, otherwise only imputed data
# return: list(imputed_data, estimation_matrix)
MLP <- function(missData,
                outcome,
                covariates,
                epochs = 100,
                batch = 64,
                hidden_dim = 64,
                lr = 0.001,
                lm.fit = TRUE,
                seed = 123) {
  # Loads tensors in R
  require(keras)
  require(tensorflow)
  set.seed(seed)
  tensorflow::tf$random$set_seed(seed)
  
  if (!tensorflow::tf$executing_eagerly()) {
    tensorflow::tf$compat$v1$enable_eager_execution()
  }
  
  # Code factor variable to numeric matrix
  fact_cols <- names(missData)[sapply(missData, is.factor)]
  numeric_cols <- setdiff(names(missData), fact_cols)
  
  for (col in names(missData)) {
    if (is.factor(missData[[col]])) {
      if (nlevels(missData[[col]]) == 2) {
        missData[[col]] <- as.numeric(missData[[col]]) - 1
      } else {
        stop(paste("Column", col, "has more than 2 levels. Cannot encoding."))
      }
    }
  }
  
  # Missingness mask (TRUE = missing)
  miss_mask <- is.na(missData)
  
  # Standardization
  x <- as.data.frame(missData)
  col_means <- sapply(x[, numeric_cols, drop = FALSE], mean, na.rm = TRUE)
  col_sds   <- sapply(x[, numeric_cols, drop = FALSE], sd,   na.rm = TRUE)
  
  x_scaled <- x
  for (col in numeric_cols) {
    x_scaled[[col]] <- (x_scaled[[col]] - col_means[col]) / col_sds[col]
  }
  
  # Initialize with mean impute
  x_filled <- x_scaled
  col_means_all <- sapply(x_scaled, mean, na.rm = TRUE) # includes 0/1 cols too
  na_idx <- which(is.na(x_filled), arr.ind = TRUE)
  x_filled[na_idx] <- col_means_all[na_idx[, 2]]
  
  x_mat <- as.matrix(x_filled)
  
  
  # MLP autoencoder
  p <- ncol(x_mat)
  
  model <- keras_model_sequential() %>%
    layer_dense(units = hidden_dim, activation = "relu", input_shape = p) %>%
    layer_dense(units = hidden_dim, activation = "relu") %>%
    layer_dense(units = p, activation = "linear")
  
  
  # Loss
  sq_error <- function(y_true, y_pred) {
    tensorflow::tf$square(y_true - y_pred)
  }
  
  model %>% compile(
    loss = sq_error,
    optimizer = optimizer_adam(learning_rate = lr)
  )
  
  cb <- callback_early_stopping(monitor = "loss", patience = 5, restore_best_weights = TRUE)
  
  # Artificial masking for supervised training 
  mask_rate <- 0.2
  obs_idx <- which(!miss_mask, arr.ind = TRUE)
  
 
  for (e in seq_len(epochs)) {
    n_mask <- max(1, floor(mask_rate * nrow(obs_idx)))
    sel <- sample(seq_len(nrow(obs_idx)), size = n_mask, replace = FALSE)
    masked_idx <- obs_idx[sel, , drop = FALSE]
    
    x_in <- x_mat
    x_in[masked_idx] <- col_means_all[masked_idx[, 2]]
    
    train_weight <- matrix(0, nrow = nrow(x_mat), ncol = ncol(x_mat))
    train_weight[masked_idx] <- 1
    train_weight <- array(as.numeric(train_weight), dim = dim(train_weight))
    
    model %>% fit(
      x = x_in,
      y = x_mat,
      sample_weight = train_weight,
      epochs = 1,
      batch_size = batch,
      verbose = 0
    )
  }
  
  reconstructed <- model %>% predict(x_mat)
  final_imputed_data <- as.matrix(x_scaled)
  final_imputed_data[miss_mask] <- reconstructed[miss_mask]
  
  # Back-transform numeric columns
  final_imputed_data[, numeric_cols] <- sweep(final_imputed_data[, numeric_cols, drop = FALSE],
                                              2, col_sds, "*")
  final_imputed_data[, numeric_cols] <- sweep(final_imputed_data[, numeric_cols, drop = FALSE],
                                              2, col_means, "+")
  
  # Convert back to data frame
  imputed_data <- as.data.frame(final_imputed_data)
  colnames(imputed_data) <- colnames(missData)

  for (col in fact_cols) {
    z <- round(imputed_data[[col]])
    z <- pmin(pmax(z, 0), 1)
    imputed_data[[col]] <- factor(z, levels = c(0, 1))
  }
  
  # Post-imputation: lm fitting
  estimation_matrix <- NULL
  
  if (lm.fit) {
    if (length(covariates) == 0) {
      lm_formula <- as.formula(paste(outcome, "~ 1"))
    } else {
      rhs <- paste(covariates, collapse = " + ")
      lm_formula <- as.formula(paste(outcome, "~", rhs))
    }
    
    lm_fit <- lm(lm_formula, data = imputed_data)
    coef_table <- summary(lm_fit)$coefficients
    ci <- confint(lm_fit)
    
    estimation_matrix <- cbind(
      Estimate   = coef_table[, "Estimate"],
      Std.Error  = coef_table[, "Std. Error"],
      `2.5 %`    = ci[, 1],
      `97.5 %`   = ci[, 2],
      `Pr(>|t|)` = coef_table[, "Pr(>|t|)"]
    )
  }
  
  return(list(
    imputed_data      = imputed_data,
    estimation_matrix = estimation_matrix
  ))
}
