# MLP-based imputation algorithm
# 12/14/2025

# Multilayer Perceptron (MLP) Imputation
# @missData: the dataframe with missing data (can take-in continuous & factor variables only)
# @outcome: name of the outcome variable, used for lm.fit (character)
# @covariates: vector of covariate names, used for lm.fit (character)
# @epochs: number of training epochs 
# @batch: batch size
# @hidden_dim: number of neurons in hidden layer
# @lr: learning rate
# @lm.fit: if TRUE, fit lm and return estimation matrix, otherwise only imputed data
# @seed: random seed
# return: list(imputed_data, estimation_matrix)
MLP <- function(miss_data,
                outcome,
                covariates,
                epochs = 100,
                batch = 32,
                hidden_dim = 32,
                lr = 0.001,
                verbose = FALSE,
                lm.fit = TRUE,
                seed = 123) {
  
  # Load required packages
  require(keras3)
  require(tensorflow)
  set.seed(seed)
  tensorflow::tf$random$set_seed(seed)
  
  if (!tensorflow::tf$executing_eagerly()) {
    tensorflow::tf$compat$v1$enable_eager_execution()
  }
  
  if (verbose) {
    message("Preprocessing data...")
  }
  
  # Encode factor variables
  factor_cols <- names(miss_data)[sapply(miss_data, is.factor)]
  numeric_cols_raw <- names(miss_data)[sapply(miss_data, is.numeric)]
  
  original_data <- miss_data
  original_cols <- colnames(miss_data)
  original_types <- vapply(miss_data, function(x) class(x)[1], character(1))
  
  onehot_info <- list()
  encoded_data <- miss_data
  
  for (col in factor_cols) {
    col_values <- miss_data[[col]]
    factor_levels <- levels(col_values)
    
    if (length(factor_levels) > 1) {
      onehot_mat <- matrix(0, nrow = nrow(miss_data), ncol = length(factor_levels))
      colnames(onehot_mat) <- paste0(col, "_", factor_levels)
      
      for (level_idx in seq_along(factor_levels)) {
        is_match <- (col_values == factor_levels[level_idx])
        onehot_mat[which(is_match), level_idx] <- 1
      }
      onehot_mat[is.na(col_values), ] <- NA
      
      onehot_info[[col]] <- list(cols = colnames(onehot_mat), levels = factor_levels)
      
      encoded_data[[col]] <- NULL
      encoded_data <- cbind(encoded_data, onehot_mat)
    } else {
      encoded_data[[col]] <- 0
    }
  }
  
  miss_data <- as.data.frame(encoded_data)
  numeric_cols <- intersect(numeric_cols_raw, colnames(miss_data))
  
  miss_mask <- is.na(miss_data)
  
  # Standardize original numeric variables
  input_data <- as.data.frame(miss_data)
  scaled_data <- input_data
  
  if (length(numeric_cols) > 0) {
    col_means <- sapply(input_data[, numeric_cols, drop = FALSE], mean, na.rm = TRUE)
    col_sds <- sapply(input_data[, numeric_cols, drop = FALSE], sd, na.rm = TRUE)
    col_sds[!is.finite(col_sds) | col_sds == 0] <- 1
    
    for (col in numeric_cols) {
      scaled_data[[col]] <- (scaled_data[[col]] - col_means[col]) / col_sds[col]
    }
  } else {
    col_means <- numeric(0)
    col_sds <- numeric(0)
  }
  
  # Initialize missing values with column means
  filled_data <- scaled_data
  for (col_idx in seq_along(filled_data)) {
    col_mean <- mean(filled_data[[col_idx]], na.rm = TRUE)
    filled_data[[col_idx]][is.na(filled_data[[col_idx]])] <- col_mean
  }
  
  input_matrix <- as.matrix(filled_data)
  
  # Define the MLP autoencoder
  n_features <- ncol(input_matrix)
  
  inputs <- layer_input(shape = n_features)
  outputs <- inputs %>%
    layer_dense(units = hidden_dim, activation = "relu") %>%
    layer_dense(units = hidden_dim, activation = "relu") %>%
    layer_dense(units = n_features, activation = "linear")
  model <- keras_model(inputs = inputs, outputs = outputs)
  
  # Define the masked loss
  mask_mat <- 1 * (!miss_mask)
  y_train <- cbind(input_matrix, mask_mat)
  
  n_feat <- n_features
  
  masked_mse <- function(y_true, y_pred) {
    observed_values <- tensorflow::tf$slice(
      y_true, begin = c(0L, 0L), size = c(-1L, as.integer(n_feat))
    )
    observed_mask <- tensorflow::tf$slice(
      y_true, begin = c(0L, as.integer(n_feat)), size = c(-1L, as.integer(n_feat))
    )
    error_sq <- tensorflow::tf$square(observed_values - y_pred)
    loss_sum <- tensorflow::tf$reduce_sum(error_sq * observed_mask)
    mask_sum <- tensorflow::tf$reduce_sum(observed_mask) + 1e-8
    loss_sum / mask_sum
  }
  
  # Train the model
  model %>% compile(
    loss = masked_mse,
    optimizer = optimizer_adam(learning_rate = lr)
  )
  
  if (verbose) {
    message("Training MLP imputation model...")
  }
  
  model %>% fit(
    x = input_matrix,
    y = y_train,
    epochs = epochs,
    batch_size = batch,
    verbose = 0
  )
  
  reconstructed <- model %>% predict(input_matrix)
  final_imputed_data <- as.matrix(scaled_data)
  final_imputed_data[miss_mask] <- reconstructed[miss_mask]
  
  # Restore numeric variables to the original scale
  if (length(numeric_cols) > 0) {
    final_imputed_data[, numeric_cols] <- sweep(
      final_imputed_data[, numeric_cols, drop = FALSE],
      2,
      col_sds,
      "*"
    )
    final_imputed_data[, numeric_cols] <- sweep(
      final_imputed_data[, numeric_cols, drop = FALSE],
      2,
      col_means,
      "+"
    )
  }
  
  imputed_data <- as.data.frame(final_imputed_data)
  colnames(imputed_data) <- colnames(miss_data)
  
  if (length(onehot_info) > 0) {
    for (col in names(onehot_info)) {
      onehot_cols <- onehot_info[[col]]$cols
      factor_levels <- onehot_info[[col]]$levels
      onehot_block <- as.matrix(imputed_data[, onehot_cols, drop = FALSE])
      max_idx <- apply(onehot_block, 1, which.max)
      imputed_data[[col]] <- factor(factor_levels[max_idx], levels = factor_levels)
      imputed_data <- imputed_data[, !(names(imputed_data) %in% onehot_cols), drop = FALSE]
    }
  }
  
  final_data <- as.data.frame(matrix(NA, nrow = nrow(imputed_data), ncol = length(original_cols)))
  colnames(final_data) <- original_cols
  
  for (col in original_cols) {
    if (!col %in% names(imputed_data)) next
    
    if (original_types[[col]] == "factor") {
      final_data[[col]] <- factor(
        imputed_data[[col]],
        levels = levels(original_data[[col]])
      )
    } else if (original_types[[col]] == "integer") {
      final_data[[col]] <- as.integer(round(imputed_data[[col]]))
    } else if (original_types[[col]] == "numeric") {
      final_data[[col]] <- as.numeric(imputed_data[[col]])
    } else if (original_types[[col]] == "logical") {
      final_data[[col]] <- as.logical(imputed_data[[col]])
    } else {
      final_data[[col]] <- imputed_data[[col]]
    }
  }
  
  imputed_data <- final_data
  
  # Optional: fit the linear regression model
  estimation_matrix <- NULL
  
  if (lm.fit) {
    if (length(covariates) == 0) {
      lm_formula <- as.formula(paste(outcome, "~ 1"))
    } else {
      formula_rhs <- paste(covariates, collapse = " + ")
      lm_formula <- as.formula(paste(outcome, "~", formula_rhs))
    }
    
    lm_fit <- lm(lm_formula, data = imputed_data)
    coef_table <- summary(lm_fit)$coefficients
    ci <- confint(lm_fit)
    
    estimation_matrix <- cbind(
      Estimate = coef_table[, "Estimate"],
      Std.Error = coef_table[, "Std. Error"],
      `2.5 %` = ci[, 1],
      `97.5 %` = ci[, 2],
      `Pr(>|t|)` = coef_table[, "Pr(>|t|)"]
    )
  }
  
  if (verbose) {
    message("Done.")
  }
  
  return(list(
    imputed_data = imputed_data,
    estimation_matrix = estimation_matrix
  ))
}
