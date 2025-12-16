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
MLP <- function(missData,
                outcome,
                covariates,
                epochs = 100,
                batch = 32,
                hidden_dim = 32,
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
  
  # Code factor variable
  fact_cols <- names(missData)[sapply(missData, is.factor)]
  numeric_cols <- setdiff(names(missData), fact_cols)

  original_cols <- colnames(missData)
  original_types <- sapply(missData, class)
  
  onehot_info <- list()
  encoded_data <- missData
  
  for (col in fact_cols) {
    vec <- missData[[col]]
    levs <- levels(vec)
    
    if (length(levs) > 1) {
      onehot_mat <- matrix(0, nrow = nrow(missData), ncol = length(levs))
      colnames(onehot_mat) <- paste0(col, "_", levs)
      
      for (i in seq_along(levs)) {
        is_match <- (vec == levs[i])
        onehot_mat[which(is_match), i] <- 1
      }
      onehot_mat[is.na(vec), ] <- NA
      
      onehot_info[[col]] <- list(cols = colnames(onehot_mat), levels = levs)
      
      encoded_data[[col]] <- NULL
      encoded_data <- cbind(encoded_data, onehot_mat)
    } else {
      encoded_data[[col]] <- 0
    }
  }
  
  missData <- as.data.frame(encoded_data)
  fact_cols <- character(0)
  numeric_cols <- colnames(missData)
  
  miss_mask <- is.na(missData)
  
  # Standardization
  x <- as.data.frame(missData)
  col_means <- sapply(x[, numeric_cols, drop = FALSE], mean, na.rm = TRUE)
  col_sds   <- sapply(x[, numeric_cols, drop = FALSE], sd,   na.rm = TRUE)
  col_sds[col_sds == 0] <- 1
  
  x_scaled <- x
  for (col in numeric_cols) {
    x_scaled[[col]] <- (x_scaled[[col]] - col_means[col]) / col_sds[col]
  }
  
  # Initialize with mean impute
  x_filled <- x_scaled
  for (j in seq_along(x_filled)) {
    m <- mean(x_filled[[j]], na.rm = TRUE)
    x_filled[[j]][is.na(x_filled[[j]])] <- m
  }
  
  
  x_mat <- as.matrix(x_filled)
  
  # Define MLP autoencoder
  p <- ncol(x_mat)
  
  model <- keras_model_sequential() %>%
    layer_dense(units = hidden_dim, activation = "relu", input_shape = p) %>%
    layer_dense(units = hidden_dim, activation = "relu") %>%
    layer_dense(units = p, activation = "linear")
  
  # Define Masked loss
  mask_mat <- 1 * (!miss_mask)
  mask_mat <- matrix(mask_mat, nrow = nrow(x_mat), ncol = p)
  y_train <- cbind(x_mat, mask_mat)
  
  masked_mse <- function(y_true, y_pred) {
    p <- as.integer(dim(y_pred)[2])
    x_true <- tensorflow::tf$slice(y_true, begin = c(0L, 0L), size = c(-1L, p))
    mask   <- tensorflow::tf$slice(y_true, begin = c(0L, p),  size = c(-1L, p))
    
    err2 <- tensorflow::tf$square(x_true - y_pred)
    num  <- tensorflow::tf$reduce_sum(err2 * mask)
    den  <- tensorflow::tf$reduce_sum(mask) + 1e-8
    num / den
  }
  
  # Model fit
  model %>% compile(
    loss = masked_mse,
    optimizer = optimizer_adam(learning_rate = lr)
  )
  
  model %>% fit(
    x = x_mat,
    y = y_train,
    epochs = epochs,
    batch_size = batch,
    verbose = 0
  )
  
  
  reconstructed <- model %>% predict(x_mat)
  final_imputed_data <- as.matrix(x_scaled)
  final_imputed_data[miss_mask] <- reconstructed[miss_mask]
  
  # Back-transform
  final_imputed_data[, numeric_cols] <- sweep(final_imputed_data[, numeric_cols, drop = FALSE],
                                              2, col_sds, "*")
  final_imputed_data[, numeric_cols] <- sweep(final_imputed_data[, numeric_cols, drop = FALSE],
                                              2, col_means, "+")

  imputed_data <- as.data.frame(final_imputed_data)
  colnames(imputed_data) <- colnames(missData)
  
  if (length(onehot_info) > 0) {
    for (col in names(onehot_info)) {
      cols <- onehot_info[[col]]$cols
      levs <- onehot_info[[col]]$levels
      onehot_block <- as.matrix(imputed_data[, cols, drop = FALSE])
      max_idx <- apply(onehot_block, 1, which.max)
      imputed_data[[col]] <- factor(levs[max_idx], levels = levs)
      imputed_data <- imputed_data[, !(names(imputed_data) %in% cols), drop = FALSE]
    }
  }
  
  imputed_data <- imputed_data[, original_cols, drop = FALSE]
  
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
