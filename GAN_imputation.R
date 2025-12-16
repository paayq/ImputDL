# GAN-based imputation algorithm
# 12/14/2025

# Generative Adversarial Imputation Network (GAIN)
# @missData: dataframe with missing data (can take-in continuous & factor variables)
# @outcome: name of the outcome variable, used for lm.fit (character)
# @covariates: vector of covariate names, used for lm.fit (character)
# @batch_size: number of samples per training batch
# @hint_rate: proportion of observed data used as hints to the discriminator
# @alpha: weighting parameter for reconstruction loss
# @iterations: number of training iterations
# @learning_rate: learning rate for Adam optimizer
# @verbose: whether to display training progress
# @lm.fit: if TRUE, fit lm and return estimation matrix, otherwise only imputed data
# @seed: random seed
# return: list(imputed_data, estimation_matrix)
GAN <- function(missData,
                outcome,
                covariates,
                batch_size = 16,
                hint_rate = 1,
                alpha = 100,
                iterations = 5000,
                learning_rate = 0.001,
                verbose = TRUE,
                lm.fit = TRUE,
                seed = 123) {
  
  require(tensorflow)
  tf <- tensorflow::tf
  set.seed(seed)
  tf$compat$v1$set_random_seed(as.integer(seed))
  
  # Use TF v1 session mode
  tf$compat$v1$disable_v2_behavior()
  
  original_cols <- colnames(missData)
  original_types <- sapply(missData, class)
  
  # Code factor variable
  factor_cols <- sapply(missData, is.factor)
  factor_names <- names(missData)[factor_cols]
  
  encoded_data <- missData
  onehot_info <- list()
  
  for (col in factor_names) {
    vec <- missData[[col]]
    levs <- levels(vec)
    
    if (length(levs) > 1) {
      onehot_mat <- matrix(0, nrow = length(vec), ncol = length(levs))
      colnames(onehot_mat) <- paste0(col, "_", levs)
      
      for (i in seq_along(levs)) {
        is_match <- (vec == levs[i])
        onehot_mat[which(is_match), i] <- 1
      }
      
      onehot_info[[col]] <- list(cols = colnames(onehot_mat), levels = levs)
      
      encoded_data <- encoded_data[, !(colnames(encoded_data) %in% col), drop = FALSE]
      encoded_data <- cbind(encoded_data, onehot_mat)
    } else {
      encoded_data[[col]] <- 0
    }
  }
  
  # Utilities: normalization + renormalization
  normalization <- function(data_x) {
    X <- as.matrix(data_x)
    ncolx <- ncol(X)
    mins <- numeric(ncolx)
    maxs <- numeric(ncolx)
    norm_X <- matrix(NA_real_, nrow = nrow(X), ncol = ncolx)
    
    for (j in seq_len(ncolx)) {
      col <- X[, j]
      mins[j] <- min(col, na.rm = TRUE)
      maxs[j] <- max(col, na.rm = TRUE)
      rng <- maxs[j] - mins[j]
      
      if (is.na(rng) || rng == 0) norm_X[, j] <- 0 else norm_X[, j] <- (col - mins[j]) / rng
    }
    
    list(norm_data = norm_X,
         norm_parameters = list(mins = mins, maxs = maxs))
  }
  
  renormalization <- function(norm_data, params) {
    mins <- params$mins
    maxs <- params$maxs
    rngs <- maxs - mins
    out <- matrix(NA_real_, nrow = nrow(norm_data), ncol = ncol(norm_data))
    
    for (j in seq_len(ncol(norm_data))) {
      if (is.na(rngs[j]) || rngs[j] == 0) out[, j] <- mins[j] else out[, j] <- norm_data[, j] * rngs[j] + mins[j]
    }
    out
  }
  
  # Initialize xavier, samplers and batch index
  xavier_init <- function(shape_vec) {
    in_dim <- as.numeric(shape_vec[1])
    stddev <- 1 / sqrt(in_dim / 2.0)
    tf$random$normal(shape = as.integer(shape_vec), stddev = stddev, dtype = tf$float32)
  }
  
  uniform_sampler <- function(low, high, n, dim) {
    matrix(runif(n * dim, min = low, max = high), nrow = n, ncol = dim)
  }
  
  binary_sampler <- function(p, n, dim) {
    matrix(rbinom(n * dim, size = 1, prob = p), nrow = n, ncol = dim)
  }
  
  sample_batch_index <- function(no, batch_size) {
    if (batch_size <= no) sample(seq_len(no), batch_size, replace = FALSE)
    else sample(seq_len(no), batch_size, replace = TRUE)
  }
  
  
  # Data prepare
  data_x <- as.data.frame(encoded_data)
  data_mat <- as.matrix(data_x)
  data_m <- 1 - is.na(data_mat) # mask: 1 observed, 0 missing
  no <- nrow(data_mat)
  dim <- ncol(data_mat)
  h_dim <- dim
  
  true_numeric_cols <- which(
    sapply(encoded_data, is.numeric) &
      !(colnames(encoded_data) %in% unlist(lapply(onehot_info, function(x) x$cols)))
  )
  
  # Normalization
  if (length(true_numeric_cols) > 0) {
    norm_res <- normalization(data_mat[, true_numeric_cols, drop = FALSE])
    norm_data_x <- data_mat
    norm_data_x[, true_numeric_cols] <- norm_res$norm_data
  } else {
    norm_res <- NULL
    norm_data_x <- data_mat
  }
  
  norm_data_x[is.na(norm_data_x)] <- 0
  
  norm_data_x <- norm_data_x + matrix(
    rnorm(length(norm_data_x), mean = 0, sd = 1e-6),
    nrow = nrow(norm_data_x),
    ncol = ncol(norm_data_x)
  )
  
  # TensorFlow graph
  tf$compat$v1$reset_default_graph()
  
  X <- tf$compat$v1$placeholder(tf$float32, shape = shape(NULL, dim), name = "X")
  M <- tf$compat$v1$placeholder(tf$float32, shape = shape(NULL, dim), name = "M")
  H <- tf$compat$v1$placeholder(tf$float32, shape = shape(NULL, dim), name = "H")
  
  # Discriminator variables
  D_W1 <- tf$Variable(xavier_init(c(dim * 2L, h_dim)), dtype = tf$float32)
  D_b1 <- tf$Variable(tf$zeros(shape(as.integer(h_dim))), dtype = tf$float32)
  D_W2 <- tf$Variable(xavier_init(c(h_dim, h_dim)), dtype = tf$float32)
  D_b2 <- tf$Variable(tf$zeros(shape(as.integer(h_dim))), dtype = tf$float32)
  D_W3 <- tf$Variable(xavier_init(c(h_dim, dim)), dtype = tf$float32)
  D_b3 <- tf$Variable(tf$zeros(shape(as.integer(dim))), dtype = tf$float32)
  
  # Generator variables
  G_W1 <- tf$Variable(xavier_init(c(dim * 2L, h_dim)), dtype = tf$float32)
  G_b1 <- tf$Variable(tf$zeros(shape(as.integer(h_dim))), dtype = tf$float32)
  G_W2 <- tf$Variable(xavier_init(c(h_dim, h_dim)), dtype = tf$float32)
  G_b2 <- tf$Variable(tf$zeros(shape(as.integer(h_dim))), dtype = tf$float32)
  G_W3 <- tf$Variable(xavier_init(c(h_dim, dim)), dtype = tf$float32)
  G_b3 <- tf$Variable(tf$zeros(shape(as.integer(dim))), dtype = tf$float32)
  
  # Generator network
  generator <- function(x, m) {
    inputs <- tf$concat(list(x, m), axis = 1L)
    G_h1 <- tf$nn$relu(tf$matmul(inputs, G_W1) + G_b1)
    G_h2 <- tf$nn$relu(tf$matmul(G_h1, G_W2) + G_b2)
    G_logit <- tf$matmul(G_h2, G_W3) + G_b3
    G_logit <- tf$clip_by_value(G_logit, -10.0, 10.0)
    tf$nn$sigmoid(G_logit)
  }
  
  # Discriminator network
  discriminator <- function(x, h) {
    inputs <- tf$concat(list(x, h), axis = 1L)
    D_h1 <- tf$nn$relu(tf$matmul(inputs, D_W1) + D_b1)
    D_h2 <- tf$nn$relu(tf$matmul(D_h1, D_W2) + D_b2)
    D_logit <- tf$matmul(D_h2, D_W3) + D_b3
    D_logit <- tf$clip_by_value(D_logit, -10.0, 10.0)
    tf$nn$sigmoid(D_logit)
  }
  
  # GAIN structure
  G_sample <- generator(X, M)
  Hat_X <- X * M + G_sample * (1 - M)
  D_prob <- discriminator(Hat_X, H)
  
  # Losses
  eps <- 1e-8
  D_loss <- -tf$reduce_sum(
    (1 - H) * (
      M * tf$math$log(D_prob + eps) +
        (1 - M) * tf$math$log(1 - D_prob + eps)
    )
  ) / (tf$reduce_sum(1 - H) + eps)
  
  G_loss_temp <- -tf$reduce_mean((1 - M) * tf$math$log(D_prob + eps))
  MSE_loss <- tf$reduce_mean((M * X - M * G_sample) ^ 2) / tf$maximum(tf$reduce_mean(M), 1e-8)
  G_loss <- G_loss_temp + alpha * MSE_loss
  
  # Optimizers (with gradient clipping)
  D_vars <- list(D_W1, D_W2, D_W3, D_b1, D_b2, D_b3)
  G_vars <- list(G_W1, G_W2, G_W3, G_b1, G_b2, G_b3)
  
  optimizer <- tf$compat$v1$train$AdamOptimizer(learning_rate)
  
  gvs_D <- optimizer$compute_gradients(D_loss, var_list = D_vars)
  capped_gvs_D <- lapply(gvs_D, function(gv) {
    if (!is.null(gv[[1]])) list(tf$clip_by_value(gv[[1]], -1.0, 1.0), gv[[2]]) else gv
  })
  D_solver <- optimizer$apply_gradients(capped_gvs_D)
  
  gvs_G <- optimizer$compute_gradients(G_loss, var_list = G_vars)
  capped_gvs_G <- lapply(gvs_G, function(gv) {
    if (!is.null(gv[[1]])) list(tf$clip_by_value(gv[[1]], -1.0, 1.0), gv[[2]]) else gv
  })
  G_solver <- optimizer$apply_gradients(capped_gvs_G)
  
  # Session
  sess <- tf$compat$v1$Session()
  sess$run(tf$compat$v1$global_variables_initializer())
  
  # Training loop
  if (verbose) {
    pb <- txtProgressBar(min = 0, max = iterations, style = 3)
    on.exit(close(pb), add = TRUE)
  }
  
  for (it in seq_len(iterations)) {
    batch_idx <- sample_batch_index(no, batch_size)
    X_mb <- norm_data_x[batch_idx, , drop = FALSE]
    M_mb <- data_m[batch_idx, , drop = FALSE]
    
    Z_mb <- uniform_sampler(0, 0.01, n = nrow(X_mb), dim = dim)
    H_mb_temp <- binary_sampler(hint_rate, n = nrow(X_mb), dim = dim)
    H_mb <- M_mb * H_mb_temp
    X_mb_in <- M_mb * X_mb + (1 - M_mb) * Z_mb
    
    sess$run(D_solver, feed_dict = dict(X = X_mb_in, M = M_mb, H = H_mb))
    res <- sess$run(list(G_solver, G_loss_temp, MSE_loss),
                    feed_dict = dict(X = X_mb_in, M = M_mb, H = H_mb))
    
    if (any(is.nan(unlist(res)))) {
      warning("Run into NaN, training stopped.")
      break
    }
    
    if (verbose && (it %% 100 == 0 || it == iterations)) {
      setTxtProgressBar(pb, it)
    }
  }
  if (verbose) setTxtProgressBar(pb, iterations)
  
  # Impute
  Z_mb <- uniform_sampler(0, 0.01, no, dim)
  X_mb_in <- data_m * norm_data_x + (1 - data_m) * Z_mb
  
  imputed_data_norm <- sess$run(G_sample, feed_dict = dict(X = X_mb_in, M = data_m))
  imputed_data_norm <- data_m * norm_data_x + (1 - data_m) * imputed_data_norm
  
  # Renormalize only true numeric columns
  imputed_data <- matrix(0, nrow = no, ncol = ncol(data_mat))
  if (length(true_numeric_cols) > 0 && !is.null(norm_res)) {
    imputed_data[, true_numeric_cols] <- renormalization(
      imputed_data_norm[, true_numeric_cols, drop = FALSE],
      norm_res$norm_parameters
    )
  }
  if (length(true_numeric_cols) < ncol(data_mat)) {
    other_cols <- setdiff(seq_len(ncol(data_mat)), true_numeric_cols)
    imputed_data[, other_cols] <- imputed_data_norm[, other_cols, drop = FALSE]
  }
  
  # Convert back to original format
  imputed_df <- as.data.frame(imputed_data)
  colnames(imputed_df) <- colnames(encoded_data)
  
  for (col in names(onehot_info)) {
    onehot_cols_names <- onehot_info[[col]]$cols
    if (all(onehot_cols_names %in% colnames(imputed_df))) {
      onehot_values <- as.matrix(imputed_df[, onehot_cols_names, drop = FALSE])
      max_indices <- apply(onehot_values, 1, which.max)
      
      imputed_df[[col]] <- factor(onehot_info[[col]]$levels[max_indices],
                                  levels = onehot_info[[col]]$levels)
      imputed_df <- imputed_df[, !(colnames(imputed_df) %in% onehot_cols_names), drop = FALSE]
    }
  }
  
  final_df <- as.data.frame(matrix(NA, nrow = no, ncol = length(original_cols)))
  colnames(final_df) <- original_cols
  
  for (col in original_cols) {
    if (col %in% colnames(imputed_df)) {
      final_df[[col]] <- imputed_df[[col]]
      if (original_types[col] %in% c("integer", "numeric")) {
        final_df[[col]] <- as.numeric(final_df[[col]])
      } else if (original_types[col] == "factor") {
        final_df[[col]] <- factor(final_df[[col]], levels = levels(missData[[col]]))
      }
    }
  }
  
  sess$close()
  
  # Post-imputation: lm fitting
  estimation_matrix <- NULL
  
  if (lm.fit) {
    if (length(covariates) == 0) {
      lm_formula <- as.formula(paste(outcome, "~ 1"))
    } else {
      rhs <- paste(covariates, collapse = " + ")
      lm_formula <- as.formula(paste(outcome, "~", rhs))
    }
    
    lm_fit <- lm(lm_formula, data = final_df)
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
    imputed_data      = final_df,
    estimation_matrix = estimation_matrix
  ))
}
