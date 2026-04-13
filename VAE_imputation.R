# VAE-based imputation algorithm
# 12/14/2025

# Variational Autoencoder (VAE) Imputation 
# @missData: the dataframe with missing data (can take-in continuous & factor variables only)
# @outcome: name of the outcome variable, used for lm.fit (character)
# @covariates: vector of covariate names, used for lm.fit (character)
# @epochs: number of training epochs 
# @latent_dim: number of neurons in latent space 
# @hidden_dim: number of neurons in hidden layer
# @lr: learning rate
# @lm.fit: if TRUE, fit lm and return estimation matrix, otherwise only imputed data
# return: list(imputed_data, estimation_matrix)
VAE <- function(miss_data,
                outcome,
                covariates,
                epochs = 100,
                latent_dim = 8,
                hidden_dim = 32,
                lr = 0.001,
                verbose = FALSE,
                lm.fit = TRUE,
                seed = 123) {
  # Load required packages
  require(torch)
  set.seed(seed)
  torch_manual_seed(seed)
  
  if (verbose) {
    message("Preprocessing data...")
  }
  
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
  
  # Convert to torch tensor
  input_matrix <- as.matrix(filled_data)
  stored_matrix <- as.matrix(scaled_data)
  miss_mask <- is.na(scaled_data)
  
  x_tensor <- torch_tensor(input_matrix, dtype = torch_float())
  x_mask <- 1 * (!miss_mask)
  mask_tensor <- torch_tensor(x_mask, dtype = torch_float())
  
  vae_module <- nn_module(
    "VAE",
    
    initialize = function(input_dim, latent_dim, hidden_dim) {
      self$fc1 <- nn_linear(input_dim, hidden_dim)
      self$fc21 <- nn_linear(hidden_dim, latent_dim)
      self$fc22 <- nn_linear(hidden_dim, latent_dim)
      self$fc3 <- nn_linear(latent_dim, hidden_dim)
      self$fc4 <- nn_linear(hidden_dim, input_dim)
    },
    
    # Encoder
    encode = function(x) {
      h1 <- torch_relu(self$fc1(x))
      list(self$fc21(h1), self$fc22(h1))
    },
    
    # Reparameterization
    reparameterize = function(mu, logvar) {
      std <- torch_exp(0.5 * logvar)
      eps <- torch_randn_like(std)
      mu + eps * std
    },
    
    # Decode
    decode = function(z) {
      h3 <- torch_relu(self$fc3(z))
      self$fc4(h3)
    },
    
    forward = function(x) {
      encoded <- self$encode(x)
      z <- self$reparameterize(encoded[[1]], encoded[[2]])
      list(self$decode(z), encoded[[1]], encoded[[2]])
    }
  )
  
  # Initialize the VAE model
  input_dim <- ncol(input_matrix)
  vae <- vae_module(input_dim = input_dim, latent_dim = latent_dim, hidden_dim = hidden_dim)
  optimizer <- optim_adam(vae$parameters, lr = lr)
  
  # Define loss function
  vae_loss <- function(recon_x, x, mu, logvar, mask) {
    error_sq <- (recon_x - x)^2
    masked_error_sq <- error_sq * mask
    recon_loss <- torch_sum(masked_error_sq)
    kl_loss <- -0.5 * torch_sum(1 + logvar - mu$pow(2) - logvar$exp())
    
    beta_kl <- 0.1
    recon_loss + beta_kl * kl_loss
  }
  
  if (verbose) {
    message("Training VAE imputation model...")
  }
  
  # Train the model
  for (epoch in 1:epochs) {
    vae$train()
    optimizer$zero_grad()
    vae_output <- vae(x_tensor)
    loss <- vae_loss(vae_output[[1]], x_tensor, vae_output[[2]], vae_output[[3]], mask_tensor)
    loss$backward()
    optimizer$step()
  }
  
  vae$eval()
  vae_output <- vae(x_tensor)
  reconstructed_data <- as_array(vae_output[[1]])
  
  final_imputed_data <- stored_matrix
  missing_indices <- (x_mask == 0)
  final_imputed_data[missing_indices] <- reconstructed_data[missing_indices]
  
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
  
  # Convert back to data frame
  imputed_data <- as.data.frame(final_imputed_data)
  colnames(imputed_data) <- colnames(miss_data)
  
  # Decode one-hot back
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
  
  # Restore original column order
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
      Estimate   = coef_table[, "Estimate"],
      Std.Error  = coef_table[, "Std. Error"],
      `2.5 %`    = ci[, 1],
      `97.5 %`   = ci[, 2],
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
