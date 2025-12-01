# VAE-based imputation algorithm
# 11/17/2025

library(torch)

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
VAE <- function(missData,
                outcome,
                covariates,
                epochs = 100,
                latent_dim = 8,
                hidden_dim = 32,
                lr = 0.001,
                lm.fit = TRUE,
                seed = 123) {
  # Loads tensors in R
  require(torch)
  torch_manual_seed(seed)
  
  fact_cols <- names(missData)[sapply(missData, is.factor)]
  numeric_cols <- setdiff(names(missData), fact_cols)
  
  # Code factor variable to numeric matrix
  for (col in names(missData)) {
    if (is.factor(missData[[col]])) {
      if (nlevels(missData[[col]]) == 2) {
        missData[[col]] <- as.numeric(missData[[col]]) - 1
      } else {
        stop(paste("Column", col, "has more than 2 levels. Cannot encoding."))
      }
    }
  }
  
  x <- as.matrix(missData)
  
  # Standardization
  col_means <- colMeans(x[, numeric_cols, drop = FALSE], na.rm = TRUE)
  col_sds   <- apply(x[, numeric_cols, drop = FALSE], 2, sd, na.rm = TRUE)
  
  x_scaled <- x
  x_scaled[, numeric_cols] <- sweep(x_scaled[, numeric_cols], 2, col_means, "-")
  x_scaled[, numeric_cols] <- sweep(x_scaled[, numeric_cols], 2, col_sds, "/")
  
  # Initialize all missingness with 0 
  x_stored <- x_scaled
  x_filled <- x_scaled
  x_filled[is.na(x_scaled)] <- 0
  
  # Convert to torch tensor
  x_tensor <- torch_tensor(x_filled, dtype = torch_float())
  x_mask <- 1 * (!is.na(x_scaled))
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
    
    # Define the forward pass 
    forward = function(x) {
      encoded <- self$encode(x)
      z <- self$reparameterize(encoded[[1]], encoded[[2]])
      list(self$decode(z), encoded[[1]], encoded[[2]])
    }
  )
  
  # Initialize VAE model
  input_dim <- ncol(x)
  vae <- vae_module(input_dim = input_dim, latent_dim = latent_dim, hidden_dim = hidden_dim)
  optimizer <- optim_adam(vae$parameters, lr = lr) #Sets up the Adam optimizer
  
  # Define loss function
  vae_loss <- function(recon_x, x, mu, logvar, mask) {
    squared_diff <- (recon_x - x)^2
    masked_squared_diff <- squared_diff * mask 
    mse <- torch_sum(masked_squared_diff)
    kld <- -0.5 * torch_sum(1 + logvar - mu$pow(2) - logvar$exp())
    
    mse + kld 
  }
  
  # Train the model
  for (epoch in 1:epochs) {
    vae$train()
    optimizer$zero_grad() # Clear the gradient buffer
    out <- vae(x_tensor) # Apply the forward pass
    loss <- vae_loss(out[[1]], x_tensor, out[[2]], out[[3]], mask_tensor)
    loss$backward()
    optimizer$step() # Parameter Update
  }
  
  vae$eval()
  output <- vae(x_tensor)
  reconstructed <- as_array(output[[1]])
  
  final_imputed_data <- x_scaled
  missing_indices <- (x_mask == 0)
  final_imputed_data[missing_indices] <- reconstructed[missing_indices]
  final_imputed_data[, numeric_cols] <- sweep(final_imputed_data[, numeric_cols],
                                              2, col_sds, "*")
  final_imputed_data[, numeric_cols] <- sweep(final_imputed_data[, numeric_cols],
                                              2, col_means, "+")
  
  # Convert back to data frame
  imputed_data <- as.data.frame(final_imputed_data)
  colnames(imputed_data) <- colnames(missData)
  for (col in fact_cols) {
    imputed_data[[col]] <- round(imputed_data[[col]])
    imputed_data[[col]] <- factor(imputed_data[[col]])
  }
  
  # Post-imputation: lm fitting
  estimation_matrix <- NULL
  
  if (lm.fit) {
    rhs <- paste(covariates, collapse = " + ")
    lm_formula <- as.formula(paste(outcome, rhs, sep = " ~ "))
    
    lm_fit <- lm(lm_formula, data = imputed_data)
    lm_summary <- summary(lm_fit)
    
    coef_table <- lm_summary$coefficients
    ci <- confint(lm_fit)
    
    estimation_matrix <- cbind(
      Estimate   = coef_table[, "Estimate"],
      Std.Error  = coef_table[, "Std. Error"],
      `2.5 %`    = ci[, 1],
      `97.5 %`   = ci[, 2],
      `Pr(>|t|)` = coef_table[, "Pr(>|t|)"]
    )
  }
  
  
  result <- list(
    imputed_data      = imputed_data,
    estimation_matrix = estimation_matrix  # NULL if lm.fit = FALSE
  )
  
  return(result)
}


