# ML-based Imputation Methods and Classic Imputation Methods
# 12/14/2025

# Complete Case Analysis (CCA)
# @missData: the dataframe with missing data (can take-in continuous & factor variables only)
# @outcome: name of the outcome variable, used for lm.fit (character)
# @covariates: vector of covariate names, used for lm.fit (character)
# return: imputed data
CCA <- function(missData,
                outcome,
                covariates) {
  require(stats)
  
  if (length(covariates) == 0) {
    formula <- as.formula(paste(outcome, "~ 1"))
  } else {
    formula <- as.formula(paste(outcome, "~", paste(covariates, collapse = " + ")))
  }
  
  completeData <- missData[complete.cases(missData), ]
  completeData
}



# MICE Imputation
# @missData: the dataframe with missing data (can take-in continuous & factor variables only)
# @outcome: name of the outcome variable, used for lm.fit (character)
# @covariates: vector of covariate names, used for lm.fit (character)
# @m: number of imputations, by default 50
# @maxit: number of maximum iterations, by default 40
# return: pooled results using Rubin’s Rules
MICE <- function(missData,
                 outcome,
                 covariates,
                 m = 50,
                 maxit = 40) {
  require(mice)
  require(stats)
  
  if (length(covariates) == 0) {
    formula <- as.formula(paste(outcome, "~ 1"))
  } else {
    formula <- as.formula(paste(outcome, "~", paste(covariates, collapse = " + ")))
  }
  
  vars <- all.vars(formula)
  
  methods <- make.method(missData)
  
  for (v in vars) {
    if (anyNA(missData[[v]])) {
      methods[v] <- "norm"
    }
  }
  
  non_formula_vars <- setdiff(names(missData), vars)
  methods[non_formula_vars] <- ""
  
  imputedData <- mice(missData, method = methods, m = m, maxit = maxit, printFlag = FALSE)
  
  formula_str <- deparse(formula)
  fit <- with(imputedData, eval(bquote(lm(.(as.formula(formula_str))))))
  
  pooled <- pool(fit)
  return(pooled)
}




# KNN Imputation
# @missData: the dataframe with missing data (can take-in continuous & factor variables only)
# @outcome: name of the outcome variable, used for lm.fit (character)
# @covariates: vector of covariate names, used for lm.fit (character)
# @k: number of nearest neighbors considered
# return: imputed data
KNN <- function(missData,
                outcome,
                covariates,
                k = 5) {
  require(VIM)
  require(stats)
  
  if (length(covariates) == 0) {
    formula <- as.formula(paste(outcome, "~ 1"))
  } else {
    formula <- as.formula(paste(outcome, "~", paste(covariates, collapse = " + ")))
  }
  
  vars <- all.vars(formula)
  miss_vars <- vars[colSums(is.na(missData[, vars, drop = FALSE])) > 0]
  
  knnImpute <- VIM::kNN(missData, variable = miss_vars, k = k, imp_var = FALSE)
  
  knnImpute
}


# SVM Imputation
# @missData: the dataframe with missing data (can take-in continuous & factor variables only)
# @outcome: name of the outcome variable, used for lm.fit (character)
# @covariates: vector of covariate names, used for lm.fit (character)
# return: imputed data
SVM <- function(missData, outcome, covariates) {
  require(e1071)
  require(stats)
  
  if (length(covariates) == 0) {
    formula <- as.formula(paste(outcome, "~ 1"))
  } else {
    formula <- as.formula(paste(outcome, "~", paste(covariates, collapse = " + ")))
  }
  
  imputed_data <- missData
  
  svm_data <- missData

  # Code factor variable
  bin_factor_levels <- list()
  for (col in names(svm_data)) {
    if (is.factor(svm_data[[col]])) {
      if (nlevels(svm_data[[col]]) == 2) {
        bin_factor_levels[[col]] <- levels(svm_data[[col]])
        svm_data[[col]] <- as.numeric(svm_data[[col]]) - 1
      } else {
        stop(paste("Column", col, "has >2 levels. Cannot encode for SVM:", col))
      }
    }
  }
  
  vars <- all.vars(formula)
  miss_vars <- vars[colSums(is.na(svm_data[, vars, drop = FALSE])) > 0]
  
  # Impute each missing variable
  for (target_var in miss_vars) {
    predictors <- setdiff(names(svm_data), target_var)
    
    train_rows <- complete.cases(svm_data[, predictors]) & !is.na(svm_data[[target_var]])
    test_rows  <- complete.cases(svm_data[, predictors]) &  is.na(svm_data[[target_var]])
    
    if (sum(train_rows) > 1 && sum(test_rows) > 0) {
      X_train <- svm_data[train_rows, predictors, drop = FALSE]
      y_train <- svm_data[train_rows, target_var]
      X_test  <- svm_data[test_rows, predictors, drop = FALSE]
      
      model_svm <- e1071::svm(X_train, y_train)
      pred_vals <- predict(model_svm, X_test)

      svm_data[test_rows, target_var] <- pred_vals
    }
  }
  
  # Convert back to original format
  for (col in names(imputed_data)) {
    if (col %in% names(bin_factor_levels)) {
      levs <- bin_factor_levels[[col]]
      z <- round(svm_data[[col]])
      z <- pmin(pmax(z, 0), 1)
      imputed_data[[col]] <- factor(levs[z + 1], levels = levs)
    } else {
      
      imputed_data[[col]] <- svm_data[[col]]
    }
  }
  
  imputed_data
}



# Random Forest Imputation
# @missData: the dataframe with missing data (can take-in continuous & factor variables only)
# @outcome: name of the outcome variable, used for lm.fit (character)
# @covariates: vector of covariate names, used for lm.fit (character)
# @m: number of imputations
# @maxit: number of maximum iterations
# @ntree: numbers of trees
# return: pooled results using Rubin’s Rules
RF <- function(missData,
               outcome,
               covariates,
               m = 15,
               maxit = 10,
               ntree = 50) {
  require(mice)
  require(stats)
  
  if (length(covariates) == 0) {
    formula <- as.formula(paste(outcome, "~ 1"))
  } else {
    formula <- as.formula(paste(outcome, "~", paste(covariates, collapse = " + ")))
  }
  
  model_vars <- all.vars(formula)
  
  methods <- make.method(missData)
  
  for (v in model_vars) {
    if (anyNA(missData[[v]])) {
      methods[v] <- "rf"
    }
  }
  
  non_model_vars <- setdiff(names(missData), model_vars)
  methods[non_model_vars] <- ""
  
  imputedData <- mice(
    missData,
    method = methods,
    m = m,
    maxit = maxit,
    printFlag = FALSE,
    ntree = ntree
  )
  
  formula_str <- deparse(formula)
  fit <- with(imputedData, eval(bquote(lm(.(as.formula(formula_str))))))
  
  pooled <- pool(fit)
  
  return(pooled)
}
