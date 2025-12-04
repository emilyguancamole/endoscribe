# rf_functions.R
# -----------------------------------------------------------------------------
# Author:             Albert Kuo
# Date last modified: Jan 28, 2018
#
# Helper functions used in rf_model.Rmd and rf_model_supplement.Rmd

library(pacman)
p_load(tidyverse, randomForest, gridExtra, ROCR, xgboost)

# Get importance scores from random forest model
get_imp_scores = function(rf_model){
  imp_scores = rf_model$importance
  imp_names = rownames(imp_scores)
  
  out = data.frame(imp_names, imp_scores)
  names(out) = c("imp_names", "imp_scores")
  return(out)
}

# Plot votes given random forest model and training data
plot_votes = function(rf_model, train_dat, y_scale = "count"){
  votes_df = data.frame(votes = rf_model$votes[, 1],
                        pep = train_dat$pep) 
  normal_votes = votes_df %>%
    filter(pep == 0) %>%
    select(votes) %>%
    unlist()
  split1 = quantile(normal_votes, 0.7)
  split2 = quantile(normal_votes, 0.95)
  levels(votes_df$pep) = paste("PEP =", levels(votes_df$pep))
  out = ggplot(votes_df)
  if(y_scale == "count"){
    out = out + geom_histogram(aes(x = votes, y = ..count..), bins = 30) +
      ggtitle("Histogram (counts)")
  } else {
    out = out + geom_histogram(aes(x = votes, y = ..density..), bins = 30) +
      ggtitle("Histogram (density):")
  }
  ypos_text = layer_scales(out)$y$range$range[2]*0.7 
  out = out +
    geom_vline(xintercept = split1, linetype = 2, color = "#D55E00") + 
    geom_text(aes(x = split1, y = ypos_text, label = "70th percentile"),
              color = "#D55E00", size = 3, angle = 90, vjust = -1) +
    geom_vline(xintercept = split2, linetype = 2, color = "#D55E00") +
    geom_text(aes(x = split2, y = ypos_text, label = "95th percentile"),
              color = "#D55E00", size = 3, angle = 90, vjust = -1) +
    facet_grid(pep ~ .) + 
    theme_bw()
  return(out)
}

# Cross validation on training set with specified model
## random forest = model "rf"
## logistic regression = model "lr"
## boosting = model "boost"
get_ROC = function(train_dat, model = "rf", folds = 10, max_depth = 6, eta = 0.3, nrounds = 100){
  # Cross-validation
  sens_vec = spec_vec = cutoff_vec = c()
  folds = createFolds(train_dat$pep, k = folds, list = F)
  predicted_values = actual_values = c()
  
  for(fold in unique(folds)){
    # Separate into train and test
    train = train_dat[folds!=fold,]
    test = train_dat[folds==fold,]
    
    # Model predictions
    if(model == "rf"){
      sample_size = min(table(train$pep)) # num of obs for bootstrap samples per class
      model_fold = randomForest(pep ~ ., data = train, proximity = T, type="classification",
                                strata = train$pep, sampsize = c(sample_size, sample_size))
      predicted_values = c(predicted_values, predict(model_fold, test, type="prob")[, 1])
    } else if (model == "lr"){
      is_unary = function(x) length(unique(x))==1
      train = train[sapply(train, function(x) !is_unary(x))]
      # Logistic regression predictions
      model_fold = glm(as.numeric(as.character(pep)) ~ ., data = train, family = binomial(link = "logit"))
      predicted_values = c(predicted_values, predict(model_fold, test, type="response"))
    } else if (model == "boost"){
      train_ls = list()
      train_ls$data = as.matrix(sapply(train %>% select(-pep), as.numeric))
      train_ls$label = sapply(as.numeric(train$pep), function(x) ifelse(x==2, 0, 1))
      test_ls = list()
      test_ls$data = as.matrix(sapply(test %>% select(-pep), as.numeric))
      dtrain = xgb.DMatrix(data = train_ls$data, label = train_ls$label)
      model_fold = xgboost(data = dtrain, verbose = 0, max_depth = max_depth,
                           eta = eta, nrounds = nrounds, objective = "binary:logistic")
      predicted_values = c(predicted_values, predict(model_fold, test_ls$data))
    }
    actual_values = c(actual_values, test$pep)
  }
  actual_values = sapply(as.numeric(actual_values), function(x) ifelse(x==2, 0, 1))
  pred = prediction(predicted_values, actual_values)
  perf = performance(pred, "sens", "spec")
  acc = performance(pred, "acc")@y.values[[1]]
  auc = as.numeric(performance(pred, "auc")@y.values)
  cutoff_vec = c(cutoff_vec, perf@alpha.values[[1]])
  sens_vec = c(sens_vec, perf@y.values[[1]])
  spec_vec = c(spec_vec, perf@x.values[[1]])
  
  sens_spec_cv_df = data.frame(sensitivity = sens_vec,
                               specificity = spec_vec,
                               cutoff = cutoff_vec,
                               model_name = model,
                               auc = auc,
                               acc = acc)
  
  return(sens_spec_cv_df)
}

# Prediction on test set ROC
pred_ROC = function(test_dat, rf_model){
  sens_vec = spec_vec = cutoff_vec = c()
  actual_values = test_dat$pep
  predicted_values = predict(rf_model, test_dat, type = "prob")[, 1]
  pred = prediction(predicted_values, actual_values)
  perf = performance(pred, "sens", "spec")
  acc = performance(pred, "acc")@y.values[[1]]
  auc = as.numeric(performance(pred, "auc")@y.values)
  cutoff_vec = c(cutoff_vec, perf@alpha.values[[1]])
  sens_vec = c(sens_vec, perf@y.values[[1]])
  spec_vec = c(spec_vec, perf@x.values[[1]])
  
  sens_spec_df = data.frame(model_name = "Test set",
                            sensitivity = sens_vec,
                            specificity = spec_vec,
                            cutoff = cutoff_vec,
                            auc = auc,
                            acc = acc)
  return(sens_spec_df)
}

# Plot ROC curve
plot_ROC = function(sens_spec_cv_df){
  sens_spec_cv_df = sens_spec_cv_df %>%
    mutate(model_name = as.factor(model_name))
  
  plt = ggplot(sens_spec_cv_df, aes(x = specificity, y = sensitivity, color = model_name)) +
    geom_point() + 
    geom_line() +
    labs(title = "10-Fold Cross-Validation Sensitivity and Specificity") + 
    annotate(geom = "text", x = 0.85, y = 1, label = paste("(Mean) AUC =", round(mean(sens_spec_cv_df$auc), 3))) +
    scale_color_discrete(name = "Model") +
    theme_bw()
  
  return(plt)
}

# Get predictions by proximity
get_ref_prop = function(train, input_dat, n_ref){
  sample_size = min(table(train$pep))
  rf_model = randomForest(x = (train %>% select(-pep)),
                          y = (train %>% pull(pep)),
                          xtest = (input_dat %>% select(-pep)),
                          ytest = (input_dat %>% pull(pep)), 
                          proximity = T, type = "classification",
                          strata = train$pep, sampsize = c(sample_size, sample_size))
  prox_matrix = t(rf_model$test$prox)
  
  # Separate by treatment class
  no_trt_ind = train %>%
    mutate(row_n = 1:n()) %>%
    filter(indo == 0) %>%
    pull(row_n)
  
  indo_ind = train %>%
    mutate(row_n = 1:n()) %>%
    filter(indo == 1) %>%
    pull(row_n)
  
  prop_pep = rep(1, nrow(input_dat))
  # Select n_ref closest samples as reference samples
  n_ref = n_ref
  for(i in 1:nrow(input_dat)){
    if(input_dat$indo[i] == 0){
      ref_ind = order(prox_matrix[no_trt_ind, i], decreasing = T)[1:n_ref]
    } else {
      ref_ind = order(prox_matrix[indo_ind, i], decreasing = T)[1:n_ref]
    }
    
    prop_pep[i] = train %>%
      slice(ref_ind) %>%
      summarize(prop_pep = mean(as.numeric(as.character(pep)))) %>%
      pull(prop_pep)
  }
  return(prop_pep)
}
