# app.R
# -----------------------------------------------------------------------------
# Author:            Priya Madduluri
# Date last modified: Nov 23rd, 2025
#
# Backend logic for risk calculation

#library(caret)
#p_load(shiny, shinythemes, shinyWidgets, shinycssloaders, tidyverse, ggrepel,
#      janitor, caret, FNN, here, lime)
#source(here("./code/my_plot_lime_features.R"))


# Define server logic required for plots
peprisk_predict <- function(input) {
  # Read in data output from pred_model.Rmd
      # Model on full dataset
  

  fit = readRDS("pep_risk_app/data/gbm_model.rds") 
  fit_sub = readRDS("pep_risk_app/data/gbm_model_trt.rds") # Model on trt subsets
  train = readRDS("pep_risk_app/data/train_new.rds")   # Training dataset (unnormalized)
  train_impute = readRDS("pep_risk_app/data/train_impute.rds") # Imputed and normalized training dataset
  var_names = readRDS("pep_risk_app/data/var_names.rds") # Variable labels
  lime_explainer = readRDS("pep_risk_app/data/lime_explainer.rds") # LIME explainer for variable importance
  # pred_dt = readRDS("data/pred_dt_gbm.rds") # Prediction scores for training set
  n_k = 20 # Number of nearest neighbors

  
    # Capture input and put in data frame
    input_dat <- data.frame(age_years = input$age_years[1],
                           gender_male_1 = input$gender_male_1[1],
                           bmi = input$bmi[1],
                           sod = as.integer(input$sod[1]),
                           history_of_pep = as.integer(input$history_of_pep[1]),
                           hx_of_recurrent_pancreatitis = as.integer(input$history_of_pep[1]),
                           pancreatic_sphincterotomy = as.integer(input$history_of_pep[1]),
                           precut_sphincterotomy = as.integer(input$precut_sphincterotomy[1]),
                           minor_papilla_sphincterotomy = as.integer(input$minor_papilla_sphincterotomy[1]),
                           failed_cannulation = as.integer(input$failed_cannulation[1]),
                           difficult_cannulation = as.integer(input$difficult_cannulation[1]),
                           pneumatic_dilation_of_intact_biliary_sphincter = as.integer(input$pneumatic_dilation_of_intact_biliary_sphincter[1]),
                           pancreatic_duct_injection = as.integer(input$pancreatic_duct_injection[1]),
                           pancreatic_duct_injections_2 = as.integer(input$pancreatic_duct_injections_2[1]),
                           acinarization = as.integer(input$acinarization[1]),
                           trainee_involvement = as.integer(input$trainee_involvement[1]),
                           cholecystectomy = as.integer(input$cholecystectomy[1]),
                           pancreo_biliary_malignancy = as.integer(input$pancreo_biliary_malignancy[1]),
                           guidewire_cannulation = as.integer(input$guidewire_cannulation[1]),
                           guidewire_passage_into_pancreatic_duct = as.integer(input$guidewire_passage_into_pancreatic_duct[1]),
                           guidewire_passage_into_pancreatic_duct_2 = as.integer(input$guidewire_passage_into_pancreatic_duct_2[1]),
                           biliary_sphincterotomy = as.integer(input$biliary_sphincterotomy[1]),
                           aggressive_hydration = c(0, 1, 0, 0, 1, 0),
                           indomethacin_nsaid_prophylaxis = c(0, 0, 1, 0, 1, 1),
                           pancreatic_duct_stent_placement = c(0, 0, 0, 1, 0, 1),
                           therapy = c("No treatment", "Aggressive hydration only", "Indomethacin only",
                                       "PD stent only", "Aggressive hydration and indomethacin", "Indomethacin and PD stent"),
                           patient_id = 1)
    
    
    # Normalize values
    message("Normalize values")
    pre_proc_values <- caret::preProcess(train %>% dplyr::select(-c("study_id", "pep", "patient_id")), method = c("center", "scale"))
    test_impute = predict(pre_proc_values, input_dat)
    
    # Prediction
    # Prediction for each treatment with models on trt subsets
    test_patients_pred_ls = list()
    for(trt in c("Aggressive hydration only", "Indomethacin only", "PD stent only", "Aggressive hydration and indomethacin", "Indomethacin and PD stent")){
      # Predict on no trt
      p1 = predict(fit_sub[[trt]], newdata = test_impute %>% filter(therapy == "No treatment"), type = "prob")[, 2]
      
      # Predict on trt
      p2 = predict(fit_sub[[trt]], newdata = test_impute %>% filter(therapy == trt), type = "prob")[, 2]
      
      # Predict on full model
      test_sub = test_impute %>% filter(therapy == "No treatment")
      pred = predict(fit, newdata = test_sub, type = "prob")[, 2]
      
      # Adjust prediction for aggressive hydration first
      if(trt == "Aggressive hydration and indomethacin"){
        p3 = predict(fit_sub[["Aggressive hydration only"]], newdata = test_impute %>% filter(therapy == "No treatment"), type = "prob")[, 2]
        p4 = predict(fit_sub[["Aggressive hydration only"]], newdata = test_impute %>% filter(therapy == "Aggressive hydration only"), type = "prob")[, 2]
        shrinkage = ifelse(p3 > 0.1, 1, p3*10)
        adj_factor = p4/p3*shrinkage + 1*(1-shrinkage)
        adj_factor[is.nan(adj_factor)] = 1
        pred = pred * adj_factor
      }
      
      # Compute adjusted prediction
      shrinkage = ifelse(p1 > 0.1, 1, p1*10)
      adj_factor = p2/p1*shrinkage + 1*(1 - shrinkage)  # take the ratio, with shrinkage towards 1 if p1 < 0.1
      adj_factor[is.nan(adj_factor)] = 1 # set adjustment factor to 1 if p1 = 0
      pred_adj = pred * adj_factor
      test_patients_pred_ls[[trt]] = tibble(patient_id = test_sub$patient_id,
                                            therapy = trt,
                                            pred = pred_adj)
    }
    
    
    # Prediction for no treatment on full model
    test_sub = test_impute %>% filter(therapy == "No treatment")
    test_no_trt = tibble(patient_id = test_sub$patient_id,
                         therapy = "No treatment",
                         pred = predict(fit, newdata = test_sub, type = "prob")[, 2])
    test_patient_pred = bind_rows(bind_rows(test_patients_pred_ls), test_no_trt)
    rv_test_patient_pred = test_patient_pred
    
    

    # Nearest neighbors (ref_samples)
    message("Nearest neighbors")
    patient_ids = train %>%
      filter(aggressive_hydration == 0 &
               indomethacin_nsaid_prophylaxis == 0 &
               pancreatic_duct_stent_placement == 0) %>%
      pull(patient_id)
    train_sub = train_impute %>%
      filter(patient_id %in% patient_ids)
    notrt_values = train_sub %>% select(aggressive_hydration, indomethacin_nsaid_prophylaxis, pancreatic_duct_stent_placement) %>% distinct()
    stopifnot(nrow(notrt_values) == 1)
    
    test_patient_impute = test_impute
    k <- knn(train = train_sub %>% select(-pep, -patient_id, -indomethacin_nsaid_prophylaxis, -aggressive_hydration, -pancreatic_duct_stent_placement),
             test = test_patient_impute %>% filter(therapy == "No treatment") %>%
               select(-patient_id, -therapy, -indomethacin_nsaid_prophylaxis, -aggressive_hydration, -pancreatic_duct_stent_placement),
             train_sub$pep, k = n_k, algorithm = "cover_tree")
    indices <- attr(k, "nn.index")[1, ]
    neighbors_1 = train_sub %>%
      slice(indices)
    ## Get prediction
    pred = predict(fit, newdata = neighbors_1, type = "prob")[, 2]
    neighbors_1 = neighbors_1 %>% mutate(pred = pred)
    
    
    # Nearest neighbors among aggressive hydration only patients (n = 325)
    trt = "Aggressive hydration only"
    patient_ids = train %>%
      filter(aggressive_hydration == 1 &
               indomethacin_nsaid_prophylaxis == 0 &
               pancreatic_duct_stent_placement == 0) %>%
      pull(patient_id)
    train_sub = train_impute %>%
      filter(patient_id %in% patient_ids)
    
    k <- knn(train = train_sub %>% select(-pep, -patient_id, -indomethacin_nsaid_prophylaxis, -aggressive_hydration, -pancreatic_duct_stent_placement),
             test = test_patient_impute %>% filter(therapy == trt) %>%
               select(-patient_id, -therapy, -indomethacin_nsaid_prophylaxis, -aggressive_hydration, -pancreatic_duct_stent_placement),
             train_sub$pep, k = n_k, algorithm = "cover_tree")
    indices <- attr(k, "nn.index")[1, ]
    neighbors_2 = train_sub %>%
      slice(indices)
    neighbors_2_notrt = neighbors_2 %>% mutate(indomethacin_nsaid_prophylaxis = notrt_values$indomethacin_nsaid_prophylaxis, aggressive_hydration = notrt_values$aggressive_hydration, pancreatic_duct_stent_placement = notrt_values$pancreatic_duct_stent_placement)
    ## Get prediction
    pred = predict(fit, newdata = neighbors_2_notrt,
                   type = "prob")[, 2]
    p1 = predict(fit_sub[[trt]], newdata = neighbors_2_notrt, type = "prob")[, 2]
    p2 = predict(fit_sub[[trt]], newdata = neighbors_2, type = "prob")[, 2]
    shrinkage = ifelse(p1 > 0.1, 1, p1*10)
    adj_factor = p2/p1*shrinkage + 1*(1 - shrinkage)  # take the ratio, with shrinkage towards 1 if p1 < 0.1
    adj_factor[is.nan(adj_factor)] = 1 # set adjustment factor to 1 if p1 = 0
    pred_adj = pred * adj_factor
    neighbors_2 = neighbors_2 %>% mutate(pred = pred_adj)
    
    # Nearest neighbors among indomethacin only patients (n = 2955)
    trt = "Indomethacin only"
    patient_ids = train %>%
      filter(aggressive_hydration == 0 &
               indomethacin_nsaid_prophylaxis == 1 &
               pancreatic_duct_stent_placement == 0) %>%
      pull(patient_id)
    train_sub = train_impute %>%
      filter(patient_id %in% patient_ids)
    
    k <- knn(train = train_sub %>% select(-pep, -patient_id, -indomethacin_nsaid_prophylaxis, -aggressive_hydration, -pancreatic_duct_stent_placement),
             test = test_patient_impute %>% filter(therapy == trt) %>%
               select(-patient_id, -therapy, -indomethacin_nsaid_prophylaxis, -aggressive_hydration, -pancreatic_duct_stent_placement),
             train_sub$pep, k = n_k, algorithm = "cover_tree")
    indices <- attr(k, "nn.index")[1, ]
    neighbors_3 = train_sub %>%
      slice(indices)
    neighbors_3_notrt = neighbors_3 %>% mutate(indomethacin_nsaid_prophylaxis = notrt_values$indomethacin_nsaid_prophylaxis, aggressive_hydration = notrt_values$aggressive_hydration, pancreatic_duct_stent_placement = notrt_values$pancreatic_duct_stent_placement)
    ## Get prediction
    pred = predict(fit, newdata = neighbors_3_notrt,
                   type = "prob")[, 2]
    p1 = predict(fit_sub[[trt]], newdata = neighbors_3_notrt, type = "prob")[, 2]
    p2 = predict(fit_sub[[trt]], newdata = neighbors_3, type = "prob")[, 2]
    shrinkage = ifelse(p1 > 0.1, 1, p1*10)
    adj_factor = p2/p1*shrinkage + 1*(1 - shrinkage)  # take the ratio, with shrinkage towards 1 if p1 < 0.1
    adj_factor[is.nan(adj_factor)] = 1 # set adjustment factor to 1 if p1 = 0
    pred_adj = pred * adj_factor
    neighbors_3 = neighbors_3 %>% mutate(pred = pred_adj)
    
    # Nearest neighbors among PD stent patients (n = 363)
    trt = "PD stent only"
    patient_ids = train %>%
      filter(aggressive_hydration == 0 &
               indomethacin_nsaid_prophylaxis == 0 &
               pancreatic_duct_stent_placement == 1) %>%
      pull(patient_id)
    train_sub = train_impute %>%
      filter(patient_id %in% patient_ids)
    
    k <- knn(train = train_sub %>% select(-pep, -patient_id, -indomethacin_nsaid_prophylaxis, -aggressive_hydration, -pancreatic_duct_stent_placement),
             test = test_patient_impute %>% filter(therapy == trt) %>%
               select(-patient_id, -therapy, -indomethacin_nsaid_prophylaxis, -aggressive_hydration, -pancreatic_duct_stent_placement),
             train_sub$pep, k = n_k, algorithm = "cover_tree")
    indices <- attr(k, "nn.index")[1, ]
    neighbors_4 = train_sub %>%
      slice(indices)
    neighbors_4_notrt = neighbors_4 %>% mutate(indomethacin_nsaid_prophylaxis = notrt_values$indomethacin_nsaid_prophylaxis, aggressive_hydration = notrt_values$aggressive_hydration, pancreatic_duct_stent_placement = notrt_values$pancreatic_duct_stent_placement)
    ## Get prediction
    pred = predict(fit, newdata = neighbors_4_notrt,
                   type = "prob")[, 2]
    p1 = predict(fit_sub[[trt]], newdata = neighbors_4_notrt, type = "prob")[, 2]
    p2 = predict(fit_sub[[trt]], newdata = neighbors_4, type = "prob")[, 2]
    shrinkage = ifelse(p1 > 0.1, 1, p1*10)
    adj_factor = p2/p1*shrinkage + 1*(1 - shrinkage)  # take the ratio, with shrinkage towards 1 if p1 < 0.1
    adj_factor[is.nan(adj_factor)] = 1 # set adjustment factor to 1 if p1 = 0
    pred_adj = pred * adj_factor
    neighbors_4 = neighbors_4 %>% mutate(pred = pred_adj)
    
    # Nearest neighbors among aggressive hydration and indomethacin (n = 79)
    trt = "Aggressive hydration and indomethacin"
    patient_ids = train %>%
      filter(aggressive_hydration == 1 &
               indomethacin_nsaid_prophylaxis == 1 &
               pancreatic_duct_stent_placement == 0) %>%
      pull(patient_id)
    train_sub = train_impute %>%
      filter(patient_id %in% patient_ids)
    
    k <- knn(train = train_sub %>% select(-pep, -patient_id, -indomethacin_nsaid_prophylaxis, -aggressive_hydration, -pancreatic_duct_stent_placement),
             test = test_patient_impute %>%
               filter(therapy == trt) %>%
               select(-patient_id, -therapy, -indomethacin_nsaid_prophylaxis, -aggressive_hydration, -pancreatic_duct_stent_placement),
             train_sub$pep, k = n_k, algorithm = "cover_tree")
    indices <- attr(k, "nn.index")[1, ]
    neighbors_5 = train_sub %>%
      slice(indices)
    neighbors_5_notrt = neighbors_5 %>% mutate(indomethacin_nsaid_prophylaxis = notrt_values$indomethacin_nsaid_prophylaxis, aggressive_hydration = notrt_values$aggressive_hydration, pancreatic_duct_stent_placement = notrt_values$pancreatic_duct_stent_placement)
    
    ## Get prediction
    pred = predict(fit, newdata = neighbors_5_notrt,
                   type = "prob")[, 2]
    ah_patient = train %>% filter(aggressive_hydration == 1) %>% slice(1) %>% pull(patient_id) # pick a patient that had aggressive_hydration
    ah_value = train_impute %>% filter(patient_id %in% ah_patient) %>% pull(aggressive_hydration)
    p3 = predict(fit_sub[["Aggressive hydration only"]], newdata = neighbors_5_notrt, type = "prob")[, 2]
    p4 = predict(fit_sub[["Aggressive hydration only"]], newdata = neighbors_5_notrt %>% mutate(aggressive_hydration = ah_value), type = "prob")[, 2]
    shrinkage = ifelse(p3 > 0.1, 1, p3*10)
    adj_factor = p4/p3*shrinkage + 1*(1-shrinkage)
    adj_factor[is.nan(adj_factor)] = 1
    pred = pred * adj_factor
    p1 = predict(fit_sub[[trt]], newdata = neighbors_5_notrt, type = "prob")[, 2]
    p2 = predict(fit_sub[[trt]], newdata = neighbors_5, type = "prob")[, 2]
    shrinkage = ifelse(p1 > 0.1, 1, p1*10)
    adj_factor = p2/p1*shrinkage + 1*(1 - shrinkage)  # take the ratio, with shrinkage towards 1 if p1 < 0.1
    adj_factor[is.nan(adj_factor)] = 1 # set adjustment factor to 1 if p1 = 0
    pred_adj = pred * adj_factor
    neighbors_5 = neighbors_5 %>% mutate(pred = pred_adj)
    
 
    
    # Nearest neighbors among Indomethacin and PD stent patients
    trt = "PD stent only"
    patient_ids = train %>%
      filter(aggressive_hydration == 0 &
               indomethacin_nsaid_prophylaxis == 1 &
               pancreatic_duct_stent_placement == 1) %>%
      pull(patient_id)
    train_sub = train_impute %>%
      filter(patient_id %in% patient_ids)
    
    k <- knn(train = train_sub %>% select(-pep, -patient_id, -indomethacin_nsaid_prophylaxis, -aggressive_hydration, -pancreatic_duct_stent_placement),
             test = test_patient_impute %>% filter(therapy == trt) %>%
               select(-patient_id, -therapy, -indomethacin_nsaid_prophylaxis, -aggressive_hydration, -pancreatic_duct_stent_placement),
             train_sub$pep, k = n_k, algorithm = "cover_tree")
    indices <- attr(k, "nn.index")[1, ]
    neighbors_6 = train_sub %>% 
      slice(indices)
    neighbors_6_notrt = neighbors_6 %>% mutate(indomethacin_nsaid_prophylaxis = notrt_values$indomethacin_nsaid_prophylaxis, aggressive_hydration = notrt_values$aggressive_hydration, pancreatic_duct_stent_placement = notrt_values$pancreatic_duct_stent_placement)
    ## Get prediction
    pred = predict(fit, newdata = neighbors_6_notrt,
                   type = "prob")[, 2]
    p1 = predict(fit_sub[[trt]], newdata = neighbors_6_notrt, type = "prob")[, 2]
    p2 = predict(fit_sub[[trt]], newdata = neighbors_6, type = "prob")[, 2]
    shrinkage = ifelse(p1 > 0.1, 1, p1*10)
    adj_factor = p2/p1*shrinkage + 1*(1 - shrinkage)  # take the ratio, with shrinkage towards 1 if p1 < 0.1
    adj_factor[is.nan(adj_factor)] = 1 # set adjustment factor to 1 if p1 = 0
    pred_adj = pred * adj_factor
    neighbors_6 = neighbors_6 %>% mutate(pred = pred_adj)
    

    
    # Store neighbors with test patient
    message("Store neighbors")
    rv_ref_samples = bind_rows(neighbors_1, neighbors_2, neighbors_3,
                               neighbors_4, neighbors_5, neighbors_6) %>%
      select(patient_id, pred, pep) %>%
      mutate(therapy = rep(c("No treatment", "Aggressive hydration only",
                             "Indomethacin only", "PD stent only", "Aggressive hydration and indomethacin",
                             "Indomethacin and PD stent"),
                           each = n_k))
    
    # Store LIME explanation
    explanation = explain(test_sub %>% select(-patient_id, -therapy),
                          lime_explainer, n_labels = 1,
                          n_features = ncol(test_sub) - 2)
    explanation = explanation %>%
      right_join(., var_names, by = c("feature" = "variable")) %>% mutate(feature_desc = var_label)
    rv_explanation = explanation
    
    # Return ALL treatment predictions (not just "No treatment")
    ref_samples = rv_ref_samples  # All reference samples for all therapies
    test_patient_pred = rv_test_patient_pred  # All treatment predictions
    
    # Overall Risk Prediction for the patient (No treatment baseline)
    pep_pred_percent = rv_test_patient_pred %>% filter(therapy == "No treatment") %>% pull(pred)
    pep_pred_percent = round(pep_pred_percent*100, digits = 1)
    
    output <- list(
      reference_samples = ref_samples, 
      test_patient_prediction = test_patient_pred,  # Now includes all 6 treatment scenarios
      explanation_text = explanation,
      final_risk = pep_pred_percent
    )
    
    return(output)
 
}



