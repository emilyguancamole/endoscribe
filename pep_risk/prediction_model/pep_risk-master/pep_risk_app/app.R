# app.R
# Author:             Albert Kuo
# Date last modified: Jan 18, 2021
# Shiny app for PEP risk prediction using random forest

library(pacman)
p_load(shiny, shinythemes, shinyWidgets, shinycssloaders, tidyverse, ggrepel,
       janitor, caret, FNN, here, lime)
source(here("./code/my_plot_lime_features.R"))

# Define UI for application
ui <- navbarPage("", id = "navbar",

                 # Choose a theme (optional)
                 # theme = shinytheme("simplex"),

                 # Application title
                 tabPanel(title = "Estimator", value = "Estimator",

                          # ----

                          fluidRow(
                            column(3, offset = 1,
                                   wellPanel(
                              h4("Risk factors"),
                              div(style="display:inline-block",
                                  awesomeRadio(inputId = "gender_male_1",
                                           label = "Sex",
                                           choices = c("Male", "Female"),
                                           inline = TRUE)),
                              br(),

                              div(style="display:inline-block",
                              numericInput(inputId = "age_years",
                                           label = "Age",
                                           value = 0)),
                              br(),

                              div(style="display:inline-block",
                                  numericInput(inputId = "bmi",
                                               label = "BMI",
                                               value = 0)),

                              br(),

                              div(style="display:inline-block",
                                  switchInput(inputId = "sod",
                                              onLabel = "Yes",
                                              offLabel = "No")),
                              strong("Sphincter of oddi dysfunction"),

                              br(),

                              div(style="display:inline-block",
                                  switchInput(inputId = "history_of_pep",
                                              onLabel = "Yes",
                                              offLabel = "No")),
                              strong("History of PEP"),

                              br(),

                              div(style="display:inline-block",
                                  switchInput(inputId = "hx_of_recurrent_pancreatitis",
                                              onLabel = "Yes",
                                              offLabel = "No")),
                              strong("History of recurrent pancreatitis"),

                              br(),

                              div(style="display:inline-block",
                                  switchInput(inputId = "pancreatic_sphincterotomy",
                                              onLabel = "Yes",
                                              offLabel = "No")),
                              strong("Pancreatic sphincterotomy"),

                              br(),

                              div(style="display:inline-block",
                                  switchInput(inputId = "precut_sphincterotomy",
                                              onLabel = "Yes",
                                              offLabel = "No")),
                              strong("Precut sphincterotomy"),

                              br(),

                              div(style="display:inline-block",
                                  switchInput(inputId = "minor_papilla_sphincterotomy",
                                              onLabel = "Yes",
                                              offLabel = "No")),
                              strong("Minor papilla sphincterotomy"),

                              br(),

                              div(style="display:inline-block",
                                  switchInput(inputId = "failed_cannulation",
                                              onLabel = "Yes",
                                              offLabel = "No")),
                              strong("Failed cannulation"),

                              br(),

                              div(style="display:inline-block",
                                  switchInput(inputId = "difficult_cannulation",
                                              onLabel = "Yes",
                                              offLabel = "No")),
                              strong("Difficult cannulation"),

                              br(),

                              div(style="display:inline-block",
                                  switchInput(inputId = "pneumatic_dilation_of_intact_biliary_sphincter",
                                              onLabel = "Yes",
                                              offLabel = "No")),
                              strong("Pneumatic dilation of intact biliary sphincter"),

                              br(),

                              div(style="display:inline-block",
                                  switchInput(inputId = "pancreatic_duct_injection",
                                              onLabel = "Yes",
                                              offLabel = "No")),
                              strong("Pancreatic duct injection"),

                              br(),

                              div(style="display:inline-block",
                                  switchInput(inputId = "pancreatic_duct_injections_2",
                                              onLabel = "Yes",
                                              offLabel = "No")),
                              strong("Pancreatic duct injections > 2"),

                              br(),

                              div(style="display:inline-block",
                                  switchInput(inputId = "acinarization",
                                              onLabel = "Yes",
                                              offLabel = "No")),
                              strong("Acinarization"),

                              br(),

                              div(style="display:inline-block",
                                  switchInput(inputId = "trainee_involvement",
                                              onLabel = "Yes",
                                              offLabel = "No")),
                              strong("Trainee involvement"),

                              br(),

                              div(style="display:inline-block",
                                  switchInput(inputId = "cholecystectomy",
                                              onLabel = "Yes",
                                              offLabel = "No")),
                              strong("Cholecystectomy"),

                              br(),

                              div(style="display:inline-block",
                                  switchInput(inputId = "pancreo_biliary_malignancy",
                                              onLabel = "Yes",
                                              offLabel = "No")),
                              strong("Pancreo biliary malignancy"),

                              br(),

                              div(style="display:inline-block",
                                  switchInput(inputId = "guidewire_cannulation",
                                              onLabel = "Yes",
                                              offLabel = "No")),
                              strong("Guidewire cannulation"),

                              br(),

                              div(style="display:inline-block",
                                  switchInput(inputId = "guidewire_passage_into_pancreatic_duct",
                                              onLabel = "Yes",
                                              offLabel = "No")),
                              strong("Guidewire passage into pancreatic duct"),

                              br(),

                              div(style="display:inline-block",
                                  switchInput(inputId = "guidewire_passage_into_pancreatic_duct_2",
                                              onLabel = "Yes",
                                              offLabel = "No")),
                              strong("Guidewire passage into pancreatic duct > 2"),

                              br(),

                              div(style="display:inline-block",
                                  switchInput(inputId = "biliary_sphincterotomy",
                                              onLabel = "Yes",
                                              offLabel = "No")),
                              strong("Biliary sphincterotomy"),

                              br(), br(),
                              actionButton("update", "Submit", icon = icon("caret-right"),
                                           style = "color: #fff; background-color: #337ab7")
                                   )
                            ),

                            column(7,
                              h3("Post ERCP Pancreatitis Risk Calculator and Decision Making Tool"),

                              br(),

                              # One-line prediction summary
                              tags$head(tags$style("#pep_pred{font-size: 12pt; display: inline}")),
                              uiOutput("pep_pred"),

                              br(),
                              br(),

                              p(span("Quick Guide:", style = "font-weight: bold"), "The predicted risk of developing PEP for the patient is labelled in",
                              span("yellow", style="background-color: #FFFF00"), "and plotted below. The patient's
                              20 nearest neighbors from our dataset are also plotted with their predicted risk and actual PEP outcome, where",
                              span("red", style = "color: #D55E00"), "indicates they developed PEP and",
                              span("blue", style = "color: #56B4E9"), "indicates they did not develop PEP.
                              The contributions of each factor in increasing or decreasing the patient's risk are estimated with ",
                              a("LIME", href = "https://lime.data-imaginist.com/index.html"), "weights below.\n",
                              "To aid in decision making, we also provide the predicted risk under different treatment options. For more details,
                              please read the", actionLink("link_to_about", "About"), "page."),

                              br(),
                              br()
                            ),
                            column(3,
                                   # Votes plot

                                   plotOutput("votesPlot") %>%
                                     withSpinner(type = 5)
                            ),
                            column(4,
                                   # LIME plot
                                   plotOutput("varImpPlot") %>%
                                     withSpinner(type = 5),

                                   br(),
                                   br(),
                            ),

                            column(7,
                                   # Votes plot
                                   plotOutput("votesTrtPlot") %>%
                                     withSpinner(type = 5),

                                   br(),

                                   # Description
                                   # tags$head(tags$style("#vote_percent{font-size: 20px; display: inline}")),
                                   # uiOutput("vote_percent"),
                                   # tags$head(tags$style("#indo_pep_percent{font-size: 20px; display: inline}")),
                                   # uiOutput("indo_pep_percent"),
                                   # tags$head(tags$style("#indo_vote_percent{font-size: 20px; display: inline}")),
                                   # uiOutput("indo_vote_percent"),

                                   br(),
                                   br(),

                                   # Trait plot
                                   # plotOutput("traitPlot") %>%
                                   #   withSpinner(type = 5)
                            )
                          )


                          #### ----
                 ),

                 tabPanel(title = "About", value = "About",
                          fluidRow(
                            column(12,
                                   h4("The App"),
                                   p("The Post ERCP Pancreatitis Risk Calculator and Decision Making Tool was designed to estimate a patient's risk of developing
                                   post ERCP pancreatitis and the effects of different treatments in reducing the risk. Its aim is to complement, rather than replace,
                                   the decision-making of physicians. The estimator was developed using data from 7,389 patients from 12 studies. The
                                   method achieved an AUC of 0.70 under 5-fold cross-validation."),
                                   h4("Citation"),
                                   p("Please cite our manuscript:"),
                                   h4("The Team"),
                                   p("Institutions contributing data include: ..."),
                                   p("(Look at ",
                                   a("https://dobbs-onc-jhmi.shinyapps.io/odxRescue/",
                                     href="https://dobbs-onc-jhmi.shinyapps.io/odxRescue/"),
                                   "for guidance on how to write an official-sounding about description.)")
                            )
                          )
                 )
)

# Define server logic required for plots
server <- function(input, output, session) {
  # Read in data output from pred_model.Rmd
  #fit = readRDS("data/gbm_model.rds")     # Model on full dataset
  fit_sub = readRDS("data/gbm_model_trt.rds") # Model on trt subsets
  train = readRDS("data/train_new.rds")   # Training dataset (unnormalized)
  train_impute = readRDS("data/train_impute.rds") # Imputed and normalized training dataset
  var_names = readRDS("data/var_names.rds") # Variable labels
  lime_explainer = readRDS("data/lime_explainer.rds") # LIME explainer for variable importance
  # pred_dt = readRDS("data/pred_dt_gbm.rds") # Prediction scores for training set
  n_k = 20 # Number of nearest neighbors

  # Wait for submit button before refreshing plots
  therapy_level_order = c("No treatment", "Aggressive hydration only", "Indomethacin only",
                          "Aggressive hydration and indomethacin", "PD stent only", "Indomethacin and PD stent")
  ref_samples_default = tibble(patient_id = 1:120,
                               pep = as.factor(0),
                               pred = 0,
                               therapy = rep(therapy_level_order, 20))
  test_patient_default = tibble(patient_id = 1,
                                pred = 0,
                                therapy = therapy_level_order)
  explanation_default = readRDS("data/explanation_default.rds")
  rv = reactiveValues(ref_samples = ref_samples_default,
                      test_patient_pred = test_patient_default,
                      explanation = explanation_default)

  observeEvent(input$update, {
    # Capture input and put in data frame
    input_dat = data.frame(age_years = input$age_years,
                           gender_male_1 = ifelse(input$gender_male_1 == "Male", 1, 0),
                           bmi = input$bmi,
                           sod = as.integer(input$sod),
                           history_of_pep = as.integer(input$history_of_pep),
                           hx_of_recurrent_pancreatitis = as.integer(input$history_of_pep),
                           pancreatic_sphincterotomy = as.integer(input$history_of_pep),
                           precut_sphincterotomy = as.integer(input$precut_sphincterotomy),
                           minor_papilla_sphincterotomy = as.integer(input$minor_papilla_sphincterotomy),
                           failed_cannulation = as.integer(input$failed_cannulation),
                           difficult_cannulation = as.integer(input$difficult_cannulation),
                           pneumatic_dilation_of_intact_biliary_sphincter = as.integer(input$pneumatic_dilation_of_intact_biliary_sphincter),
                           pancreatic_duct_injection = as.integer(input$pancreatic_duct_injection),
                           pancreatic_duct_injections_2 = as.integer(input$pancreatic_duct_injections_2),
                           acinarization = as.integer(input$acinarization),
                           trainee_involvement = as.integer(input$trainee_involvement),
                           cholecystectomy = as.integer(input$cholecystectomy),
                           pancreo_biliary_malignancy = as.integer(input$pancreo_biliary_malignancy),
                           guidewire_cannulation = as.integer(input$guidewire_cannulation),
                           guidewire_passage_into_pancreatic_duct = as.integer(input$guidewire_passage_into_pancreatic_duct),
                           guidewire_passage_into_pancreatic_duct_2 = as.integer(input$guidewire_passage_into_pancreatic_duct_2),
                           biliary_sphincterotomy = as.integer(input$biliary_sphincterotomy),
                           aggressive_hydration = c(0, 1, 0, 0, 1, 0),
                           indomethacin_nsaid_prophylaxis = c(0, 0, 1, 0, 1, 1),
                           pancreatic_duct_stent_placement = c(0, 0, 0, 1, 0, 1),
                           therapy = c("No treatment", "Aggressive hydration only", "Indomethacin only",
                                       "PD stent only", "Aggressive hydration and indomethacin", "Indomethacin and PD stent"),
                           patient_id = 1)


    # Normalize values
    pre_proc_values <- preProcess(train %>% select(-c("study_id", "pep", "patient_id")), method = c("center", "scale"))
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
    rv$test_patient_pred = test_patient_pred

    # Nearest neighbors (ref_samples)
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
    rv$ref_samples = bind_rows(neighbors_1, neighbors_2, neighbors_3,
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
      right_join(., var_names, by = c("feature" = "variable")) %>%
      mutate(feature_desc = var_label)
    rv$explanation = explanation
  })

  # Plot of votes for no treatment
  output$votesPlot = renderPlot({
    set.seed(1) # for geom_jitter to look the same every time
    ref_samples = rv$ref_samples %>% filter(therapy == "No treatment")
    test_patient_pred = rv$test_patient_pred %>% filter(therapy == "No treatment")

    input$update
    isolate(
      ggplot() +
        geom_jitter(data = ref_samples, width = 0.1, height = 0,
                    aes(x = factor(therapy, levels = therapy_level_order),
                        y = pred, size = 3, color = pep, alpha = 0.8)) +
        geom_point(data = test_patient_pred,
                   aes(x = factor(therapy, levels = therapy_level_order),
                       y = pred),
                   size = 4, shape = 21, color = "black", fill = "yellow") +
        geom_label_repel(data = test_patient_pred,
                         aes(label = paste("patient risk:", paste0(round(pred*100, digits = 1), "%")),
                             x = therapy, y = pred),
                         direction = "y", nudge_x = 0.1, fill = "yellow",
                         color = "#333333", alpha = 0.8,
                         segment.color = "#333333", segment.alpha = 0.8) +
        scale_x_discrete(labels = str_wrap("No prophylaxis", width = 20)) +
        scale_y_continuous(limits = c(0, min(1, max(0.2, max(rv$ref_samples$pred), max(rv$test_patient_pred$pred)))),
                           labels = scales::percent_format(accuracy = 1)) +
        scale_color_manual(labels = c("No", "Yes"), values = c("#56B4E9", "#D55E00")) +
        guides(size = F,
               alpha = F,
               color = guide_legend(override.aes = list(size = 4, alpha = 0.8))) +
        labs(title = "Predicted risk of PEP",
             x = "Treatment",
             y = "Probability of developing PEP",
             color = "Developed PEP") +
        theme_classic(base_size = 15) +
        theme(plot.title = element_text(hjust = 0.5),
              #axis.text.x = element_blank(),
              axis.title.x = element_blank(),
              #axis.ticks.x = element_blank(),
              legend.text = element_text(size = 15),
              legend.position = "top")
    )
  })

  # Plot LIME
  output$varImpPlot = renderPlot({
    explanation = rv$explanation

    input$update
    isolate(
      my_plot_lime_features(explanation)
    )
  })

  # Plot of votes for other treatment options
  output$votesTrtPlot = renderPlot({
    set.seed(1) # for geom_jitter to look the same every time
    ref_samples = rv$ref_samples %>% filter(therapy != "No treatment")
    test_patient_pred = rv$test_patient_pred %>% filter(therapy != "No treatment")

    input$update
    isolate(
      ggplot() +
        geom_jitter(data = ref_samples, width = 0.2, height = 0,
                    aes(x = factor(therapy, levels = therapy_level_order),
                        y = pred, size = 3, color = pep, alpha = 0.8)) +
        geom_point(data = test_patient_pred,
                   aes(x = factor(therapy, levels = therapy_level_order), y = pred),
                   size = 4, shape = 21, color = "black", fill = "yellow") +
        geom_label_repel(data = test_patient_pred,
                         aes(label = paste("patient risk:", paste0(round(pred*100, digits = 1), "%")),
                             x = therapy, y = pred),
                         direction = "y", nudge_x = 0.4, fill = "yellow",
                         color = "#333333", alpha = 0.8,
                         segment.color = "#333333", segment.alpha = 0.8) +
        scale_x_discrete(labels = c("Aggressive hydration only", "Rectal NSAIDs only",
                                    "Aggressive hydration and Rectal NSAIDs", "PD stent only",
                                    "Rectal NSAIDs and PD stent")) + #function(x) str_wrap(x, width = 20)) +
        scale_y_continuous(limits = c(0, min(1, max(0.2, max(rv$ref_samples$pred), max(rv$test_patient_pred$pred)))),
                           labels = scales::percent_format(accuracy = 1)) +
        scale_color_manual(labels = c("No", "Yes"), values = c("#56B4E9", "#D55E00")) +
        guides(size = F,
               alpha = F,
               color = guide_legend(override.aes = list(size = 4, alpha = 0.8))) +
        labs(title = "Predicted risk of PEP with treatment",
             x = "Treatment",
             y = "Probability of developing PEP",
             color = "Developed PEP") +
        theme_classic(base_size = 15) +
        theme(plot.title = element_text(hjust = 0.5),
              axis.text.x = element_text(size = 13),
              legend.text = element_text(size = 15),
              legend.position = "top")
    )
  })

  observeEvent(input$link_to_about, {
    updateTabsetPanel(session, "navbar", "About")
  })

  # Plot of shared traits
  # output$traitPlot = renderPlot({
  #   input$update
  #   isolate(
  #     rv$ref_samples %>%
  #       filter(trt == "No treatment") %>%
  #       summarize(age = mean(age >= input$age - 10 & age <= input$age + 10),
  #                 sex = mean(sex == ifelse(input$sex=="Female", 1, 0)),
  #                 gw_cann = mean(gw_cann == as.integer(input$gw_cann)),
  #                 x2gw_pass_pd = mean(x2gw_pass_pd == as.integer(input$x2gw_pass_pd)),
  #                 history_of_pep = mean(history_of_pep == as.integer(input$history_of_pep)),
  #                 panc_sphinc = mean(panc_sphinc == as.integer(input$panc_sphinc)),
  #                 diff_cann = mean(diff_cann == as.integer(input$diff_cann)),
  #                 sod = mean(sod == as.integer(input$sod)),
  #                 pb_mal = mean(pb_mal == as.integer(input$pb_mal)),
  #                 panc_div = mean(panc_div == as.integer(input$panc_div)),
  #                 trainee = mean(trainee == as.integer(input$trainee)),
  #                 min_pap_sphinc = mean(min_pap_sphinc == as.integer(input$min_pap_sphinc))) %>%
  #       gather(variable, prop) %>%
  #       mutate(variable = factor(variable, levels = c("age", "sex", "gw_cann",
  #                                                     "x2gw_pass_pd", "history_of_pep", "panc_sphinc",
  #                                                     "diff_cann", "sod", "pb_mal",
  #                                                     "panc_div", "trainee", "min_pap_sphinc"))) %>%
  #       ggplot(aes(x = variable, y = prop)) +
  #       geom_col() +
  #       scale_y_continuous(limits = c(0, 1)) +
  #       scale_x_discrete(labels = c("Age ± 10", "Sex", "Guidewire cannulation",
  #                                   ">2 Guidewire passes", "History of PEP",
  #                                   "Pancreatic sphincterotomy", "Difficult cannulation",
  #                                   "SOD", "Pancreo-biliary malignancy",
  #                                   "Pancreas divisum", "Trainee involvement",
  #                                   "Minor papillary sphincterotomy")) +
  #       labs(title = "Proportion of reference cases (no treatment) sharing the same characteristic",
  #            x = "Patient risk factors",
  #            y = "Proportion") +
  #       theme_classic(base_size = 15) +
  #       theme(axis.text.x = element_text(angle = 40, hjust = 1, size = 15),
  #             plot.title = element_text(hjust = 0.5))
  #   )
  # }, width = 700)

  output$pep_pred = renderText({
    pep_pred_percent = rv$test_patient_pred %>% filter(therapy == "No treatment") %>% pull(pred)
    pep_pred_percent = round(pep_pred_percent*100, digits = 1)
    HTML(paste0("Based on the input risk factors for the patient, <b> the risk of developing PEP
                without prophylaxis is ", pep_pred_percent, "%.</b>"))
  })

  # # % developing PEP in reference cases
  # output$pep_percent = renderText({
  #   pep_percent = rv$ref_samples %>%
  #     filter(trt == "No treatment") %>%
  #     mutate(pep = as.numeric(as.character(pep))) %>%
  #     summarize(mean_pep = mean(pep)*100) %>%
  #     pull(mean_pep) %>%
  #     round(digits = 2)
  #
  #   n = rv$ref_samples %>%
  #     filter(trt == "No treatment") %>%
  #     summarize(n = n())
  #
  #   HTML(paste0("Based on the characteristics of ",
  #          "<b>", n, "</b>", " similar cases from the reference database, ",
  #          "<b>", pep_percent, "</b>", "% of similar cases developed PEP."))
  # })
  #
  # # vote percent for no treatment
  # output$vote_percent = renderText({
  #   if(rv$vote_no_trt < 0.5)
  #     risk_text = "low risk."
  #   else if(rv$vote_no_trt < 0.75)
  #     risk_text = "moderate risk."
  #   else
  #     risk_text = "high risk."
  #
  #   HTML(paste0("The risk metric (vote percentage) is ", "<b>", rv$vote_no_trt, "</b>",
  #          ", which is ", "<b>", risk_text, "</b>"))
  # })
  #
  # # % developing PEP in reference cases with indo
  # output$indo_pep_percent = renderText({
  #   pep_percent = rv$ref_samples %>%
  #     filter(trt == "Indomethacin") %>%
  #     mutate(pep = as.numeric(as.character(pep))) %>%
  #     summarize(mean_pep = mean(pep)*100) %>%
  #     pull(mean_pep) %>%
  #     round(digits = 2)
  #
  #   n = rv$ref_samples %>%
  #     filter(trt == "Indomethacin") %>%
  #     summarize(n = n())
  #
  #   HTML(paste0("If the patient is treated with indomethacin, then ",
  #          "<b>", pep_percent, "</b>", "% of ",
  #          "<b>", n, "</b>", " similar cases treated with indomethacin developed PEP."))
  # })
  #
  # # vote percent for indo
  # output$indo_vote_percent = renderText({
  #   if(rv$vote_indo < 0.5)
  #     risk_text = "low risk."
  #   else if(rv$vote_indo < 0.8)
  #     risk_text = "moderate risk."
  #   else
  #     risk_text = "high risk."
  #
  #   HTML(paste0("The risk metric (vote percentage) with indomethacin treatment will be ",
  #          "<b>", rv$vote_indo, "</b>", ", which is ", "<b>", risk_text, "</b>"))
  # })

  # Info pop-ups
  observeEvent(input$show_trainee, {
    showModal(modalDialog(
      title = "Trainee involvement",
      "Whether a trainee was involved in the ERCP procedure",
      easyClose = TRUE
    ))
  })

  observeEvent(input$show_x2gw_pass_pd, {
    showModal(modalDialog(
      title = ">2 Guidewire passes",
      "More than two guidewire passes into pancreatic duct",
      easyClose = TRUE
    ))
  })
}

# Run the application
shinyApp(ui = ui, server = server)

