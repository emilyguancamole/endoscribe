# pep_risk

This project builds a prediction model for PEP (post-ERCP pancreatitis). The Shiny app provides a real-time estimation of the risk of PEP for a given patient.

**Repo organization:**

* `code/pred_model.Rmd` - Modeling and analysis
  
  * Old code:
    
    * `code/build_data.Rmd` - Builds clean data from the three studies' raw data
    
    * `code/rf_model.Rmd` - Main document for the major points of the analysis, can run standalone
    
    * `code/rf_model_supplement.Rmd` - Supplementary details (e.g. sampling options, feature selection options, etc.)
    
    * `code/rf_functions.R` - Helper functions

* `code/my_plot_lime_features.R` - Helper function for LIME

* `code/test_patients.R` - List of test example patients

* `code/generate_figures.Rmd` - Figures for publication

* `pep_risk_app` - Shiny app
