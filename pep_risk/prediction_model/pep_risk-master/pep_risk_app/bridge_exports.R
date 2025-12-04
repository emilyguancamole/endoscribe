# bridge_exports.R
# -----------------------------------------------------------------------------
# Author:             Priya Madduluri
# Date last modified: Nov 23, 2025
#
# Python bridge for PEP risk prediction 
user_lib <- "~/Rlibs"
dir.create(user_lib, showWarnings = FALSE, recursive = TRUE)
.libPaths(c(user_lib, .libPaths()))

# ---- 1) Minimal backend packages only ----
needed <- c("dplyr", "caret", "FNN", "lime","gbm")

# ---- 2) Install ONLY what is missing (Depends + Imports, not Suggests) ----
missing <- needed[!sapply(needed, requireNamespace, quietly = TRUE)]

if (length(missing) > 0) {
  message("Installing missing packages: ", paste(missing, collapse = ", "))
  
  install.packages(
    missing,
    repos = "https://cloud.r-project.org",
    dependencies = NA   # IMPORTANT: installs Depends+Imports only, skips Suggests
  )
}

# ---- 3) Fail fast if still missing after install ----
missing_after <- needed[!sapply(needed, requireNamespace, quietly = TRUE)]
if (length(missing_after) > 0) {
  stop(
    "Missing packages even after install: ",
    paste(missing_after, collapse = ", "),
    "\nInstall them once in the R used by rpy2 (conda-forge preferred)."
  )
}

# ---- 4) Attach packages ----
suppressPackageStartupMessages({
  library(dplyr)
  library(caret)
  library(FNN)
  library(lime)
  library(gbm)
})

# Determine project root directory
# Check if we're in the pep_risk_app directory, if so go up one level
current_dir <- getwd()
if (basename(current_dir) == "pep_risk_app") {
  setwd("..")
} else if (file.exists("pep_risk_app/Peprisc_cal.R")) {
  # Already in project root
  # Do nothing
} else if (file.exists("prediction_model/pep_risk-master/pep_risk_app/Peprisc_cal.R")) {
  # We're in endoscribe directory
  setwd("prediction_models/pep_risk-master")
} else {
  stop("Cannot find pep_risk-master project root. Current directory: ", current_dir)
}

source("./code/my_plot_lime_features.R")
source("./pep_risk_app/Peprisc_cal.R")



