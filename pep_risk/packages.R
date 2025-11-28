options(repos = c(CRAN = "https://packagemanager.posit.co/cran/latest"))

packages <- c("lime", "gbm", "shiny", "shinythemes", "shinyWidgets", 
              "shinycssloaders", "tidyverse", "ggrepel", "janitor", 
              "caret", "FNN", "here")

install.packages(packages, type = "binary")