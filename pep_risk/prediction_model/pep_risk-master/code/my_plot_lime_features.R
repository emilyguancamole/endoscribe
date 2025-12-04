# my_plot_lime_features.R
# -----------------------------------------------------------------------------
# Author:            Venkata
# Date last modified: Jan 18, 2021
#
# Plot LIME features (adapted from plot_features in lime R package)
#library(lime)

my_plot_lime_features <- function (explanation, ncol = 2, cases = NULL){
  type_pal <- c("Increase Risk", "Decrease Risk")
  if (!is.null(cases)) {
    explanation <- explanation[explanation$case %in% cases,
                               , drop = FALSE]
  }
  if (nrow(explanation) == 0)
    stop("No explanations to plot", call. = FALSE)
  explanation$type <- factor(ifelse(sign(explanation$feature_weight) ==
                                      1, type_pal[1], type_pal[2]), levels = type_pal)
  description <- paste0(explanation$case, "_", explanation[["label"]])
  desc_width <- max(nchar(description)) + 1
  description <- paste0(format(description, width = desc_width),
                        explanation$feature_desc)
  explanation$description <- factor(description, levels = description[order(abs(explanation$feature_weight))])
  explanation$case <- factor(explanation$case, unique(explanation$case))
  explanation$`Explanation fit` <- format(explanation$model_r2,
                                          digits = 2)
  if (explanation$model_type[1] == "classification") {
    explanation$probability <- format(explanation$label_prob,
                                      digits = 2)
    explanation$label <- factor(explanation$label, unique(explanation$label[order(explanation$label_prob,
                                                                                  decreasing = TRUE)]))

  }
  ggplot(explanation) + geom_col(aes_(~description, ~feature_weight, fill = ~type)) +
    coord_flip() +
    scale_fill_manual(values = c("#D55E00", "#56B4E9"), drop = FALSE) +
    scale_x_discrete(labels = function(lab) substr(lab, desc_width + 1, nchar(lab))) +
    labs(title = "LIME weights",
         y = "Weight", x = "", fill = "") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 15),
          legend.position = "top",
          panel.grid.major.y = element_blank())
}
