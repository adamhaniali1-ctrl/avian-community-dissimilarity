#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(gdm))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) stop("Usage: fit_final_gdm.R train.csv evaluation.csv model.rds predictions.csv parameters.csv")
train_path <- args[[1]]
evaluation_path <- args[[2]]
model_path <- args[[3]]
prediction_path <- args[[4]]
parameter_path <- args[[5]]

train <- read.csv(train_path, check.names = FALSE)
evaluation <- read.csv(evaluation_path, check.names = FALSE)
features <- sub("^s1\\.", "", grep("^s1\\.ssi_[0-9]{3}$", names(train), value = TRUE))
if (length(features) != 60) stop("GDM input does not contain 60 ordered SSI predictors")
if (!identical(features, sprintf("ssi_%03d", 0:59))) stop("GDM SSI order is invalid")
if (ncol(train) != 126 || ncol(evaluation) != 126) stop("GDM site-pair table has an invalid width")
if (any(!is.finite(as.matrix(train))) || any(!is.finite(as.matrix(evaluation)))) stop("GDM input is nonfinite")
if (any(train$weights != 1) || any(evaluation$weights != 1)) stop("GDM pair weights are not one")

model <- gdm(train, geo = FALSE, splines = rep(3, 60), knots = NULL)
before <- as.numeric(predict(model, evaluation))
if (length(before) != nrow(evaluation) || any(!is.finite(before))) stop("Invalid pre-save GDM predictions")
if (any(before < 0) || any(before > 1)) stop("GDM prediction outside [0,1]")
saveRDS(model, model_path, version = 3)
reloaded <- readRDS(model_path)
after <- as.numeric(predict(reloaded, evaluation))
if (!identical(before, after)) stop("Reloaded GDM predictions are not exact")

write.table(data.frame(prediction = after), prediction_path, sep = ",", row.names = FALSE, col.names = TRUE, quote = FALSE)
coefficients <- as.numeric(reloaded$coefficients)
knots <- as.numeric(reloaded$knots)
splines <- as.integer(reloaded$splines)
if (length(splines) != 60 || any(splines != 3)) stop("Saved spline structure is invalid")
if (length(coefficients) != 180 || length(knots) != 180) stop("Saved GDM parameter vectors are invalid")
rows <- vector("list", 180)
offset <- 0
for (j in seq_along(features)) {
  count <- splines[[j]]
  index <- seq.int(offset + 1, offset + count)
  total <- sum(coefficients[index])
  for (k in seq_len(count)) {
    rows[[offset + k]] <- data.frame(
      feature_name = features[[j]], spline_index = k,
      knot = knots[[offset + k]], coefficient = coefficients[[offset + k]],
      total_coefficient_contribution = total,
      zero_effect = identical(as.numeric(total), 0.0), stringsAsFactors = FALSE
    )
  }
  offset <- offset + count
}
parameters <- do.call(rbind, rows)
write.table(parameters, parameter_path, sep = ",", row.names = FALSE, col.names = TRUE, quote = FALSE)
