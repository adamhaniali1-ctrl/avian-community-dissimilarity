#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(gdm))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("Usage: validate_final_gdm.R model.rds evaluation.csv predictions.csv")
model <- readRDS(args[[1]])
evaluation <- read.csv(args[[2]], check.names = FALSE)
expected <- read.csv(args[[3]], check.names = FALSE)
prediction <- as.numeric(predict(model, evaluation))
if (length(prediction) != nrow(expected)) stop("GDM validation row count mismatch")
if (any(!is.finite(prediction)) || any(prediction < 0) || any(prediction > 1)) stop("Invalid GDM validation prediction")
difference <- max(abs(prediction - as.numeric(expected$prediction)))
if (!is.finite(difference) || difference > 1e-15) stop(paste("GDM validation prediction mismatch; maximum difference", format(difference, digits = 17)))
cat("PASS\n")
