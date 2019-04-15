# -*- coding: utf-8 -*-
from mlflow import log_metric, keras as mlflow_keras


def log(model, error):
    log_metric("mae", error.describe().T["mean"])
    log_metric("median_ae", error.describe().T["50%"])
    log_metric("max_ae", error.describe().T["max"])
    log_metric("min_ae", error.describe().T["min"])
    mlflow_keras.log_model(model, "models")
