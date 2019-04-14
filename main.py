# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import mlflow
from mlflow import log_metric, keras as mlflow_keras
from sklearn.model_selection import train_test_split
from src.neuralnet import train_neural_net
from src.gradientboost import train_gbdt
from src.helper import norm
from src.play import play
from src.helper import denorm
import shutil

shutil.rmtree("./mlruns/")
shutil.rmtree("./logs/")
experiment_id = mlflow.create_experiment("Predict_Temperature")
np.random.seed = 42
features = [
    "D1",
    "D2",
    "D1D2",
    "Perimeter1",
    "Perimeter2",
    "Area1",
    "Area2",
    "Identifier",
]
labels = ["Temperature"]


if __name__ == "__main__":
    df = pd.read_excel("data/ML_Data.xlsx")
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffles dataframe rows randomly
    train, test = train_test_split(df, shuffle=False, test_size=0.2)
    test = test.reset_index(drop=True)
    for ntrial in range(50):
        with mlflow.start_run(source_name="main.py", source_version="0"):
            train = train.sample(frac=1).reset_index(drop=True)
            X_train = train[features]
            y_train = train[labels]
            # train_features['Identifier'] = train_features['Identifier'].map({1: 100, 0: 0})

            method = "stdev"
            model, train_error = train_neural_net(X_train, y_train)
            # model, difference = train_gbdt(train_features, train_labels)

            X_test = test[features]
            y_test = test[labels]
            test_stats = y_test.describe().transpose()
            norm_predictions = play(model=model, data=X_test, method=method)
            predictions = denorm(norm_predictions, test_stats, method=method)
            test_error = abs((y_test - predictions)["Temperature"])
            log_metric("mae", test_error.describe().T["mean"])
            log_metric("median_ae", test_error.describe().T["50%"])
            log_metric("max_ae", test_error.describe().T["max"])
            log_metric("min_ae", test_error.describe().T["min"])
            mlflow_keras.log_model(model, "models")
