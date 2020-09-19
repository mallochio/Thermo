# -*- coding: utf-8 -*-
import os
import shutil
import sys
import warnings

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import src
from run_pipeline import run_training, run_testing
from src import config
from src import logger

if os.path.isdir("./mlruns"):
    shutil.rmtree("./mlruns/")

if os.path.isdir("./logs"):
    shutil.rmtree("./logs/")

experiment_id = mlflow.create_experiment("Predict_Temperature")
np.random.seed = 42

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # read training data
    data = src.data_manager.load_dataset(file_name=config.DATA_FILE)

    # divide train and test
    train, test = train_test_split(data, shuffle=True, test_size=0.1, random_state=0)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    x_train, x_test, y_train, y_test = (
        train[config.FEATURES],
        test[config.FEATURES],
        train[config.LABELS],
        test[config.LABELS],
    )
    params = [float(i) for i in sys.argv[1:]] if len(sys.argv) > 1 else None

    for ntrial in range(1):
        with mlflow.start_run(experiment_id=experiment_id):
            pipeline = run_training(x_train, y_train)
            predictions = run_testing(x_test, pipeline)
            predictions = pd.DataFrame(predictions, columns=config.LABELS)
            test_error = abs((y_test - predictions)["Temperature"])
            print(max(test_error))
            # logger.log(model=model, error=test_error)
