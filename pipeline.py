# -*- coding: utf-8 -*-
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from src import config
from src.neuralnet import build_model

reg = KerasRegressor(
    build_fn=build_model,
    epochs=config.EPOCHS,
    validation_split=config.VALIDATION_SPLIT,
    shuffle=False,
    verbose=0,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=80),
        keras.callbacks.TensorBoard(log_dir=config.TF_LOG_DIR, histogram_freq=1),
    ],
)

neuralPipe = Pipeline([("scale", StandardScaler()), ("regressor", reg)])
