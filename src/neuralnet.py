# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src import config


def build_model(params=None):
    model = keras.Sequential(
        [
            layers.Dense(
                64,
                activation=tf.nn.relu,
                input_shape=[config.NFEATURES],
                kernel_initializer="glorot_normal",
            ),
            layers.BatchNormalization(),
            # layers.Dropout(0.1),
            layers.Dense(
                128, activation=tf.nn.relu, kernel_initializer="glorot_normal"
            ),
            layers.BatchNormalization(),
            # layers.Dropout(0.1),
            layers.Dense(config.NLABELS),
        ]
    )
    if not params:
        # Default to currently best values
        optimizer = keras.optimizers.Adam(
            lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=1.0
        )
    else:
        optimizer = keras.optimizers.Adam(
            lr=params[0],
            beta_1=params[1],
            beta_2=params[2],
            epsilon=params[3],
            clipnorm=params[4],
        )

    model.compile(
        loss="logcosh",
        optimizer=optimizer,
        metrics=["logcosh", "mean_absolute_error", "mean_squared_error"],
    )
    return model
