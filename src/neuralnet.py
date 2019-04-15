# -*- coding: utf-8 -*-
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime
from src.helper import norm, denorm


def build_model(train_features, train_labels, params=None):
    model = keras.Sequential(
        [
            layers.Dense(
                64,
                activation=tf.nn.relu,
                input_shape=[len(train_features.keys())],
                kernel_initializer="glorot_normal",
            ),
            layers.BatchNormalization(),
            # layers.Dropout(0.1),
            layers.Dense(
                128, activation=tf.nn.relu, kernel_initializer="glorot_normal"
            ),
            layers.BatchNormalization(),
            # layers.Dropout(0.1),
            layers.Dense(len(train_labels.keys())),
        ]
    )
    if not params:
        # Default to currently best values
        optimizer = keras.optimizers.Adam(
            lr=0.05,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            clipnorm=1.0
        )
    else:
        optimizer = keras.optimizers.Adam(
            lr=params[0],
            beta_1=params[1],
            beta_2=params[2],
            epsilon=params[3],
            clipnorm=params[4]
        )

    model.compile(
        loss="logcosh",
        optimizer=optimizer,
        metrics=["logcosh", "mean_absolute_error", "mean_squared_error"],
    )
    return model


def train_neural_net(train_features, train_labels, params=None):
    train_feat_stats = train_features.describe().transpose()
    train_label_stats = train_labels.describe().transpose()
    method = "stdev"
    norm_train_features = norm(train_features, train_feat_stats, method=method)
    norm_train_labels = norm(train_labels, train_label_stats, method=method)
    model = build_model(norm_train_features, norm_train_labels, params=params)
    EPOCHS = 500
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    history = model.fit(
        norm_train_features,
        norm_train_labels,
        epochs=EPOCHS,
        validation_split=0.30,
        shuffle=True,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=80),
            tensorboard_callback,
        ],
    )

    # plot_history(history)
    outputs = denorm(
        pd.DataFrame(model.predict(norm_train_features), columns=["Temperature"]),
        train_label_stats,
        method=method,
    )

    difference = abs((train_labels - outputs)["Temperature"])
    return model, difference
