# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from src import config


def compile_model(model, params=None):
    if not params:
        # Default to currently best values
        optimizer = tf.keras.optimizers.Adam(
            lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=1.0
        )
    else:
        optimizer = tf.keras.optimizers.Adam(
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


def build_mobilenet(params=None):
    mobilenet= tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=[None, None, config.NFEATURES, None],
        include_top=False,
    )
    x = mobilenet.output
    out = tf.keras.layers.Dense(1)(x)
    model = Model(inputs=[mobilenet.inputs], outputs=out)
    model = compile_model(model, params)
    return model


def build_model(params=None):
    model = tf.keras.Sequential(
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
    model = compile_model(model, params)
    return model


def train_neural_net(X_train, y_train, params=None):
    model = build_model(params)
    hist = model.fit(X_train, y_train, epochs=100)
    return model