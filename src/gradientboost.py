# -*- coding: utf-8 -*-
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from src.helper import norm, denorm


def train_gbdt(train_features, train_labels) :
    train_feat_stats = train_features.describe().transpose()
    train_label_stats = train_labels.describe().transpose()

    method = "stdev"
    norm_train_features = norm(train_features, train_feat_stats, method=method)
    norm_train_labels = norm(train_labels, train_label_stats, method=method)

    X_train, X_test, y_train, y_test = train_test_split(
        norm_train_features, norm_train_labels, test_size=0.2, shuffle=True
    )

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        "boosting_type" : "gbdt",
        "objective" : "regression",
        "metric" : {"logcosh", "l2", "l1"},
        "num_leaves" : 31,
        "learning_rate" : 0.05,
        "feature_fraction" : 0.9,
        "bagging_fraction" : 0.8,
        "bagging_freq" : 5,
        "verbose" : 0,
    }

    print("Starting training...")
    # train
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=200,
        valid_sets=lgb_eval,
        early_stopping_rounds=15,
    )

    print("Starting predicting...")

    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    # eval
    print("The rmse of prediction is:", mean_squared_error(y_test, y_pred) ** 0.5)
    diffs = abs(
        gbm.predict(train_features, num_iteration=gbm.best_iteration)
        - train_labels.values.T[0]
    )
    outputs = gbm.predict(train_features, num_iteration=gbm.best_iteration)
    difference = abs(
        train_labels
        - denorm(
            pd.DataFrame(outputs, columns=["Temperature"]),
            train_label_stats,
            method=method,
        )
    )

    return gbm, difference
