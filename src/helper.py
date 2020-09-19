# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={"figure.figsize" : (12, 8)})


def plot_history(history) :
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Logcosh Error")
    plt.plot(hist["epoch"], hist["logcosh"], label="Train Error")
    plt.plot(hist["epoch"], hist["val_logcosh"], label="Val Error")
    # plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Abs Error")
    plt.plot(hist["epoch"], hist["mean_absolute_error"], label="Train Error")
    plt.plot(hist["epoch"], hist["val_mean_absolute_error"], label="Val Error")
    # plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Square Error")
    plt.plot(hist["epoch"], hist["mean_squared_error"], label="Train Error")
    plt.plot(hist["epoch"], hist["val_mean_squared_error"], label="Val Error")
    # plt.ylim([0,20])
    plt.legend()
    plt.show()


def norm(df, stats=None, method=None):
    df = df.astype(float)
    result = df.copy()
    for feature_name in df.columns:
        if feature_name != 'Identifier':
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            diff = max(1e-6, max_value - min_value)
            result[feature_name] = (df[feature_name] - min_value) / diff
    return result


def denorm(df, stats, method=None):
    diff = stats['max'].iloc[0] - stats['min'].iloc[0]
    df['Temperature'] = df['Temperature'].map(lambda x: x * diff + stats['min'].iloc[0])
    return df