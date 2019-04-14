# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={"figure.figsize" : (12, 8)})


def norm(x, data_stats, method=None) :
    if method == "stdev" :
        return (x - data_stats["mean"]) / (data_stats["std"])
    else :
        return (x - data_stats["min"]) / (data_stats["max"] - data_stats["min"])


def denorm(x, data_stats, method=None) :
    if method == "stdev" :
        denormed_x = x * data_stats["std"] + data_stats["mean"]
    else :
        denormed_x = x * (data_stats["max"] - data_stats["min"]) + data_stats["min"]
    return denormed_x


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
