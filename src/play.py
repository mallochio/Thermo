# -*- coding: utf-8 -*-
import pandas as pd
from src.helper import norm, denorm


def play(model, data, method) :
    # breakpoint()
    stats = data.describe().transpose()
    norm_datapoint = norm(data, stats, method=method)
    prediction = model.predict(norm_datapoint)
    prediction_df = pd.DataFrame(prediction, columns=["Temperature"])
    return prediction_df
