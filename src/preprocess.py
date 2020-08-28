import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class StdTransformer(BaseEstimator, TransformerMixin):
    """Normalize dataframe using standard scaler."""
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X):
        return self

    def transform(self, X):
        X = X.copy()
        data_stats = X.describe().T
        for feature in self.variables:
            X[feature] = (X - data_stats["mean"]) / (data_stats["std"])
        assert isinstance(X, pd.DataFrame)
        return X
