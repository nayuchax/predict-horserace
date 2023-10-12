import pandas as pd
from sklearn import linear_model


def regression_model(train_X: pd.DataFrame, train_y: pd.Series) -> object:
    model = linear_model.SGDRegressor(max_iter=10000)
    model_fitted = model.fit(train_X, train_y)

    return model_fitted
