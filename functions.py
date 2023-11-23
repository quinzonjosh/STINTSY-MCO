import numpy as np


def compute_RMSE(y_true, y_pred):
    # TODO: Compute the Root Mean Squared Error
    rmse = np.sqrt( np.square(np.subtract(y_true, y_pred)).mean() )

    return rmse


def compute_R2_Score(y_true, y_pred):
    # Compute the mean of the true values
    mean_true = sum(y_true) / len(y_true)

    # Compute Total Sum of Squares
    tss = sum((yi - mean_true) ** 2 for yi in y_true)

    # Compute Residual Sum of Squares
    rss = sum((yi - yhat) ** 2 for yi, yhat in zip(y_true, y_pred))

    # Compute R-squared
    r2_score = 1 - rss / tss

    return r2_score