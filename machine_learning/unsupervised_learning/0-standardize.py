#!/usr/bin/env python3
"""Task 0: Feature Standardization using Scikit-learn."""

from sklearn import preprocessing


def Standardize(X):
    """
    Standardizes tabular data using Scikit-learn's StandardScaler.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features)

    Returns:
        numpy.ndarray: The standardized version of the input data,
                       with the same shape as X.
    """
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
