#!/usr/bin/env python3
"""Module for feature standardization using Scikit-learn."""
from sklearn import preprocessing


def Standardize(X):
    """Standardize input data using StandardScaler."""
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)
