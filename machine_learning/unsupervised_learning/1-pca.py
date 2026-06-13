#!/usr/bin/env python3
from sklearn import decomposition


def Apply_PCA(X, n_components, random_state):
    pca = decomposition.PCA(
        n_components=n_components,
        random_state=random_state
    )
    X_pca = pca.fit_transform(X)
    return X_pca, pca
