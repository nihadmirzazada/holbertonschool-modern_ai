#!/usr/bin/env python3
"""Task 3: Choosing the Optimal K for K-Means using silhouette scores and inertia."""

from sklearn import metrics

K_Means = __import__('2-k_means').K_Means


def optimal_k(X, max_clusters, random_state):
    """
    Evaluates K-Means clustering quality using silhouette scores and inertia
    for cluster counts from 2 up to max_clusters.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features)
        max_clusters (int): Maximum number of clusters to evaluate (>=2)
        random_state (int): Random seed for reproducibility

    Returns:
        list[int]: Evaluated cluster numbers (from 2 to max_clusters)
        list[float]: Inertia values corresponding to each cluster number
        list[float]: Silhouette scores corresponding to each cluster number
    """
    ks = list(range(2, max_clusters + 1))
    inertia_values = []
    silhouette_values = []

    for k in ks:
        model = K_Means(X, n_clusters=k, random_state=random_state)
        inertia_values.append(model.inertia_)
        score = metrics.silhouette_score(X, model.labels_)
        silhouette_values.append(score)

    return ks, inertia_values, silhouette_values
