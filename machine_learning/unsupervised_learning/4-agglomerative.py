#!/usr/bin/env python3
"""Task 4: Agglomerative Hierarchical Clustering using Scikit-learn."""

from sklearn import cluster
from sklearn import metrics

Apply_PCA = __import__('1-pca').Apply_PCA


def Agglomerative_Clustering(X, n_clusters, random_state, n_components,
                              use_pca_data=True):
    """
    Performs Agglomerative hierarchical clustering on tabular data.

    The function performs three main tasks:
    1. Dimensionality reduction (optional): Apply PCA to reduce data to
       n_components dimensions if use_pca_data=True.
    2. Clustering: Fit an Agglomerative Clustering model on the
       (PCA-reduced or original) data using Ward linkage.
    3. Evaluation: Compute the silhouette score for the clustering
       (if n_clusters > 1).

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features)
        n_clusters (int): Number of clusters
        random_state (int): Random seed for reproducibility
        n_components (int): Number of PCA components (used only if use_pca_data=True)
        use_pca_data (bool, default=True): Whether to apply PCA before clustering

    Returns:
        sklearn.cluster.AgglomerativeClustering: Fitted Agglomerative Clustering instance
        numpy.ndarray: Data used for fitting (PCA-reduced or original)
        float: Silhouette score of the clustering (None if n_clusters=1)
    """
    if use_pca_data:
        X_used, _ = Apply_PCA(X, n_components=n_components,
                               random_state=random_state)
    else:
        X_used = X

    model = cluster.AgglomerativeClustering(n_clusters=n_clusters,
                                            linkage='ward')
    model.fit(X_used)

    if n_clusters > 1:
        score = metrics.silhouette_score(X_used, model.labels_)
    else:
        score = None

    return model, X_used, score
