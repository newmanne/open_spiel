from sklearn.decomposition import PCA

def projectPCA(X, components=10):
    """
    Find the principal axes of X.
    
    Arguments:
    - X: (n, d) array of data
    - components: number of PCA components to return
    
    Returns:
    - X': (n, components) array of data, projected onto principal components
    - explained_variance: ratio of variance explained by each axis
    """
    pca = PCA(n_components=components)
    X_transformed = pca.fit_transform(X)
    return X_transformed, pca.explained_variance_ratio_
