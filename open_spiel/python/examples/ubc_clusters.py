from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import umap
from tqdm import tqdm

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

def projectUMAP(X, n_neighbors=15, min_dist=0.1, rescale=False):
    # TODO: make UMAP settings configurable. unclear how we'd like to set them, or if good constant values exist.
    umap_reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    if rescale:
        X = StandardScaler().fit_transform(X)
    umap_embedding = umap_reducer.fit_transform(X)
    return umap_embedding

def fitGMM(X, verbose=False):
    """
    Fit a Gaussian mixture model, choosing the number of components to minimize AIC score.
    
    TODO: make range of n_components configurable.

    Arguments:
    - X: (n, d) array of data

    Returns: tuple of
    - gmm: GaussianMixture with lowest AIC score
    - clusters: result of gmm.predict(X)
    - scores: AIC scores for all models
    """

    gmms = {}
    clusters = {}
    scores = {}
    for n_components in tqdm([1, 2, 5, 10, 20, 50, 100], disable=not verbose): # TODO: don't hardcode
        gmms[n_components] = GaussianMixture(n_components=n_components)
        clusters[n_components] = gmms[n_components].fit_predict(X)
        scores[n_components] = gmms[n_components].aic(X)

    # sort by score and pick lowest
    best_n_components = min(scores.items(), key = lambda x: x[1])[0]
    return gmms[best_n_components], clusters[best_n_components], scores


