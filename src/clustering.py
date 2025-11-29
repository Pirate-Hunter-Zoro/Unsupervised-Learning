import numpy as np
import random

def euclidean_distance(a: np.array, b: np.array) -> float:
    """Euclidean distance helper

    Args:
        a (np.array): first point
        b (np.array): second point

    Returns:
        float: resulting distance
    """
    return np.linalg.norm(a-b)

class KMeans:
    
    def __init__(self, n_clusters=3, max_iters=1000, tol=1e-4):
        """Initialization values for KMeans clustering object

        Args:
            n_clusters (int, optional): Number of gaussian clusters. Defaults to 3.
            max_iters (int, optional): Maximum iterations allowed to classify points. Defaults to 1000.
            tol (float, optional): Stop early if centroids of guessed 'clusters' move by less than this. Defaults to 1e-4.
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        
    def fit(self, X: np.array):
        """Method to run the kmeans algorithm to estimate the centroids

        Args:
            X (np.array): Points generated from various gaussian distributions (whose means and standard deviations we do not know here)
        """
        # TODO - keep in numpy
        centers = np.array(random.sample(X.tolist(), self.n_clusters))
        
        for _ in range(self.max_iters):
            # Assign each point to its closest centroid
            centroid_distances = [()]