import numpy as np

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
        self.centers = None
        
    def fit(self, X: np.array):
        """Method to run the kmeans algorithm to estimate the centroids

        Args:
            X (np.array): Points generated from various gaussian distributions (whose means and standard deviations we do not know here)
        """
        centers = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)] # Shape (K, Features), whereas X has shape (N, Features)
        X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
        centers = centers.reshape(1, centers.shape[0], centers.shape[1])
        # Now when we subtract, numpy will automatically broadcast and we'll get shape (N, K, Features), where each point will have its distance vector from all center points
        for _ in range(self.max_iters):
            # Assign each point to its closest centroid
            diff_vectors = X_reshaped - centers # (N, K, Features)
            center_distances = np.linalg.norm(diff_vectors, axis=2) # (N, K)
            # Each point is assigned the center it is closest to
            assignments = np.argmin(center_distances, axis=1) # (N, 1)
            # Now each class assignment has a different centroid
            new_centers = centers.copy()
            for center_class in range(self.n_clusters):
                points = X[assignments==center_class]
                if points.shape[0] == 0:
                    continue
                class_mean = np.mean(points, axis=0)
                new_centers[0][center_class] = class_mean
            if np.linalg.norm(new_centers - centers) < self.tol:
                break
            else:
                centers = new_centers
        self.centers = centers.reshape(centers.shape[1], centers.shape[2]) # (K, Features)
        
    def predict(self, X: np.array) -> np.array:
        """Prediction method to use the class instance's centers to predict the class of each observation in X

        Args:
            X (np.array): Input observations

        Returns:
            np.array: Output predicted classes
        """
        X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
        diff_vectors = X_reshaped - self.centers.reshape(1, self.centers.shape[0], self.centers.shape[1]) # (N, K, Features)
        center_distances = np.linalg.norm(diff_vectors, axis=2) # (N, K)
        # Each point is assigned the center it is closest to
        return np.argmin(center_distances, axis=1) # (N, 1)