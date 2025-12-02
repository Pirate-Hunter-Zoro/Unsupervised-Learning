import numpy as np
import math

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
    
    
class GaussianMixture:
    def __init__(self, n_clusters=3, max_iters=1000, tol=1e-4):
        """Initialization values for GaussianMixture object

        Args:
            n_clusters (int, optional): Number of gaussian clusters. Defaults to 3.
            max_iters (int, optional): Maximum iterations allowed to classify points. Defaults to 1000.
            tol (float, optional): Stop early if centroids of guessed 'clusters' move by less than this. Defaults to 1e-4.
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centers = None
        
    def _gaussian(self, X: np.array, mean: np.array, covariance: np.array) -> np.array:
        """For each of the input points return their probability that they were part of this gaussian

        Args:
            X (np.array): Input points
            mean (np.array): Mean of gaussian distribution
            covariance (np.array): Standard deviation of gaussian distribution

        Returns:
            np.array: Probabilities for each input point
        """
        k = X.shape[1] # number of features
        determinant = np.linalg.det(covariance)
        scaling_factor = 1 / math.sqrt((2*math.pi)**k*determinant)
        cov_inv = np.linalg.inv(covariance)
        left = (X - mean) @ cov_inv # (N x K) TIMES (K x K) -> (N x K)
        exponent = left * (X - mean) # Element-wise multiplication -> (N x K)
        exponent = np.sum(exponent, axis=1) # (N x 1)
        return scaling_factor*np.exp(-0.5*exponent) # (N x 1)
    
    def fit(self, X: np.array):
        """Method to fit gaussians according to the input observations X

        Args:
            X (np.array): Input observations
        """
        self.means = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)] # Shape (K, Features), whereas X has shape (N, Features)
        self.covariances = np.array([np.eye(X.shape[1]) for _ in range(self.n_clusters)]) # Shape (K, Features, Features)
        self.weights = 1/self.n_clusters * np.ones(shape=(self.n_clusters,1)) # Shape (K, 1)
        log_likelihoods = []
        
        for _ in range(self.max_iters):
            resp = np.ones(shape=(X.shape[0], self.n_clusters, 1)) # For each point, the responsibility of each gaussian distribution # (N, K, 1)
            for k in range(self.n_clusters):
                point_probs_from_cluster = self._gaussian(X, self.means[k], self.covariances[k]).reshape((X.shape[0], 1)) # (N, 1)
                resp[:, k, :] = self.weights[k] * point_probs_from_cluster
            resp_sum = resp.sum(axis=1, keepdims=True) # (N, 1, 1) - total likelihoods for each point
            resp = resp / resp_sum # (N, K, 1)
            
            # Update the weights, means, and covariances of our gaussian centers
            for k in range(self.n_clusters):
                cluster_resp_sum = np.squeeze(np.sum(resp[:, k, :], axis=0)) # (1, 1, 1) -> scalar
                cluster_resp_sum += 1e-10 # For numerical stability
                # Weight update
                self.weights[k] = cluster_resp_sum / X.shape[0]
                # Mean update - cluster responsibilities of each point over the total responsibility of all clusters
                self.means[k] = np.dot(resp[:, k].T, X) / cluster_resp_sum # (N, Features)
                # Covariance update
                diff = X - self.means[k] # (N x Features)
                self.covariances[k] = np.dot(diff.T, diff * resp[:, k]) / cluster_resp_sum # (Features x Features)
                
            # Ensure point responsibility values changed sufficiently to consider continuing the iterations
            log_likelihoods.append(np.sum(np.log(resp_sum)))
            if len(log_likelihoods) > 1:
                if np.abs(log_likelihoods[-2] - log_likelihoods[-1]) < self.tol:
                    break
                
    def predict(self, X: np.array) -> np.array:
        """Given an input set of observations, predict their classes

        Args:
            X (np.array): Input observations

        Returns:
            np.array: Resulting class predictions
        """
        resp = np.ones(shape=(X.shape[0], self.n_clusters, 1)) # For each point, the responsibility of each gaussian distribution - (N, K, 1)
        for k in range(self.n_clusters):
            resp[:, k, :] = self.weights[k] * self._gaussian(X, self.means[k], self.covariances[k]).reshape(-1,1) # (N, 1)
        return np.argmax(resp, axis=1).flatten()