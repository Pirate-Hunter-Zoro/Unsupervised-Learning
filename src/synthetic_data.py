import numpy as np
from typing import Tuple

def generate_mixture(num_points_per_source: int, means: list[float], stds: list[float]) -> Tuple[np.array, np.array]:
    """Create random classified points generated from Gaussian Distributions

    Args:
        num_points_per_source (int): Number of points from each gaussian distribution
        means (list[float]): Means of each gaussian distribution
        stds (list[float]): Standard deviations of each gaussian distribution

    Returns:
        Tuple[np.array, np.array]: Resulting points and respective gaussians
    """
    if len(means) != len(stds):
        raise ValueError(f"Error - uneven number of means ({len(means)}) and standard deviations ({len(stds)})...")
    
    X = np.zeros(shape=(len(means), num_points_per_source))
    y = np.zeros(shape=X.shape)
    for i, (mean, std) in enumerate(zip(means, stds)):
        points = np.random.normal(loc=mean, scale=std, size=num_points_per_source)
        X[i] = points
        y[i, :] = i
    
    X = X.reshape(-1,1)  
    y = y.flatten()
    return (X, y)