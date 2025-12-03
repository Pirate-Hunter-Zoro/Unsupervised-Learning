import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple
import pandas as pd
from pathlib import Path

def load_fashion_mnist() -> Tuple[np.array, np.array]:
    """Helper method to load the fashion mnist data and normalize the pixel values

    Returns:
        Tuple[np.array, np.array]: Normalized pixel-value observations paired with their classes
    """
    data_path = Path("data/mnist/fashion-mnist_train.csv")
    df = pd.read_csv(data_path)
    y = np.array(df.iloc[:, 0].values, dtype=int)
    X = np.array(df.iloc[:, 1:].values)
    X = X / 255.0 # Normalize pixel values
    return (X, y)

def run_pca(X: np.array, n_components: int) -> Tuple[np.array, PCA]:
    """Run principal component analysis given the input data and the number of desired resulting PCA components

    Args:
        X (np.array): Input observations
        n_components (int): Desired number of components

    Returns:
        Tuple[np.array, PCA]: Reduced data matrix and resulting PCA object
    """
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X)
    return (X_transformed, pca)