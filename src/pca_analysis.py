import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple
import pandas as pd
from pathlib import Path
import glob
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize

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

def load_pubfig(data_dir: Path, limit: int=500) -> np.array:
    """Load a bunch of images from the specifiied data directory into one 2D numpy array of each vectorized image

    Args:
        data_dir (Path): Location of images
        limit (int, optional): Number of images to be put into the array. Defaults to 500.

    Returns:
        np.array: Each vectorized image
    """
    all_images = glob.glob(pathname="**/*.jpg", root_dir=data_dir, recursive=True)
    processed_faces = []
    for image in all_images:
        if len(processed_faces) >= limit:
            break
        else:
            # Turns image into 2D matrix
            loaded_image_array = rgb2gray(imread(data_dir / image))
            transformed_image = resize(loaded_image_array, output_shape=(64, 64)).flatten() # 1 x 4,096
            processed_faces.append(transformed_image)
    return np.array(processed_faces)