import matplotlib.pyplot as plt
from src.clustering import KMeans
import os
from pathlib import Path
import numpy as np

def compress_image(image_path_str: str, save_path_str: str):
    """Helper method to run image compression on the given images with different numbers of clusters

    Args:
        image_path_str (str): Original image
        save_path_str (str): Output graph location
    """
    X_orig = plt.imread(image_path_str)
    height = X_orig.shape[0]
    width = X_orig.shape[1]
    # In case transparency value is present in addition to the three colors
    X = X_orig[:, :, :3]
    X = X.reshape(-1,3) # Every pixel has 3 color values - just flatten it into such a vector where each component has 3 values
    if X.dtype == np.uint8:
        X = X / 255.0
    
    # Display the results for each k value as well as the original image
    _, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(X_orig)
    axes[0].set_title("Original")
    axes[0].axis("off")
    for i, k in enumerate([3, 5, 10]):
        classifier = KMeans(n_clusters=k)
        classifier.fit(X)
        labels = classifier.predict(X)
        # Every pixel now has a center whose color it is assigned to
        new_pixels = classifier.centers[labels]
        new_pixels = new_pixels.reshape((height, width, 3))
        
        axes[i+1].imshow(new_pixels)
        axes[i+1].set_title(f"{k}-bin Compression")
        axes[i+1].axis("off")

    graph_path = Path(f"results/{save_path_str}.png")
    os.makedirs(graph_path.parent, exist_ok=True)
    plt.savefig(str(graph_path))
    plt.close()