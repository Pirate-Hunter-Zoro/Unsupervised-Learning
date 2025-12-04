# Unsupervised Learning: PCA, k-Means, and Expectation Maximization

## Overview

This project implements and analyzes three fundamental unsupervised learning algorithms from scratch:

1. **Principal Component Analysis (PCA)**: Used for dimensionality reduction, eigenface generation, and noise filtering on the Fashion MNIST and PubFig datasets.
2. **k-Means Clustering**: Implemented manually to cluster synthetic Gaussian data and compress real images.
3. **Expectation Maximization (EM)**: Implemented as a Gaussian Mixture Model (GMM) to analyze synthetic datasets with soft clustering.

## Project Structure

```text
.
├── data/                   # Dataset storage (Fashion MNIST, PubFig, Custom Images)
├── results/                # Output graphs and compressed images
├── src/                    # Source code for algorithms and utilities
│   ├── __init__.py
│   ├── clustering.py       # Custom implementations of KMeans and GaussianMixture
│   ├── pca_analysis.py     # PCA logic using Scikit-Learn
│   ├── synthetic_data.py   # Generators for synthetic Gaussian mixtures
│   ├── synthetic_data_runner.py # Execution logic for synthetic trials
│   └── visualization.py    # Plotting helpers
├── experiments.ipynb       # Main execution notebook for all experiments
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
````

## Setup and Installation

1. **Environment**: Ensure you are running Python 3.8+.
2. **Dependencies**: Install the required packages.

    ```bash
    pip install -r requirements.txt
    ```

    *Key dependencies:* `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scikit-image`.

## Usage

### 1\. Synthetic Data Experiments (k-Means & EM)

To run the 6 parameter variation experiments (varying sources, points, sigma, spacing, etc.):

- Open `experiments.ipynb`.
- Execute the cells corresponding to the synthetic data generation and model fitting.
- Results (Rand Index scores) are saved to `results/*.json`.

### 2\. Image Compression (k-Means)

- Place your target images in `data/images/`.
- Run the image compression cells in `experiments.ipynb`.
- The algorithm will compress colors to $k \in \{3, 5, 10\}$.
- Output comparison images are saved to `results/compressed_images/`.

### 3\. PCA Analysis (Fashion MNIST & PubFig)

- **Fashion MNIST**:
  - The code handles data loading (via local CSV fallback or direct binary download).
  - Analyses included: Variance ratio plot, 2D projection, and Denoising visualization.
- **PubFig (Eigenfaces)**:
  - Ensure the `CelebDataProcessed` folder is in `data/`.
  - Run the Eigenfaces section in `experiments.ipynb` to visualize the principal components and face reconstruction.

## Implementation Details

- **Clustering**: The `KMeans` and `GaussianMixture` classes are implemented purely in NumPy (vectorized) without external clustering libraries.
- **PCA**: Uses `sklearn.decomposition.PCA` for matrix decomposition.
- **Data Loading**: Includes robust fallback mechanisms for loading datasets when standard APIs fail.
