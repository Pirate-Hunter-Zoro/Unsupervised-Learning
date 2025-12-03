from src.synthetic_data import generate_mixture
from sklearn.metrics import adjusted_rand_score
from pathlib import Path
import os
import json
import random

def run_experiment(experiment_name: str, model_class: any, param_to_vary: str, param_values: list, default_params: dict):
    """Perform experiments based on the input model and the specified variables to both change and leave fixed

    Args:
        experiment_name (str): Label of experiment
        model_class (any): Classification model class (KMeans or GaussianMixture) - NOT AN INSTANTIATION but a reference to the actual class
        param_to_vary (str): Parameter to modify
        param_values (list): Values to set said parameter to
        default_params (dict): Fixed values for all other parameters
    """
    scores = []
    for v in param_values:
        params_copy = default_params.copy()
        params_copy[param_to_vary] = v
        num_sources, num_clusters, num_points, sigma, delta_mu = params_copy['num_sources'], params_copy['num_clusters'], params_copy['num_points'], params_copy['sigma'], params_copy['delta_mu']
        gaussian_means = [delta_mu*i for i in range(num_sources)]
        # Trials
        avg_score = 0
        for _ in range(params_copy['num_trials']):
            if sigma == "random":
                std_devs = [0.75 + 1.25*random.random() for _ in range(num_sources)]
            else:
                std_devs = [sigma for _ in range(num_sources)]
            # Source points from the gaussians, and keep track of the true gaussian distribution each point came from
            X, y_true = generate_mixture(num_points_per_source=num_points, means=gaussian_means, stds=std_devs)
            X = X.reshape((X.shape[0],1))
            y_true = y_true.flatten()
            # The number of clusters the model guesses from is the same as the number of sources the data is sampled from unless otherwise specified
            num_clusters = num_sources if num_clusters is None else num_clusters
            model = model_class(n_clusters=num_clusters)
            model.fit(X)
            y = model.predict(X).flatten()
            avg_score += adjusted_rand_score(labels_true=y_true, labels_pred=y)
        avg_score /= params_copy['num_trials']
        scores.append({
            "parameters": params_copy,
            "avg_score": avg_score
        })
    
    results_path = Path(f"results/clustering/{experiment_name}.json")
    os.makedirs(results_path.parent, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(scores, f, indent=4)