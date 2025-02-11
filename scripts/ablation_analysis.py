import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def load_experiments(base_path):
    """
    Reads metrics_comb.pkl and hyperparameters.pkl from all experiment folders
    and stores them in a structured dictionary.

    Args:
        base_path (str): Path to the directory containing experiment folders.

    Returns:
        dict: A dictionary with experiment names as keys and another dictionary containing
              'metrics' and 'hyperparameters' as values.
    """
    experiments = {}

    for exp_name in os.listdir(base_path):
        exp_path = os.path.join(base_path, exp_name)
        if os.path.isdir(exp_path):  # Ensure it's a directory
            metrics_file = os.path.join(exp_path, "metrics_comb.pkl")
            hyperparams_file = os.path.join(exp_path, "hyperparameters.pkl")

            if os.path.exists(metrics_file) and os.path.exists(hyperparams_file):
                with open(metrics_file, "rb") as f:
                    metrics = pickle.load(f)

                with open(hyperparams_file, "rb") as f:
                    hyperparameters = pickle.load(f)

                experiments[exp_name] = {
                    "metrics": metrics,
                    "hyperparameters": hyperparameters
                }

    return experiments

def find_best_experiment(experiments, metric_name, direction="min"):
    """
    Finds the best experiment based on the given metric.

    Args:
        experiments (dict): The dictionary returned by `load_experiments`.
        metric_name (str): The name of the metric to optimize.
        direction (str): "max" for maximization, "min" for minimization.

    Returns:
        tuple: (best_experiment_name, best_value, best_std)
    """
    best_experiment = None
    best_value = None
    best_std = None

    for exp_name, data in experiments.items():
        if metric_name in data["metrics"]:
            mean, std = data["metrics"][metric_name]

            if best_value is None or \
               (direction == "max" and mean > best_value) or \
               (direction == "min" and mean < best_value):
                best_value = mean
                best_std = std
                best_experiment = exp_name

    return best_experiment, best_value, best_std

def plot_metric_across_experiments(experiments, metric_name, output_dir, best_experiment=None):
    """
    Plots a given metric across all experiments using discrete points with error bars and saves the plot.
    Optionally highlights the best experiment with a horizontal line and shaded region.

    Args:
        experiments (dict): The dictionary returned by `load_experiments`.
        metric_name (str): The name of the metric to plot.
        output_dir (str): Directory where the plot should be saved.
        best_experiment (tuple, optional): Output of `find_best_experiment`, i.e., 
                                           (best_experiment_name, best_mean, best_std).
    """
    import os

    # Extract and sort experiment names alphabetically
    sorted_exp_names = sorted(experiments.keys())

    means = []
    stds = []
    filtered_exp_names = []

    for exp_name in sorted_exp_names:
        if metric_name in experiments[exp_name]["metrics"]:
            mean, std = experiments[exp_name]["metrics"][metric_name]
            filtered_exp_names.append(exp_name)
            means.append(mean)
            stds.append(std)

    if not filtered_exp_names:
        print(f"No valid metric data found for {metric_name}")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Plotting using discrete points with error bars (no line joining)
    plt.figure(figsize=(10, 5))
    plt.errorbar(range(len(filtered_exp_names)), means, yerr=stds, fmt='o', capsize=5, elinewidth=2, markeredgewidth=2, label="Experiments")

    # Plot best experiment reference line if provided
    if best_experiment:
        _, best_mean, best_std = best_experiment
        plt.axhline(best_mean, color="red", linestyle="--", label="Best Mean")
        plt.fill_between(range(len(filtered_exp_names)), best_mean - best_std, best_mean + best_std, color="red", alpha=0.2, label="Best ± Std")

    plt.xlabel("Experiment")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Across Experiments")
    plt.xticks(range(len(filtered_exp_names)), filtered_exp_names, rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    # Save plot
    plot_path = os.path.join(output_dir, f"{metric_name}_comparison.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    print(f"Plot saved at: {plot_path}")


if __name__ == "__main__":
    base_path = "/home/users/o/oleksiyu/WORK/hyperproject/workspaces/HYPER/"
    output_dir = "/home/users/o/oleksiyu/WORK/hyperproject/workspaces/HYPER/figures/"
    os.makedirs(output_dir, exist_ok=True)
    
    experiments = load_experiments(base_path)
    metrics = [["mean_SBtoSB_closure_AUC", "min"], ["laSB_closure_AUC", "min"], ["mean_all_closures", "min"], ["gen_t", "min"], ["tra_t", "min"], ["tra_n_gen_t", "min"]]

    # Find the best experiment for a given metric
    for metric, direction in metrics:
        best_exp = find_best_experiment(experiments, metric, direction=direction)
        print(f"Best experiment for {metric}: {best_exp[0]} with mean {best_exp[1]} ± {best_exp[2]}")

        # Plot metric across experiments
        plot_metric_across_experiments(experiments, metric, output_dir, best_experiment=best_exp)