import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def search_files(folder, target_name, target_extension):
    """Recursively search for files with a specific name and extension."""
    result_files = []
    for path in Path(folder).rglob(f'{target_name}{target_extension}'):
        result_files.append(path)
    return result_files

def load_pickle_files(file_paths):
    """Load pickle files and store dictionaries in a new dictionary."""
    data_dict = {}
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            # Construct the key from the parent.parent folder name
            parent_folder = file_path.parent.parent.parent.parent.parent.name + '/' + file_path.parent.parent.parent.parent.name + '/' + file_path.parent.parent.parent.name
            data_dict[parent_folder] = data
    return data_dict

def compute_metrics(data_dict, metric_name):
    """Compute average, standard deviation, minimum, and maximum for a given metric."""
    metric_stats = {}
    metric_values = [data[metric_name] for data in data_dict.values()]
    if metric_values:
        metric_stats = {
            'metric_values': metric_values,
            'mean': np.mean(metric_values),
            'std': np.std(metric_values),
            'outliers': [value for value in metric_values if value < np.mean(metric_values) - np.std(metric_values) or value > np.mean(metric_values) + np.std(metric_values)]
        }
    if metric_stats=={}:
        print(f"Warning: No data found for metric '{metric_name}' in the provided data.")
    return metric_stats

def plot_metrics(metrics, metric_name, reference_name=None, display_all_points=False, save_dir="user/plots"):
    """Create an error bar plot with the computed metrics.

    Args:
        metrics (dict): A dictionary containing computed metrics for each experiment.
        metric_name (str): The name of the metric being plotted.
        reference_name (str, optional): The reference experiment for comparison.
        display_all_points (bool, optional): Whether to display all points instead of mean, std, and outliers.
        save_dir (str, optional): The directory where the figure should be saved. Defaults to "user/plots".
    """
    keys = list(metrics.keys())
    means = [metrics[key]['mean'] for key in keys]
    stds = [metrics[key]['std'] for key in keys]
    outliers = [metrics[key]['outliers'] for key in keys]
    all_points = [metrics[key]['metric_values'] for key in keys]

    # Count and print the number of points for each folder
    print(f"Number of points for metric '{metric_name}':")
    for key, points in zip(keys, all_points):
        print(f"  {key}: {len(points)} points")

    plt.figure(figsize=(10, 6))
    colors = plt.get_cmap('tab10', len(keys))

    for i, key in enumerate(keys):
        if display_all_points:
            plt.scatter([key] * len(all_points[i]), all_points[i], color=colors(i), marker='o', label=f'{key} Points')
        else:
            # Add error bars with caps
            plt.errorbar(
                [key], [means[i]], yerr=[stds[i]], fmt='o', color=colors(i),
                capsize=10, label=f'{key} Mean ± Std'
            )
            plt.scatter([key] * len(outliers[i]), outliers[i], color=colors(i), marker='x', label=f'{key} Outliers')

    if reference_name and not display_all_points:
        if reference_name == 'min':
            reference_key = keys[np.argmin(means)]
        elif reference_name == 'max':
            reference_key = keys[np.argmax(means)]
        else:
            reference_key = reference_name

        if reference_key in metrics:
            ref_mean = metrics[reference_key]['mean']
            ref_std = metrics[reference_key]['std']
            plt.axhline(ref_mean, color='black', linestyle='--', label=f'{reference_key} Mean')
            plt.fill_between(keys, ref_mean - ref_std, ref_mean + ref_std, color='gray', alpha=0.2, label=f'{reference_key} ± Std')

    plt.xlabel('Experiment')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(which='both', axis='y')

    # Ensure the save directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Save the figure
    plt.savefig(f"{save_dir}/{metric_name}.png")
    plt.close()

def plot_auc_vs_data_ratio(folder, target_name, target_extension, y_metric="sb1to2_AUC", save_dir="user/plots"):
    """
    Load metrics for one folder and plot the selected metric (e.g., sb1to2_AUC, sb2to1_AUC, deb_score)
    vs len_SB2_data / len_SB1_data.

    Args:
        folder (str): The folder to search for files.
        target_name (str): The target file name to search for.
        target_extension (str): The target file extension to search for.
        y_metric (str): The metric to plot on the y-axis. Options: "sb1to2_AUC", "sb2to1_AUC", "deb_score".
        save_dir (str, optional): The directory where the figure should be saved. Defaults to "user/plots".
    """
    # Search for files in the folder
    file_paths = search_files(folder, target_name, target_extension)
    
    # Load data from pickle files
    data_dict = load_pickle_files(file_paths)
    
    # Extract metrics
    y_values = []
    data_ratios = []
    for data in data_dict.values():
        if y_metric in data and "len_SB1_data" in data and "len_SB2_data" in data:
            y_values.append(data[y_metric])
            if data["len_SB1_data"] > 0:  # Avoid division by zero
                data_ratios.append(data["len_SB2_data"] / data["len_SB1_data"])
            else:
                data_ratios.append(0)

    # Plot the selected metric vs len_SB2_data / len_SB1_data
    plt.figure(figsize=(10, 6))
    plt.scatter(data_ratios, y_values, color="blue", label=f"{y_metric} vs Data Ratio")
    plt.xlabel("len_SB2_data / len_SB1_data")
    plt.ylabel(y_metric)
    plt.title(f"{y_metric} vs len_SB2_data / len_SB1_data")
    plt.grid(True)
    plt.legend()
    plt.xscale('log')

    # Ensure the save directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Save the figure
    plt.savefig(f"{save_dir}/{y_metric}_vs_data_ratio.png")
    plt.close()

def main(folders, target_name, target_extension, metric_names, reference_name=None, display_all_points=False, save_dir="user/plots"):
    for metric_name in metric_names:
        metrics = {}
        for name, folder in folders.items():
            file_paths = search_files(folder, target_name, target_extension)
            if not file_paths:
                print(f"Warning: No files found in {folder} for {target_name}{target_extension}.")
                #continue
                exit()
            data_dict = load_pickle_files(file_paths)
            metrics[name] = compute_metrics(data_dict, metric_name)
        plot_metrics(metrics, metric_name, reference_name, display_all_points, save_dir=save_dir)

if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description='Find and plot metrics from pickle files.')
    # parser.add_argument('folders', type=str, help='The root folders to search for files.')
    # parser.add_argument('target_name', type=str, help='The target file name to search for.')
    # parser.add_argument('target_extension', type=str, help='The target file extension to search for.')
    # parser.add_argument('metric_name', type=str, help='The name of the metric to compute and plot.')
    # parser.add_argument('--reference_name', type=str, help='The reference name for plotting.')
    # parser.add_argument('--display_all_points', action='store_true', help='Display all points instead of mean, std, and outliers.')
    # args = parser.parse_args()
    # main(args.folders, args.target_name, args.target_extension, args.metric_name, args.reference_name, args.display_all_points)
    main(folders={
                  #"transitsky1_bdt_before_cut": "/home/users/o/oleksiyu/WORK/skycurtains/workspaces/transit_sky_dev/skyTransit/transitsky_bdteval1",
                  #"transitsky2_bdt_before_cut": "/home/users/o/oleksiyu/WORK/skycurtains/workspaces/transit_sky_dev/skyTransit/transit_sky2",
                  #"transitsky3_bdt_before_cut": "/home/users/o/oleksiyu/WORK/skycurtains/workspaces/transit_sky_dev/skyTransit/transit_sky3",
                  #"transitsky1_bdt_after_cut": "/home/users/o/oleksiyu/WORK/skycurtains/workspaces/transit_sky_dev_cuts/skyTransit/transit_sky1_bdt",
                  #"transitsky2_bdt_after_cut": "/home/users/o/oleksiyu/WORK/skycurtains/workspaces/transit_sky_dev_cuts/skyTransit/transit_sky2_bdt",
                  #"transitsky3_bdt_after_cut": "/home/users/o/oleksiyu/WORK/skycurtains/workspaces/transit_sky_dev_cuts/skyTransit/transit_sky3_bdt",
                  #"transitsky4_bdt_after_cut": "/home/users/o/oleksiyu/WORK/skycurtains/workspaces/transit_sky_dev_cuts/skyTransit/transit_sky4_bdt",
                  #"transitsky5_bdt_after_cut": "/home/users/o/oleksiyu/WORK/skycurtains/workspaces/transit_sky_dev_cuts/skyTransit/transit_sky5_bdt",
                  #"transitsky1v1\ntop3full": "/srv/beegfs/scratch/groups/rodem/skycurtains/results/final/skyTransit/transit_sky_pipe_v1v1",
                  "transitsky1v1\nKDE,5w": "/srv/beegfs/scratch/groups/rodem/skycurtains/results/final/skyTransit/transit_sky_pipe_v1v2_exp1_5w_1v1",
                  "transitsky1v2\nKDE,5w": "/srv/beegfs/scratch/groups/rodem/skycurtains/results/final/skyTransit/main_exp2_5w_1v2",
                  "transitsky2v3\nKDE,5w": "/srv/beegfs/scratch/groups/rodem/skycurtains/results/final/skyTransit/main_5w_v2v3_KDE",
                  "transitsky2v4\nKDE,5w": "/srv/beegfs/scratch/groups/rodem/skycurtains/results/final/skyTransit/main_5w_v2v4_KDE",
                  "transitsky2v5\nKDE,5w": "/srv/beegfs/scratch/groups/rodem/skycurtains/results/final/skyTransit/main_5w_v2v5_KDE",
                  "transitsky2v2\nsampling,5w": "/srv/beegfs/scratch/groups/rodem/skycurtains/results/final/skyTransit/main_5w_v2v2_nokde",
                  "transitsky2v5\nsampling,5w": "/srv/beegfs/scratch/groups/rodem/skycurtains/results/final/skyTransit/main_5w_v2v5_nokde",
                  "transitsky2v6\nsampling,5w": "/srv/beegfs/scratch/groups/rodem/skycurtains/results/final/skyTransit/main_5w_v2v6_nokde",
                  "transitsky2v7\nsampling,5w": "/srv/beegfs/scratch/groups/rodem/skycurtains/results/final/skyTransit/main_5w_v2v7_nokde",
                  "transitsky2v7\nsampling,ntop3full": "/srv/beegfs/scratch/groups/rodem/skycurtains/results/final/skyTransit/transit_sky_pipe_v2v7"
                  #"transitsky2v2\n5w": "/srv/beegfs/scratch/groups/rodem/skycurtains/results/final/skyTransit/main_5w_v2v2_nokde",
                  },
         target_name="results",
         target_extension=".pkl",
         metric_names=["deb_score", "sb1to2_AUC", "sb2to1_AUC", "len_SB1_data", "len_SB2_data"],
         reference_name="min",
         save_dir="user/plots",
         display_all_points=True)

    main(folders={
                  #"transitsky1_bdt_before_cut": "/home/users/o/oleksiyu/WORK/skycurtains/workspaces/transit_sky_dev/skyTransit/transitsky_bdteval1",
                  #"transitsky2_bdt_before_cut": "/home/users/o/oleksiyu/WORK/skycurtains/workspaces/transit_sky_dev/skyTransit/transit_sky2",
                  #"transitsky3_bdt_before_cut": "/home/users/o/oleksiyu/WORK/skycurtains/workspaces/transit_sky_dev/skyTransit/transit_sky3",
                  #"transitsky1_bdt_after_cut": "/home/users/o/oleksiyu/WORK/skycurtains/workspaces/transit_sky_dev_cuts/skyTransit/transit_sky1_bdt",
                  #"transitsky2_bdt_after_cut": "/home/users/o/oleksiyu/WORK/skycurtains/workspaces/transit_sky_dev_cuts/skyTransit/transit_sky2_bdt",
                  #"transitsky3_bdt_after_cut": "/home/users/o/oleksiyu/WORK/skycurtains/workspaces/transit_sky_dev_cuts/skyTransit/transit_sky3_bdt",
                  #"transitsky4_bdt_after_cut": "/home/users/o/oleksiyu/WORK/skycurtains/workspaces/transit_sky_dev_cuts/skyTransit/transit_sky4_bdt",
                  #"transitsky5_bdt_after_cut": "/home/users/o/oleksiyu/WORK/skycurtains/workspaces/transit_sky_dev_cuts/skyTransit/transit_sky5_bdt",
                  "transitsky1v1\ntop3_l0.00_b-32.23": "/srv/beegfs/scratch/groups/rodem/skycurtains/results/final/skyTransit/transit_sky_pipe_v1v1/patches_with_pretrain_cuts/patch_l0.00_b-32.23",
                  "transitsky1v1\ntop3_l18.00_b-32.23": "/srv/beegfs/scratch/groups/rodem/skycurtains/results/final/skyTransit/transit_sky_pipe_v1v1/patches_with_pretrain_cuts/patch_l18.00_b-32.23",
                  "transitsky1v1\ntop3_l342.00_b-32.23": "/srv/beegfs/scratch/groups/rodem/skycurtains/results/final/skyTransit/transit_sky_pipe_v1v1/patches_with_pretrain_cuts/patch_l342.00_b-32.23",
                  #"transitsky2v2\n5w": "/srv/beegfs/scratch/groups/rodem/skycurtains/results/final/skyTransit/main_5w_v2v2_nokde",
                  },
         target_name="results",
         target_extension=".pkl",
         metric_names=["deb_score", "sb1to2_AUC", "sb2to1_AUC", "len_SB1_data", "len_SB2_data"],
         reference_name="min",
         save_dir="user/plots/differnt_patches",
         display_all_points=True)
    
    plot_auc_vs_data_ratio(
        folder="/srv/beegfs/scratch/groups/rodem/skycurtains/results/final/skyTransit/transit_sky_pipe_v1v1/patches_with_pretrain_cuts/",
        target_name="results",
        target_extension=".pkl",
        y_metric="sb1to2_AUC",
        save_dir="user/plots/auc_vs_data_ratio_v1v1_sb1to2",
    )
    plot_auc_vs_data_ratio(
        folder="/srv/beegfs/scratch/groups/rodem/skycurtains/results/final/skyTransit/transit_sky_pipe_v1v1/patches_with_pretrain_cuts/",
        target_name="results",
        target_extension=".pkl",
        y_metric="sb2to1_AUC",
        save_dir="user/plots/auc_vs_data_ratio_v1v1_sb2to1",
    )
    plot_auc_vs_data_ratio(
        folder="/srv/beegfs/scratch/groups/rodem/skycurtains/results/final/skyTransit/transit_sky_pipe_v1v1/patches_with_pretrain_cuts/",
        target_name="results",
        target_extension=".pkl",
        y_metric="deb_score",
        save_dir="user/plots/auc_vs_data_ratio_v1v1_debscore",
    )
    
    plot_auc_vs_data_ratio(
        folder="/srv/beegfs/scratch/groups/rodem/skycurtains/results/final/skyTransit/main_5w_v2v5_nokde/",
        target_name="results",
        target_extension=".pkl",
        y_metric="sb1to2_AUC",
        save_dir="user/plots/auc_vs_data_ratio_v2v5_nokde_sb1to2",
    )
    plot_auc_vs_data_ratio(
        folder="/srv/beegfs/scratch/groups/rodem/skycurtains/results/final/skyTransit/main_5w_v2v5_nokde/",
        target_name="results",
        target_extension=".pkl",
        y_metric="sb2to1_AUC",
        save_dir="user/plots/auc_vs_data_ratio_v2v5_nokde_sb2to1",
    )
    plot_auc_vs_data_ratio(
        folder="/srv/beegfs/scratch/groups/rodem/skycurtains/results/final/skyTransit/main_5w_v2v5_nokde/",
        target_name="results",
        target_extension=".pkl",
        y_metric="deb_score",
        save_dir="user/plots/auc_vs_data_ratio_v2v5_nokde_debscore",
    )