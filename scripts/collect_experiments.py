import os
import pickle
import pandas as pd

def collect_metrics(base_dir, selected_metrics=None, sort_by=None):
    data = {}
    
    # Traverse the directories to find the results.pickle files
    for experiment in os.listdir(base_dir):
        exp_dir = os.path.join(base_dir, experiment)
        if os.path.isdir(exp_dir):
            plots_dir = os.path.join(exp_dir, 'plots')
            results_file = os.path.join(plots_dir, 'results.pkl')
            
            if os.path.exists(results_file):
                with open(results_file, 'rb') as f:
                    metrics = pickle.load(f)
                    
                    # Filter metrics if selected_metrics is provided
                    if selected_metrics:
                        metrics = {key: metrics[key] for key in selected_metrics if key in metrics}
                    
                    data[experiment] = metrics
    
    # Create a DataFrame from the collected data
    df = pd.DataFrame.from_dict(data, orient='index')
    
    # Sort the DataFrame if sort_by is provided
    if sort_by and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=False)
    
    return df

if __name__ == "__main__":
    base_dir = '/home/users/o/oleksiyu/WORK/hyperproject/workspaces/LHCO'  # Change this to your base directory
    
    # Optional parameters
    selected_metrics = ["max_abs_pearson",  "min_abs_pearson",  "mean_abs_pearson", "hilbert_schmidt", "DisCo", "template_max_lazy_ROCAUC"]  # Change to None if you want all metrics
    sort_by = "template_max_lazy_ROCAUC"  # Change to None if you don't want to sort
    
    metrics_df = collect_metrics(base_dir, selected_metrics, sort_by)
    print(metrics_df)
    
    # Save to CSV for further analysis if needed
    metrics_df.to_csv('collected_metrics.csv', index=True)