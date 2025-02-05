import pandas as pd
import numpy as np

def generate_dataset(features, num_samples, output_file, random_seed=None, oversampling=None):
    """
    Generates a pandas DataFrame with normally distributed random values.
    
    Parameters:
    features (dict): A dictionary where keys are feature names and values are tuples (mean, std_dev)
    num_samples (int): Number of samples to generate
    output_file (str): Path to save the dataset as an HDF5 file
    random_seed (int, optional): Random seed for reproducibility
    oversampling (float, optional): Oversampling factor, if provided, the dataset is resampled with replacement
    
    Returns:
    pd.DataFrame: A DataFrame containing the generated dataset
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    data = {}
    
    for feature, (mean, std_dev) in features.items():
        data[feature] = np.random.normal(loc=mean, scale=std_dev, size=num_samples)
    
    df = pd.DataFrame(data)
    
    if oversampling is not None and oversampling > 0:
        df = df.sample(n=int(num_samples * oversampling), replace=True, random_state=random_seed)
    
    df.to_hdf(output_file, key="template", mode="w")
    return df

# Example usage
if __name__ == "__main__":
    output_file = "/home/users/o/oleksiyu/WORK/hyperproject/user/mock_data/tem200Kx7_seed1.h5"
    feature_info = {
        "m_j1": (0, 1),   
        "del_m": (0, 1),   
        "del_R": (0, 1),
        "tau21_j1": (0, 1),
        "tau21_j2": (0, 1),
        "m_jj": (0, 1),
    }
    num_samples = 200000
    random_seed = 1
    oversampling = 7  # Example: Increase dataset size by a factor of 2
    dataset = generate_dataset(feature_info, num_samples, output_file, random_seed, oversampling)
    print(dataset.head())
