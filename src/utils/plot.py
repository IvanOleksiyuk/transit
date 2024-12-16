import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_dataframe(df, plot_dir, plot_name):
    os.makedirs(plot_dir, exist_ok=True)
    sns.pairplot(df)
    plt.savefig(os.path.join(plot_dir, f'{plot_name}.png'))
    plt.close()