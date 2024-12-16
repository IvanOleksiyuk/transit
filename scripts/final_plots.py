import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import auc, roc_curve
config = {}
config["save_path"] = "/home/users/o/oleksiyu/WORK/hyperproject/plots/final/"
config["methods"] = {}
config["methods"]["supervised"] = {}
config["methods"]["supervised"]["scores_file"] = "/srv/beegfs/scratch/groups/rodem/LHCO/low_class_outs_final/supervised/n_sig_3000/run0/result/scores_classifier.csv"

#TODO: Combine scores from several runs

def main(cfg):
    ROCs = {}
    for key, method in cfg["methods"].items():
        df = load_scores_file(method["scores_file"])
        inspect_df(df)
        fpr, tpr, _ = roc_curve(df["is_signal"], df["outputs"]) # IS NOT CORRECT IN CASE TEMPLATE IS PRESENT CHANGE IT THEN
        ROCs[key] = (fpr, tpr)
    plot_SI_v_rej(ROCs, config["save_path"]+"SI_v_rej.png", xlim=[1, 20000])

def inspect_df(df):
    print("======Inspecting DataFrame======")
    print("len:", len(df))
    print("columns:", df.columns)
    #print(df.head())
    print("is_signal:", df["is_signal"].unique())
    print("is_signal==0:", sum(df["is_signal"]==0))
    print("is_signal==1:", sum(df["is_signal"]==1))
    
    print("sot_label:", df["sot_label"].unique())
    print("sot_label==0:", sum(df["sot_label"]==0))
    print("sot_label==1:", sum(df["sot_label"]==1))
    
    print("labels:", df["labels"].unique())
    print("labels==0:", sum(df["labels"]==0))
    print("labels==1:", sum(df["labels"]==1))
    
    print("min_output:", min(df["outputs"]))
    print("min_mjj:", min(df["mjj"]))
    print("max_mjj:", max(df["mjj"]))
    print("================================")

def load_scores_file(scores_file):
    print("Loading scores file:", scores_file)
    return pd.read_csv(scores_file)

def plot_SI_v_rej(ROCs, save_path, title="SI_v_rej", xlim=None):
    # Plot the curve
    plt.figure()
    for key, roc in ROCs.items():
        fpr_list, tpr_list = roc
        SI = np.array(tpr_list) / np.sqrt(np.array(fpr_list))
        rej = 1 / np.array(fpr_list)
        plt.plot(rej, SI)
    plt.title(title)
    plt.xscale('log')
    plt.xlabel('Rejection')
    plt.ylabel('SI')
    plt.xlim(xlim)
    plt.grid(which='major')
    plt.savefig(str(save_path)+".png", bbox_inches='tight', dpi=300)
  


if __name__=="__main__":
    main(config)