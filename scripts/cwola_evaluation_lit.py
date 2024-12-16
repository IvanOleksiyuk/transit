# This script  just build SIC curves and simmilar plots.

import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")
import logging
import hydra
from omegaconf import DictConfig
import os
from pathlib import Path 
import pandas as pd
log = logging.getLogger(__name__)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import auc, roc_curve

def main() -> None:
    n_dope_per_run=[0, 50, 100, 333, 500, 667, 1000, 3000, 4000, 6000, 8000]
    runs = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]
    seeds=[0, 1, 2, 3, 4]
    run_number=8
    #CURTAINS
    # for run_number in runs:
    #     for seed in seeds:
    #         run_dir = Path(f"/home/users/o/oleksiyu/WORK/hyperproject/lit/curtains/run_{run_number}/seed_"+str(seed)+"/")
    #         os.makedirs(run_dir, exist_ok=True)
    #         file_path = f"/srv/beegfs/scratch/groups/rodem/forcomparison/curtains/run_{run_number}/bdt/standard/seed_"+str(seed)+"/fulldata_mean.h5"
    #         data = {}
    #         with pd.HDFStore(file_path, "r") as store:
    #                 # Iterate over all the keys (dataset names) in the file
    #                 for key in store:
    #                     # Read each dataset into a pandas DataFrame and store in the dictionary
    #                     data[key[1:]] = store[key]
                        
    #         print(data)
    #         df = data["df"]
    #         true_data = pd.concat([df[df["CWoLa Label"] == 1], df[df["CWoLa Label"] == -2]])

    #         plot_ROC(true_data["preds"], 
    #                 np.abs(true_data["Truth"]), 
    #                 save_path=run_dir / "ROC")
    #         do_SI_v_rej(true_data["preds"], 
    #                     np.abs(true_data["Truth"]), 
    #                     save_path=run_dir / "SI_v_rej")
    #         do_rejection_v_TPR(true_data["preds"], 
    #                             np.abs(true_data["Truth"]),
    #                             save_path=run_dir / "rejection_v_TPR")
    # exit()
    #OLIWS Idealised Supervised
    dopings = [0, 50, 100, 333, 500, 667, 1000, 3000]
    for doping in dopings:
        for method in ["supervised", "idealised", "standard"]:
            for seed in seeds:
                run_dir = Path(f"/home/users/o/oleksiyu/WORK/hyperproject/lit/radot/dope_{doping}/"+method+"/seed_"+str(seed)+"/")
                os.makedirs(run_dir, exist_ok=True)
                file_path = f"/srv/beegfs/scratch/groups/rodem/oliws/sbwidthimpure/window_2900_3300__3700_4100/dope_{doping}/"+method+"/seed_"+str(seed)+"/cwola_outputs.h5"
                data = {}
                with pd.HDFStore(file_path, "r") as store:
                        # Iterate over all the keys (dataset names) in the file
                        for key in store:
                            # Read each dataset into a pandas DataFrame and store in the dictionary
                            data[key[1:]] = store[key]
                            
                print(data)
                df = data["df"]
                if method == "supervised":
                    true_data = df
                if method == "idealised":
                    true_data = df
                if method == "standard":
                    true_data = df[df["CWoLa"] == 1]
                    
                plot_ROC(df["preds"][df["is_signal"] != 1], 
                        df["CWoLa"][df["is_signal"] != 1], 
                        save_path=run_dir / "closure_ROC")
                plot_ROC(true_data["preds"], 
                        true_data["is_signal"], 
                        save_path=run_dir / "ROC")
                do_SI_v_rej(true_data["preds"], 
                            true_data["is_signal"], 
                            save_path=run_dir / "SI_v_rej")
                do_rejection_v_TPR(true_data["preds"], 
                                    true_data["is_signal"],
                                    save_path=run_dir / "rejection_v_TPR")
                
                if method == "standard":
                    templatesr = df[df["CWoLa"] == 0]
                    datasr_bkg = true_data[true_data["is_signal"] == 0]
                    preds = pd.concat([templatesr["preds"], datasr_bkg["preds"]])
                    labels = np.concatenate([np.zeros(len(templatesr)), np.ones(len(datasr_bkg))])
                    plot_ROC(preds, 
                        labels, 
                        save_path=run_dir / "ROC_closure",
                        title="ROC closure")
    
def calc_TPR_FPR(scores, true_labels):
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = np.array(scores)[sorted_indices]
    sorted_labels = np.array(true_labels)[sorted_indices]

    # Initialize variables to calculate TPR and FPR
    TP = 0
    FP = 0
    FN = sum(true_labels)  # All positives initially
    TN = len(true_labels) - FN  # All negatives initially

    tpr_list = []
    fpr_list = []

    # Calculate TPR and FPR for each threshold
    for score, label in zip(sorted_scores, sorted_labels):
        if label == 1:
            TP += 1
            FN -= 1
        else:
            FP += 1
            TN -= 1
        
        if TP + FN == 0:
            TPR = 0
        else:
            TPR = TP / (TP + FN)
        if FP + TN == 0:
            FPR = 0
        else:
            FPR = FP / (FP + TN)
        
        tpr_list.append(TPR)
        fpr_list.append(FPR)

    # Add (0,0) and (1,1) points to the ROC curve
    tpr_list.insert(0, 0.0)
    fpr_list.insert(0, 0.0)
    tpr_list.append(1.0)
    fpr_list.append(1.0)
    return tpr_list, fpr_list

def plot_ROC(scores, true_labels, save_path, title="ROC", make_plot=True, save_npy=True):
    fpr_list, tpr_list, _ = roc_curve(true_labels, scores)
    # Plot the ROC curve
    if make_plot:
        plt.figure()
        auc_score = auc(fpr_list, tpr_list)
        plt.plot(fpr_list, tpr_list,  label=f"Template AUC: {auc_score:.3f}")
        plt.legend()
        plt.title(title)
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.grid(which='major')
        plt.gca().set_aspect('equal')
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.savefig(str(save_path)+".png", bbox_inches='tight', dpi=300)
    if save_npy:
        np.save(str(save_path)+".npy", np.array([fpr_list, tpr_list]))

def do_SI_v_rej(scores, true_labels, save_path, title="SI_v_rej", make_plot=True, save_npy=True):
    fpr_list, tpr_list, _ = roc_curve(true_labels, scores)
    SI = np.array(tpr_list) / np.sqrt(np.array(fpr_list))
    rej = 1 / np.array(fpr_list)
    # Plot the curve
    if make_plot:
        plt.figure()
        plt.errorbar(rej, SI)
        plt.title(title)
        plt.grid()
        plt.xscale('log')
        plt.xlabel('Rejection')
        plt.ylabel('SI')
        plt.savefig(str(save_path)+".png", bbox_inches='tight', dpi=300)
    if save_npy:
        np.save(str(save_path)+".npy", np.array([rej, SI]))

def do_rejection_v_TPR(scores, true_labels, save_path, title="rej_v_TPR", make_plot=True, save_npy=True):
    fpr_list, tpr_list, _ = roc_curve(true_labels, scores)
    rej = 1 / np.array(fpr_list)
    # Plot the curve
    if make_plot:
        plt.figure()
        plt.plot(tpr_list, rej)
        plt.title(title)
        plt.grid()
        plt.yscale('log')
        plt.xlabel('Rejection')
        plt.ylabel('SI')
        plt.savefig(str(save_path)+".png", bbox_inches='tight', dpi=300)
    if save_npy:
        np.save(str(save_path)+".npy", np.array([tpr_list, rej]))

def mass_sculpting(masses, scores, true_labels, save_path, rej_cuts = [0.5, 0.9, 0.95, 0.99], title="mass_sculpting", density=False, bins=100):
    sorted_indices = np.argsort(scores)[::-1]
    sorted_masses = np.array(masses)[sorted_indices]
    sorted_scores = np.array(scores)[sorted_indices]
    sorted_true_labels = np.array(true_labels)[sorted_indices]
    
    plt.figure()
    _, bins, _ = plt.hist(sorted_masses, bins=bins, alpha=1, label="original", color='black', histtype='step', density=density)
    plt.hist(sorted_masses[sorted_true_labels==1], bins=bins, alpha=1, label="signal", color='gray', histtype='step', density=density)

    plt.yscale('log')
    plt.title(title)
    plt.xlabel('mass')
    plt.ylabel('counts')
    
    colormap = cm.get_cmap('jet', len(rej_cuts)+2)
    for i, rej_cut in enumerate(rej_cuts):
        cut_index = -int(len(sorted_scores) * rej_cut)
        plt.hist(sorted_masses[:cut_index], bins=bins, label=f"rejection {rej_cut}", color=colormap(i+1), histtype='step', density=density)
    plt.legend()
    plt.grid()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

def plot_cuts_mass_hists():
    pass

if __name__ == "__main__":
    main()
