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
import time
# TODO pyroot utils will remove the need for ../configs
@hydra.main(
    version_base=None, config_path=str('../config'), config_name="transit_reco_cons0.01_smls0.001_adv3_LHCO_CURTAINS1024b"
)
def main(cfg: DictConfig) -> None:
    print("Starting CWoLa evaluation")
    start_time = time.time()
    cwola_eval_path = Path(cfg.cwola_eval_path)
    for seed in cfg.seeds:
        plot_path = cwola_eval_path / ("seed_" + str(seed))
        os.makedirs(plot_path, exist_ok=True)
        file_path=cfg.cwola_path+cfg.cwola_subfolders+f"standard/seed_{seed}/"+"cwola_outputs.h5"
        file_path_extra_bkg=cfg.cwola_path+cfg.cwola_subfolders+f"standard/seed_{seed}/"+"cwola_outputs_extra_bkg.h5"
        file_path_extra_sig=cfg.cwola_path+cfg.cwola_subfolders+f"standard/seed_{seed}/"+"cwola_outputs_extra_sig.h5"
        data = {}
        with pd.HDFStore(file_path, "r") as store:
                # Iterate over all the keys (dataset names) in the file
                for key in store:
                    # Read each dataset into a pandas DataFrame and store in the dictionary
                    data[key[1:]] = store[key]

        data_extra_sig = {}
        with pd.HDFStore(file_path_extra_sig, "r") as store:
                # Iterate over all the keys (dataset names) in the file
                for key in store:
                    # Read each dataset into a pandas DataFrame and store in the dictionary
                    data_extra_sig[key[1:]] = store[key]
                            
        data_extra_bkg = {}
        with pd.HDFStore(file_path_extra_bkg, "r") as store:
                # Iterate over all the keys (dataset names) in the file
                for key in store:
                    # Read each dataset into a pandas DataFrame and store in the dictionary
                    data_extra_bkg[key[1:]] = store[key]
        
        print(data)
        datasr = data["df"][data["df"]["CWoLa"] == 1]
        preds = pd.concat([datasr["preds"], data_extra_sig["df"]["preds"]])
        labels = pd.concat([datasr["is_signal"], data_extra_sig["df"]["is_signal"]])
        plot_ROC(preds, 
                labels, 
                save_path=plot_path / "ROC",
                title="ROC")
        plot_SI_v_rej(preds, 
                    labels, 
                    save_path=plot_path / "SI_v_rej")
        do_rejection_v_TPR(preds, 
                        labels, 
                        save_path=plot_path / "rejection_v_TPR")
        
        templatesr = data["df"][data["df"]["CWoLa"] == 0]
        datasr_bkg = datasr[datasr["is_signal"] == 0]
        preds = pd.concat([templatesr["preds"], datasr_bkg["preds"]])
        labels = np.concatenate([np.zeros(len(templatesr)), np.ones(len(datasr_bkg))])
        plot_ROC(preds, 
                labels, 
                save_path=plot_path / "ROC_closure",
                title="ROC closure")
        
        # do_mass_sculpting(data["df"]["m_jj"], 
        #                   data["df"]["preds"], 
        #                   data["df"]["is_signal"], 
        #                   save_path=plot_path / "mass_sculpting.png")
        # do_mass_sculpting(data["df"]["m_jj"], 
        #                   data["df"]["preds"], 
        #                   data["df"]["is_signal"], 
        #                   save_path=plot_path / "mass_sculpting_den2.png",
        #                   density=True,
        #                   rej_cuts = [0.9, 0.99],
        #                   bins=30)
        do_mass_sculpting(data["df"]["m_jj"], 
                          data["df"]["preds"], 
                          data["df"]["is_signal"], 
                          save_path=plot_path / "mass_sculpting_den2_bkg.png",
                          density=True,
                          filter_bkg=True,
                          rej_cuts = [0.9, 0.99],
                          draw_signal=False,
                          bins=30)
        # do_mass_sculpting(pd.concat([datasr["m_jj"], data_extra_bkg["df"]["m_jj"]]), 
        #             pd.concat([datasr["preds"], data_extra_bkg["df"]["preds"]]), 
        #             pd.concat([datasr["is_signal"], data_extra_bkg["df"]["m_jj"]*0]), 
        #             save_path=plot_path / "mass_sculpting_density.png", 
        #             density=True, 
        #             rej_cuts = [0.9, 0.99], bins=100)
        # do_mass_sculpting(pd.concat([datasr["m_jj"], data_extra_bkg["df"]["m_jj"]]), 
        #             pd.concat([datasr["preds"], data_extra_bkg["df"]["preds"]]), 
        #             pd.concat([datasr["is_signal"], data_extra_bkg["df"]["m_jj"]*0]), 
        #             save_path=plot_path / "mass_sculpting_density_bkg_only.png", 
        #             density=True, 
        #             filter_bkg=True, 
        #             rej_cuts = [0.9, 0.99], 
        #             bins=100)
    print(f"Finished in {(time.time()-start_time)/60} minutes")

    
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

def plot_SI_v_rej(scores, true_labels, save_path, title="SI_v_rej", make_plot=True, save_npy=True):
    fpr_list, tpr_list, _ = roc_curve(true_labels, scores)
    SI = np.array(tpr_list) / np.sqrt(np.array(fpr_list))
    rej = 1 / np.array(fpr_list)
    # Plot the curve
    if make_plot:
        plt.figure()
        plt.plot(rej, SI)
        plt.title(title)
        plt.grid()
        plt.xscale('log')
        plt.xlabel('Rejection')
        plt.ylabel('SI')
        plt.grid(which='major')
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
        plt.grid(which='major')
        plt.savefig(str(save_path)+".png", bbox_inches='tight', dpi=300)
    if save_npy:
        np.save(str(save_path)+".npy", np.array([tpr_list, rej]))

def do_mass_sculpting(masses, scores, true_labels, save_path, rej_cuts = [0.5, 0.9, 0.95, 0.99], title="mass_sculpting", density=False, bins=100, filter_bkg=False, draw_signal=True):
    sorted_indices = np.argsort(scores)[::-1]
    sorted_masses = np.array(masses)[sorted_indices]
    sorted_scores = np.array(scores)[sorted_indices]
    sorted_true_labels = np.array(true_labels)[sorted_indices]
    
    if filter_bkg:
        sorted_masses = sorted_masses[sorted_true_labels==0]
        sorted_scores = sorted_scores[sorted_true_labels==0]
        sorted_true_labels = sorted_true_labels[sorted_true_labels==0]
    
    plt.figure()
    _, bins, _ = plt.hist(sorted_masses, bins=bins, alpha=1, label="original", color='black', histtype='step', density=density)
    if draw_signal:
        plt.hist(sorted_masses[sorted_true_labels==1], bins=bins, alpha=1, label="signal", color='gray', histtype='step', density=density)

    plt.yscale('log')
    plt.title(title)
    plt.xlabel('mass')
    plt.ylabel('counts')
    
    try:
        colormap = cm.get_cmap('jet', len(rej_cuts)+2)
    except:
        colormap = plt.get_cmap('jet', len(rej_cuts)+2)

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

# DEBS CODE
def get_curtains_classifier_artifacts(job_path, template=False):
    num_classifiers = list(job_path.glob("seed_*"))
    fprs, tprs = [], []
    for seed in num_classifiers:
        df = pd.read_hdf(seed/"fulldata_mean.h5", "df")
        df.sort_values(by=["m_j1","del_m"])
        truth = df["Truth"].values
        preds = df["preds"].values
        if template is True:
            print("Template mode")
            template_mask = truth == -1
            template_preds = preds[template_mask]
            bg_mask = truth == 0
            bg_preds = preds[bg_mask]
            ret_preds = np.concatenate([template_preds, bg_preds])
            ret_truth = np.concatenate([np.zeros(len(template_preds)), np.ones(len(bg_preds))])
            fpr, tpr, _ = roc_curve(ret_truth, ret_preds)
            fprs.append(fpr)
            tprs.append(tpr)
        else:
            bg_mask = truth == 0.
            bg_preds = preds[bg_mask]
            signal_mask = (truth == 1.) | (truth == -2.)
            signal_preds = preds[signal_mask]
            ret_preds = np.concatenate([signal_preds, bg_preds])
            ret_truth = np.concatenate([np.ones(len(signal_preds)), truth[bg_mask]])
            fpr, tpr, _ = roc_curve(ret_truth, ret_preds)
            fprs.append(fpr)
            tprs.append(tpr)
    try:
        print(f"Total signal events: {len(signal_preds)}")
        print(f"Total background events: {len(bg_preds)}")
    except Exception as e:
        pass
    return fprs, tprs

def oliws_classifier_artifacts(job_path, template=False):
    num_classifiers = list(job_path.glob("seed_*"))
    fprs, tprs = [], []
    for seed in num_classifiers:
        df = pd.read_hdf(seed/"cwola_outputs.h5", "df")
        df.sort_values(by=["m_j1","del_m"])
        truth = df["is_signal"].values
        preds = df["preds"].values
        if template:
            template_mask = truth == -1
            template_preds = preds[template_mask]
            bg_mask = truth == 0
            bg_preds = preds[bg_mask]
            ret_preds = np.concatenate([template_preds, bg_preds])
            ret_truth = np.concatenate([np.zeros(len(template_preds)), np.ones(len(bg_preds))])
            fpr, tpr, _ = roc_curve(ret_truth, ret_preds)
            fprs.append(fpr)
            tprs.append(tpr)
        else:
            data_mask = truth >= 0
            data_preds = preds[data_mask]
            data_truth = truth[data_mask]
            fpr, tpr, _ = roc_curve(data_truth, data_preds)
            fprs.append(fpr)
            tprs.append(tpr)
            signal_preds = len(preds[truth==1])
            bg_preds = len(preds[truth==0])
    try:
        print(f"Total signal events: {len(signal_preds)}")
        print(f"Total background events: {len(bg_preds)}")
    except Exception as e:
        pass
    return fprs, tprs