import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

from typing import Union
import logging
import hydra
import time
from hydra.core.global_hydra import GlobalHydra

from pathlib import Path
import os
from transit.src.utils.hydra_utils import reload_original_config

import pandas as pd
import transit.src.utils.plotting as pltt
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from omegaconf import OmegaConf
import torch
from scipy.stats import pearsonr, spearmanr, kendalltau
import pickle
from transit.src.models.distance_correlation import DistanceCorrelation
from transit.src.utils.hsic import HSIC_torch
from sklearn.model_selection import train_test_split
import copy
import wandb

log = logging.getLogger(__name__)

# Optional modules 
try:
    import dcor
except ImportError:
    dcor = None
    log.warning("dcor module is not available. Distance correlation calculations will be skipped. (its non-essential only for diagnostics)")

def to_np(inpt: Union[torch.Tensor, tuple]) -> np.ndarray:
    """More consicse way of doing all the necc steps to convert a pytorch
    tensor to numpy array.

    - Includes gradient deletion, and device migration
    """
    if isinstance(inpt, (tuple, list)):
        return type(inpt)(to_np(x) for x in inpt)
    if inpt.dtype == torch.bfloat16:  # Numpy conversions don't support bfloat16s
        inpt = inpt.half()
    return inpt.detach().cpu().numpy().reshape(inpt.shape)

def check_data_loaded(names_list, data):
    missing = []
    for name in names_list:
        if name not in data:
            return missing.append(name)
    return missing

@rank_zero_only
def reload_original_config(cfg: OmegaConf, get_best: bool = False) -> OmegaConf:
    """Replaces the cfg with the one stored at the checkpoint location.

    Will also set the chkpt_dir to the latest version of the last or
    best checkpoint
    """

    # Load the original config found in the the file directory
    orig_cfg = OmegaConf.load(Path(cfg.paths.full_path+"/full_config.yaml"))

    # Get the latest updated checkpoint with the prefix last or best
    flag = "best" if get_best else "last"
    orig_cfg.ckpt_path = str(
        sorted(Path(cfg.paths.full_path).glob(f"checkpoints/{flag}*.ckpt"), key=os.path.getmtime)[-1]
    )

    # Set the wandb logger to attempt to resume the job
    if hasattr(orig_cfg, "loggers"):
        if hasattr(orig_cfg.loggers, "wandb"):
            orig_cfg.loggers.wandb.resume = True

    return orig_cfg

@hydra.main(
    version_base=None, config_path=str('../conf'), config_name="evaluate"
)
def main(cfg):
    log.info("Starting evaluation")
    results = {}
    # Get two dataframes to compare
    Path(cfg.general.run_dir+"/plots/").mkdir(parents=True, exist_ok=True) 
    
    with open(cfg.general.run_dir+"/template/wandb_id.txt", "r") as f:
        run_id = f.read().strip()
    try:
        wandb.init(id=run_id, resume="allow")
    except:
        print("Could not resume wandb run")
    
    # Plot the transport from SB1 to SB2
    data = {} # a dictionary to store the data
    for key in cfg.step_evaluate.data:
        if "file" in key:
            data[key] = pd.read_hdf(cfg.step_evaluate.data[key])
            log.info(f"Loaded data {key} with n={len(data[key])}, and vars {data[key].columns.tolist()}")
        else:
            data[key] = hydra.utils.instantiate(cfg.step_evaluate.data[key])
            if cfg.step_evaluate.data[key]["_target_"]=="src.data.gdr3_dataclasses.GDR3DataModuleTop":
                data[key].setup(stage="test")
                data[key] = pd.DataFrame(data[key].test_dataset[:][0], columns=data[key].get_features_all())
            if cfg.step_evaluate.data[key]["_target_"]=="transit.src.data.data.InMemoryDataFrameDict":
                data[key] = data[key].data["data"]
            if cfg.step_evaluate.data[key]["_target_"]=="transit.src.data.data.SimpleDataModule":
                data[key].setup(stage="test")
                data[key] = pd.concat([data[key].test_data.data["data"], data[key].test_data.data["mass_paired"]], axis=1)
            if cfg.step_evaluate.data[key]["_target_"]=="src.data.data.SimpleDataModule":
                data[key].setup(stage="test")
                data[key] = pd.concat([data[key].test_data.data["data"], data[key].test_data.data["cond"]], axis=1)
            if cfg.step_evaluate.data[key]["_target_"]=="src.data.data.InMemoryDataFrameDict":
                data[key] = data[key].data["data"]
            log.info(f"Loaded data {key} with n={len(data[key])}, and vars {data[key].columns.tolist()}")    
    log.info("Data loaded")
    
    variables = data["original_data"].columns.tolist()

    # Plot the contour plot for the generated template on SR
    if getattr(cfg.step_evaluate, "plot_contour_SR", True):
        if cfg.step_evaluate.debug_eval:
            plot_mode="diagnose"
        else:
            plot_mode=""
        log.info("Starting contour plot "+plot_mode)
        time_start = time.time()
        pltt.plot_feature_spread(
            data["target_data"][variables].to_numpy(),
            data["template_file"][variables].to_numpy(),
            original_data = data["original_data"][variables].to_numpy(),
            feature_nms = variables,
            save_dir=Path(cfg.general.run_dir)/ "plots/",
            plot_mode=plot_mode,
            do_2d_hist_instead_of_contour=cfg.step_evaluate.do_2d_hist_instead_of_contour,
            x_bounds=cfg.step_evaluate.x_bounds or None,
            tag = ["SB1, SB2", "SR"],
            save_name="SB2nSB1_to_SR")
        log.info("contour plot is done, in "+str(time.time()-time_start)+" seconds")
        
    if getattr(cfg.step_evaluate, "plot_contour_SB1toSB2transport", True):
        if check_data_loaded(["original_for_SB1_data", "SB1_gen_file", "original_for_SB2_data", "target_for_SB1_data", "target_for_SB2_data"], data)!=[]:
            print("Missing data: ", check_data_loaded(["original_for_SB1_data", "SB1_gen_file", "original_for_SB2_data", "target_for_SB1_data", "target_for_SB2_data"], data))
        else:
            pltt.plot_feature_spread(
                data["target_for_SB1_data"][variables].to_numpy(),
                data["SB1_gen_file"][variables].to_numpy(),
                original_data = data["original_for_SB1_data"][variables].to_numpy(),
                feature_nms = variables,
                save_dir=Path(cfg.general.run_dir+"/plots/"),
                plot_mode=plot_mode,
                do_2d_hist_instead_of_contour=cfg.step_evaluate.do_2d_hist_instead_of_contour,
                x_bounds=cfg.step_evaluate.x_bounds or None,
                tag = ["SB2", "SB1"],
                save_name="SB2_to_SB1")
            log.info("Plotted SB2 to SB1 transport")
            pltt.plot_feature_spread(
                data["target_for_SB2_data"][variables].to_numpy(),
                data["SB2_gen_file"][variables].to_numpy(),
                original_data = data["original_for_SB2_data"][variables].to_numpy(),
                feature_nms = variables,
                save_dir=Path(cfg.general.run_dir+"/plots/"),
                plot_mode=plot_mode,
                do_2d_hist_instead_of_contour=cfg.step_evaluate.do_2d_hist_instead_of_contour,
                x_bounds=cfg.step_evaluate.x_bounds or None,
                tag = ["SB1", "SB2"],
                save_name="SB1_to_SB2")
            log.info("Plotted SB1 to SB2 transport")
    
    if getattr(cfg.step_evaluate, "closure_SKYclassifier_SBtoSB_transport", False):
        from src.model.denseclassifier import run_classifier_folds
        
        SB1_data = data["target_for_SB1_data"].to_numpy()[:, :-1]
        SB1_gen = data["SB1_gen_file"].to_numpy()[:, :-1]
        SB2_data = data["target_for_SB2_data"].to_numpy()[:, :-1]
        SB2_gen = data["SB2_gen_file"].to_numpy()[:, :-1]
        
        # Limit the number of events to train the classifier on faster
        n_max=cfg.step_evaluate.get("n_max_class_train", 10000)
        if n_max is not None and n_max>0:
            if len(SB1_data)>n_max:
                SB1_data = SB1_data[:n_max]
            if len(SB1_gen)>n_max:
                SB1_gen = SB1_gen[:n_max]
            if len(SB2_data)>n_max:
                SB2_data = SB2_data[:n_max]
            if len(SB2_gen)>n_max:
                SB2_gen = SB2_gen[:n_max]
            
        log.info("Starting classifier train/eval")
        auc_score_1to2, threshold, data_preds = run_classifier_folds(
            SB2_data, 
            SB2_gen,
            save_dir=Path(cfg.general.run_dir),
            tag=f"sb1to2",
            return_threshold=False,  # if key == "sb12r" else False,
        )
        log.info("Finish classifier train/eval")
        log.info(f"SB1toSB2 vs SB2 AUC={auc_score_1to2}")
        wandb.log({"evaluation/sb1to2_AUC": auc_score_1to2})
        results["sb1to2_AUC"] = auc_score_1to2
        
        log.info("Starting classifier train/eval")
        auc_score_2to1, threshold, data_preds = run_classifier_folds(
            SB1_data, 
            SB1_gen,
            save_dir=Path(cfg.general.run_dir),
            tag=f"sb2to1",
            return_threshold=False,  # if key == "sb12r" else False,
        )
        log.info("Finish classifier train/eval")
        log.info(f"SB2toSB1 vs SB2 AUC={auc_score_2to1}")
        wandb.log({"evaluation/sb2to1_AUC": auc_score_2to1})
        results["sb2to1_AUC"] = auc_score_2to1
        
    if getattr(cfg.step_evaluate, "closure_SKYclassifier_SBtoSR", False):
        from src.model.denseclassifier import run_classifier_folds
        
        SR_data = data["target_data"].to_numpy()[:, :-1]
        SB1toSR_gen = data["SB1toSR_gen_file"].to_numpy()[:, :-1]
        SB2toSR_gen = data["SB2toSR_gen_file"].to_numpy()[:, :-1]
        
        # Limit the number of events to train the classifier on faster
        n_max=cfg.step_evaluate.get("n_max_class_train", 10000)
        if n_max is not None and n_max>0:
            if len(SR_data)>n_max:
                SR_data = SR_data[:n_max]
            if len(SB1toSR_gen)>n_max:
                SB1toSR_gen = SB1toSR_gen[:n_max]
            if len(SB2toSR_gen)>n_max:
                SB2toSR_gen = SB2toSR_gen[:n_max]
            
        log.info("Starting classifier train/eval")
        auc_score_SB1toSR, threshold, data_preds = run_classifier_folds(
            SB1toSR_gen, 
            SR_data,
            save_dir=Path(cfg.general.run_dir),
            tag=f"sb1to2",
            return_threshold=False,  # if key == "sb12r" else False,
        )
        log.info("Finish classifier train/eval")
        
        log.info(f"SB1toSR vs SR AUC={auc_score_SB1toSR}")
        wandb.log({"evaluation/auc_score_SB1toSR_AUC": auc_score_SB1toSR})
        results["sb1toSR_AUC"] = auc_score_SB1toSR
        
        log.info("Starting classifier train/eval")
        auc_score_SB2toSR, threshold, data_preds = run_classifier_folds(
            SB2toSR_gen, 
            SR_data,
            save_dir=Path(cfg.general.run_dir),
            tag=f"sb2to1",
            return_threshold=False,  # if key == "sb12r" else False,
        )
        log.info("Finish classifier train/eval")
        log.info(f"SB2toSR vs SR AUC={auc_score_SB2toSR}")
        wandb.log({"evaluation/auc_score_SB2toSR_AUC": auc_score_SB2toSR})
        results["sb2toSR_AUC"] = auc_score_SB2toSR
    
    if getattr(cfg.step_evaluate, "closure_SKYclassifier_SBtoSR", False) and getattr(cfg.step_evaluate, "closure_SKYclassifier_SBtoSB2transport", False):
        deb_score = ((auc_score_1to2+auc_score_2to1)*2+auc_score_SB1toSR+auc_score_SB2toSR)/6
        log.info(f"deb_score={deb_score}")
        wandb.log({"evaluation/deb_score": deb_score})
        with open(cfg.general.run_dir+"/template/evaluate_sbtosb.txt", "w") as f:
            f.write(f"n_max_class_train={n_max}\n")
            f.write(f"sb1to2 vs sb2 AUC={auc_score_2to1}\n")
            f.write(f"sb2to1 vs sb1 AUC={auc_score_1to2}\n")

    if getattr(cfg.step_evaluate, "plot_everything_else", True):
        evaluate_model(cfg, data["original_data"], data["target_data"], data["template_file"])


def plot_matrix(matrix, title, vmin=-1, vmax=1, abs=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    if abs:
        im = ax.imshow(np.abs(matrix), vmin=0, vmax=vmax, cmap="Blues")
    else:
        im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="RdBu")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    return fig, ax

def evaluate_model(cfg, original_data, target_data, template_data):

    # define some plotting parameters
    scatter_alpha=1
    scatter_s=2
    
    # get the config from the run
    if GlobalHydra().is_initialized():
        GlobalHydra().clear()
    hydra.initialize(version_base=None, config_path= "../config")
    log.info("Loading training config")
    cfg_exp = cfg.step_export_template
    print(cfg_exp.paths.full_path)
    orig_cfg = reload_original_config(cfg_exp, get_best=cfg_exp.get_best)
    cfg_exp = orig_cfg
    plot_path= orig_cfg["paths"]["output_dir"]+"/../plots/"
    os.makedirs(plot_path, exist_ok=True)

    log.info("Loading checkpoint")
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model_class = hydra.utils.get_class(orig_cfg.model._target_)
    model = model_class.load_from_checkpoint(orig_cfg.ckpt_path, map_location=device)
    #model.to(device)

    #log.info("Instantiating original trainer")
    #trainer = hydra.utils.instantiate(orig_cfg.trainer)

    # Instantiate the datamodule use a different config for data then for training
    datamodule = hydra.utils.instantiate(orig_cfg.data.datamodule)
    if hasattr(datamodule, "setup"):
        datamodule.setup("test")
    var_group_list=datamodule.get_var_group_list()

    tra_dataloader = datamodule.train_dataloader()
    torch.manual_seed(0)
    batch1 = next(iter(tra_dataloader))
    #print("batch1:", batch1)

    model.eval() # Set the model to evaluation mode to deactivate dropout layers
    x_inp = batch1[0]
    w2 = batch1[1]
    m_dn = w2

    e1 = model.encode_content(x_inp, m_dn)
    e2 = model.encode_style(m_dn)
    if len(e2.shape) == 1:
        e2 = e2.unsqueeze(1)
    recon = model.decode(e1, e2)
    
    if recon.shape[1] == x_inp.shape[1]:
        w1 = x_inp
    else:
        w1 = torch.concatenate((x_inp, w2), dim=1)
    
    if e1.shape[1] == e2.shape[1]:
        matrix = e1 @ e2.T

        plt.figure()
        plot_matrix(to_np(matrix), "e1 @ e2")
        plt.savefig(plot_path+"e1_at_e2_matrix.png", bbox_inches="tight")
        plt.figure()
        plt.hist(to_np(matrix).flatten(), bins=100)
        plt.title("One batch latent embedding product <e1, e2>")
        plt.xlabel("<e1, e2>")
        plt.savefig(plot_path+"e1_at_e2_hist.png", bbox_inches="tight")
        plt.figure()
        plt.title("One batch latent embedding product <e1, e2> only diagonal")
        plt.xlabel("<e1, e2>")
        plt.hist(np.diagonal(to_np(matrix)), bins=100)
        plt.savefig(plot_path+"e1_at_e2_diag_hist.png", bbox_inches="tight")

        # Same mass
        plt.figure()	
        plt.scatter(np.diagonal(to_np(matrix)), to_np(m_dn), alpha=scatter_alpha, s=scatter_s)
        plt.xlabel("e1 @ e2 diagonal elements")
        plt.ylabel("mjj")
        plt.savefig(plot_path+"e1_at_e2_diag_vs_mjj.png", bbox_inches="tight")

        # Different mass
        n = to_np(matrix).shape[0]
        plt.figure()	
        diag_mask = np.eye(n, dtype=bool)

        # Invert the mask to get the non-diagonal elements
        non_diag_mask = np.logical_not(diag_mask)
        non_diag_elements = to_np(matrix)[non_diag_mask]
        m_1_non_diag = np.tile(m_dn, (1, n))[non_diag_mask]
        m_2_non_diag = np.tile(m_dn.T, (n, 1))[non_diag_mask]
        plt.figure()
        plt.scatter(non_diag_elements, m_1_non_diag, alpha=scatter_alpha, s=scatter_s)
        plt.xlabel("e1 @ e2 non-diagonal elements")
        plt.ylabel("mjj1")
        plt.savefig(plot_path+"e1_at_e2_non_diag_vs_mjj1.png", bbox_inches="tight")
        plt.figure()
        plt.scatter(non_diag_elements, m_2_non_diag, alpha=scatter_alpha, s=scatter_s)
        plt.xlabel("e1 @ e2 non-diagonal elements")
        plt.ylabel("mjj2")
        plt.savefig(plot_path+"e1_at_e2_non_diag_vs_mjj2.png", bbox_inches="tight")
        plt.figure()
        plt.scatter(non_diag_elements, m_1_non_diag-m_2_non_diag, alpha=scatter_alpha, s=scatter_s)
        plt.xlabel("e1 @ e2 non-diagonal elements")
        plt.ylabel("mjj1 - mjj2")
        plt.savefig(plot_path+"e1_at_e2_non_diag_vs_mjj1-mjj2.png", bbox_inches="tight")


    bins= np.linspace(-3, 3, 30)
    for i in range(w1.shape[1]):
        plt.figure()
        plt.hist(to_np(w1[:, i]), bins=bins, histtype='step', label="input")
        plt.hist(to_np(recon[:, i]), bins=bins, histtype='step', label="reconstruction")
        plt.xlabel(f"dim{i}")
        plt.legend()
        plt.savefig(plot_path+f"w1_reco_hist_{i}.png", bbox_inches="tight")
        plt.figure()
        plt.scatter(to_np(w1[:, i]), to_np(recon[:, i]), alpha=scatter_alpha, s=scatter_s)
        plt.xlabel(f"dim{i}_input")
        plt.xlabel(f"dim{i}_reco")
        plt.legend()
        plt.savefig(plot_path+f"w1_reco_scater_{i}.png", bbox_inches="tight")
    # Plot linear correlateion plots for the latent space
    one_corretation_plot=True
    os.makedirs(plot_path+"corerlations/", exist_ok=True)
    person_correlations, spearman_correlations, kendalltaus = plot_correlation_plots(e1, e2, plot_path, one_corretation_plot=True, name="latent_space_correlations", c=m_dn)
    
    if getattr(cfg.step_evaluate.procedures, "mass_correlation_plots", True):
        # Mass correlation plots
        person_correlations_mass = []
        spearman_correlations_mass = []
        kendalltaus_mass = []
        if one_corretation_plot:
            fig, axes = plt.subplots(1, e1.shape[1],  figsize=(3*e1.shape[1], 3))
            fig.suptitle('Scatter plots with Pearson Correlation', fontsize=16)
            for i in range(e1.shape[1]):
                axes[i].scatter(to_np(e1[:, i]), to_np(m_dn), marker="o", label="e1", c=m_dn, cmap="viridis")
                pearson_correlation, p_value = pearsonr(to_np(e1[:, i]), to_np(m_dn).T[0])
                person_correlations_mass.append(pearson_correlation)
                spearman_correlation, p_value = spearmanr(to_np(e1[:, i]), to_np(m_dn).T[0])
                spearman_correlations_mass.append(spearman_correlation)
                kendalltau_correlation, p_value = kendalltau(to_np(e1[:, i]), to_np(m_dn).T[0])
                kendalltaus_mass.append(kendalltau_correlation)
                axes[i].set_title(f"Corr={pearson_correlation:.3f}")
                axes[i].set_xlabel(f"dim{i} e1")
                axes[i].set_ylabel(f"mjj")
            plt.tight_layout()
            plt.savefig(plot_path+"corerlations/"+"latent_space_e1_mass_correlations.png", bbox_inches="tight")


    # Compute some numerical metrics as a summary about model performance
    results = {"max_abs_pearson": np.max(np.abs(person_correlations)), "min_abs_pearson": np.min(np.abs(person_correlations)), "mean_abs_pearson": np.mean(np.abs(person_correlations))}
    results.update({"max_abs_spearman": np.max(np.abs(spearman_correlations)), "min_abs_spearman": np.min(np.abs(spearman_correlations)), "mean_abs_spearman": np.mean(np.abs(spearman_correlations))})
    results["kernel_pearson"] = None
    results["hilbert_schmidt"] = HSIC_torch(e1, e2, cuda=False).detach().cpu().numpy()
    results["DisCo"] = dcor.distance_correlation(to_np(e1), to_np(e2)) if dcor else None
    dcor_torch = DistanceCorrelation() if dcor else None
    results["dcor_torch"] = dcor_torch(e1, e2) if dcor else None

    # Do some fast calassification
    if getattr(cfg.step_evaluate.procedures, "lazy_predict", True):
        from lazypredict.Supervised import LazyClassifier
        print("starting lazy perdict block")
        print(len(target_data))
        print(len(template_data))
        use_n=min(10000, len(target_data), len(template_data))
        X = pd.concat((target_data[:use_n], template_data[:use_n]))
        y = np.concatenate((np.ones(len(target_data[:use_n])), np.zeros(len(template_data[:use_n]))))
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)
        clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
        models,predictions = clf.fit(X_train, X_test, y_train, y_test)

        results["template_max_lazy_Accuracy"] = np.max(models["Accuracy"])
        results["template_max_lazy_ROCAUC"] = np.max(models["ROC AUC"])

    pickle.dump(results, open(plot_path+"results.pkl", "wb"))
    with open(plot_path+"results.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    for key, value in results.items():
        print(key, value)
    w1 = batch1[0]
    w2 = batch1[1]
    
    processor = pickle.load(open(orig_cfg["paths"]["output_dir"]+"/../cathode_preprocessor.pkl", "rb"))
    for var in range(w1.shape[1]):
        var_name=var_group_list[0][var]
        if var_name=="del_R":
            var_name="$\Delta R$"
        if var_name=="del_m":
            var_name="$\Delta m [GeV]$"
        _draw_event_transport_trajectories(model, plot_path, w1, w2, var=var, var_name=var_name, masses=np.linspace(3000, 4600, 1000), max_traj=20, processor=processor)

def _draw_event_transport_trajectories(model, plot_path, w1_, m_pair_, var, var_name, masses=np.linspace(-2.5, 2.5, 126), max_traj=20, processor=None):
    if processor is not None:
        masses_true = copy.deepcopy(masses)
        tensor = torch.zeros((len(masses), 6))
        tensor[:, -1]=torch.Tensor(masses)
        masses = processor.transform(tensor)[:, -1].cpu().detach().numpy().flatten()
    else:
        masses_true = masses
    w1 = copy.deepcopy(w1_)[:max_traj]
    m_pair = m_pair_[:max_traj]
    if processor is not None:
        tensor = torch.zeros((len(m_pair), 6))
        tensor[:, -1]=torch.Tensor(m_pair).flatten()
        m_pair_plot = processor.inverse_transform(tensor)[:, -1:]
    content = model.encode_content(w1, m_pair)
    recons = []
    if model.adversarial:
        zs = []
    for m in masses:
        w2 = torch.tensor(m).unsqueeze(0).expand(w1.shape[0], 1).float().to(w1.device)
        style = model.encode_style(w2)
        recon = model.decode(content, style)
        if processor is not None:
            tensor = torch.zeros((len(recon), 6))
            tensor[:, :-1]=recon
            recon = processor.inverse_transform(tensor)[:, :-1]
        recons.append(recon)
        if model.adversarial:
            if model.use_disc_lat:
                zs.append(model.disc_lat(content, style))
            elif model.use_disc_reco:
                if hasattr(model, "use_disc_reco_doublecond") and model.use_disc_reco_doublecond:
                    zs.append(model.disc_reco(w1, torch.cat([w2, m_pair], dim=1)))
                else:
                    zs.append(model.disc_reco(w1, w2))
    if model.adversarial:
        vmin = min([float(z[:max_traj].min().cpu().detach().numpy()) for z in zs])
        vmax = max([float(z[:max_traj].max().cpu().detach().numpy()) for z in zs])
    plt.figure()
    if max_traj is None:
        max_traj = x.shape[0]
    for i in range(max_traj):
        x=masses_true
        y = np.array([float(recon[i, var].cpu().detach().numpy()) for recon in recons])
        if model.adversarial:
            z = np.array([float(z[i].cpu().detach().numpy()) for z in zs])
            plt.plot(x, y, "black", zorder=i*2+1)
            plt.scatter(x, y, c=z, cmap="turbo", s=2, zorder=i*2+2, vmin=vmin, vmax=vmax)
            if i==0:
                plt.colorbar()
        else:
            plt.plot(x, y, "r")

    for i in range(max_traj):
        if processor is not None:
            tensor = torch.zeros((len(w1), 6))
            tensor[:, :-1]=w1
            w1_plot = processor.inverse_transform(tensor)[:, :-1]
        else:
            w1_plot = w1
        plt.scatter(to_np(m_pair_plot)[:max_traj], to_np(w1_plot[:, var])[:max_traj],  marker="x", label="originals", c="green")
    plt.xlabel("$m_{jj} [GeV]$")
    plt.ylabel(var_name)
    plt.title(f"Event transport for {var_name}")
    plt.savefig(plot_path+f"event_transport_trajectories{var}.png", bbox_inches="tight")
    
def draw_event_transport_trajectories_old(model, plot_path, w1, w2, var, var_name, mass_name="m_jj", masses=np.linspace(-4, 4, 801), max_traj=20):
    recons = []
    w1 = w1[:max_traj]
    w2 = w2[:max_traj]
    e1 = model.encode_content(w1, w2)
    for m in masses:
        w2_new = torch.tensor(m).unsqueeze(0).expand(w1.shape[0], 1).float()
        e2 = model.encode_style(w2_new)
        recon = model.decode(e1, e2)
        recons.append(recon)
    
    plt.figure()
    if max_traj is None:
        max_traj = w1.shape[0]
    for i in range(max_traj):
        plt.plot(masses, [float(recon[i, var].detach().numpy()) for recon in recons], "r")
    for i in range(max_traj):
        plt.scatter(to_np(w1[:, -1])[:max_traj], to_np(w1[:, var])[:max_traj], marker="x", label="originals", c="green")
    plt.xlabel(mass_name)
    plt.ylabel(var_name)
    plt.savefig(plot_path+f"event_transport_trajectories{var}.png", bbox_inches="tight")

def draw_event_transport_trajectories_2d_der(model, plot_path, w1, var, var_name, masses=np.linspace(-4, 4, 801), max_traj=20):
    recons = []
    for m in masses:
        w2 = torch.tensor(m).unsqueeze(0).expand(w1.shape[0], 1).float()
        e1, e2 = model.encode(w1, w2)
        latent = torch.cat([e1, e2], dim=1)
        recon = model.decoder(latent)
        recons.append(recon)
    
    plt.figure()
    if max_traj is None:
        max_traj = w1.shape[0]
    for i in range(max_traj):
        x = masses
        y = np.array([float(recon[i, var].detach().numpy()) for recon in recons])
        plt.plot(x, (2*y[1:-1]-y[:-2]+y[2:])/0.01**2, "r")
    plt.xlabel("mass")
    plt.ylabel(f"dim{var}")
    plt.savefig(plot_path+f"event_transport_trajectories_2nd_der{var}.png", bbox_inches="tight")

def plot_correlation_plots(e1, e2, plot_path, name, c=None, one_corretation_plot=True):
    person_correlations =np.zeros((e1.shape[1], e2.shape[1]))
    spearman_correlations = np.zeros((e1.shape[1], e2.shape[1]))
    kendalltaus = np.zeros((e1.shape[1], e2.shape[1]))
    if one_corretation_plot:
        fig, axes = plt.subplots(e1.shape[1], e2.shape[1], figsize=(3*e1.shape[1], 3*e1.shape[1]))
        fig.suptitle('Scatter plots with Pearson Correlation', fontsize=16)
        for i in range(e1.shape[1]):
            if e2.shape[1] == 1:
                axes[i].scatter(to_np(e1[:, i]), to_np(e2[:]), marker="o", label="e1", c=c, cmap="viridis")
                pearson_correlation, p_value = pearsonr(to_np(e1[:, i]), to_np(e2[:]).flatten())
                person_correlations[i, 0] = pearson_correlation
                spearman_correlation, p_value = spearmanr(to_np(e1[:, i]), to_np(e2[:]).flatten())
                spearman_correlations[i, 0] = spearman_correlation	
                kendalltau_correlation, p_value = kendalltau(to_np(e1[:, i]), to_np(e2[:]).flatten())
                kendalltaus[i, 0] = kendalltau_correlation
                axes[i].set_title(f"Corr={pearson_correlation:.3f}")
                axes[i].set_xlabel(f"dim{i} e1")
                axes[i].set_ylabel(f"dim{0} e2")
            else:
                for j in range(e2.shape[1]):
                    axes[i, j].scatter(to_np(e1[:, i]), to_np(e2[:, j]), marker="o", label="e1", c=c, cmap="viridis")
                    pearson_correlation, p_value = pearsonr(to_np(e1[:, i]), to_np(e2[:, j]))
                    person_correlations[i, j] = pearson_correlation
                    spearman_correlation, p_value = spearmanr(to_np(e1[:, i]), to_np(e2[:, j]))
                    spearman_correlations[i, j] = spearman_correlation	
                    kendalltau_correlation, p_value = kendalltau(to_np(e1[:, i]), to_np(e2[:, j]))
                    kendalltaus[i, j] = kendalltau_correlation
                    axes[i, j].set_title(f"Corr={pearson_correlation:.3f}")
                    axes[i, j].set_xlabel(f"dim{i} e1")
                    axes[i, j].set_ylabel(f"dim{j} e2")
        plt.tight_layout()
        plt.savefig(plot_path+"corerlations/"+name+".png", bbox_inches="tight")
    else:
        for dim1 in range(e1.shape[1]):
            for dim2 in range(e2.shape[1]):
                plt.figure()
                plt.scatter(to_np(e1[:, dim1]), to_np(e2[:, dim2]), marker="o", label="e1", c=c, cmap="viridis")
                pearson_correlation, p_value = pearsonr(to_np(e1[:, i]), to_np(e2[:, j]))
                person_correlations[i, j] = pearson_correlation
                spearman_correlation, p_value = spearmanr(to_np(e1[:, i]), to_np(e2[:, j]))
                spearman_correlations[i, j] = spearman_correlation	
                kendalltau_correlation, p_value = kendalltau(to_np(e1[:, i]), to_np(e2[:, j]))
                kendalltaus[i, j] = kendalltau_correlation
                plt.title(f"Latent space (color=m) pearson={pearson_correlation}")
                plt.xlabel(f"dim{dim1} e1")
                plt.ylabel(f"dim{dim2} e2")
                plt.savefig(plot_path+"corerlations/"+f"{name}_{dim1}_{dim2}.png", bbox_inches="tight")
    
    return person_correlations, spearman_correlations, kendalltaus