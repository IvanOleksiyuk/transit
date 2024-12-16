import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

import logging
import hydra
from pathlib import Path
import os
from omegaconf import DictConfig, OmegaConf
from transit.src.utils.hydra_utils import instantiate_collection, log_hyperparameters, print_config, reload_original_config, save_config
log = logging.getLogger(__name__)
import matplotlib.pyplot as plt

@hydra.main(
    version_base=None, config_path=str('../config'), config_name="transit_reco_cons_DisCo_wide_LHCO_CURTAINS_nogap_train.yaml"
)
def main(cfg: DictConfig) -> None:
    log.info("Setting up full job config")
    print_config(cfg)
    data_cfg = cfg.data.template_training.train_data
    dataset_name = "train_data_cathode"
    data_cfg.plotting_path = "plot/user/inspect_data3/" + dataset_name
    log.info("Instantiating the data module")
    dataset = hydra.utils.instantiate(data_cfg)
    
    plt.figure()
    plt.hist2d(dataset.data["data"]["m_jj"], dataset.data["data"]["del_R"], bins=100, cmap="turbo")
    plt.colorbar()
    plt.xlabel("m_jj")
    plt.ylabel("del_R")
    plt.savefig(data_cfg.plotting_path + "/mjj_vs_delR.png", bbox_inches='tight')

    plt.figure()
    plt.hist(dataset.data["data"]["m_jj"], bins=100)
    plt.xlabel("m_jj")
    plt.savefig(data_cfg.plotting_path + "/mjj.png", bbox_inches='tight')
    
    plt.figure()
    plt.hist(dataset.data["data"]["del_R"], bins=100)
    plt.xlabel("del_R")
    plt.ylim(-3, 4)
    plt.savefig(data_cfg.plotting_path + "/delR.png", bbox_inches='tight')

if __name__ == "__main__":
    main()