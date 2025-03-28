import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

#from libs_snap.anomdiff.src.datamodules.cnst_lhco import LHCOLowDataset
import logging
import hydra
from pathlib import Path
import os
from omegaconf import DictConfig, OmegaConf
from transit.src.utils.hydra_utils import instantiate_collection, log_hyperparameters, print_config, reload_original_config, save_config
log = logging.getLogger(__name__)

@hydra.main(
    version_base=None, config_path=str('../config'), config_name="TTL_reco_LLCR_1024b_lr2-4_dl.yaml"
) 
def main(cfg: DictConfig) -> None:
    log.info("Setting up full job config")
    print_config(cfg)
    data = cfg.data.template_training.dataset    
    dataset = hydra.utils.instantiate(data)
    print(dataset)
    print(len(dataset))


if __name__ == "__main__":
    main()