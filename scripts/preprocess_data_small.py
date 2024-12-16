import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

import logging
import hydra
from pathlib import Path
import os
from omegaconf import DictConfig, OmegaConf
from transit.src.utils.hydra_utils import instantiate_collection, log_hyperparameters, print_config, reload_original_config, save_config
log = logging.getLogger(__name__)

@hydra.main(
    version_base=None, config_path=str('../config/data'), config_name="LHCO_preprocess_CURTAINS_small.yaml"
) 
def main(cfg: DictConfig) -> None:
    log.info("Setting up full job config")
    print_config(cfg)
    data = cfg.data
    log.info("Instantiating the data module")
    dataset = hydra.utils.instantiate(data)
    dataset.reset_index()
    dataset.save("/home/users/o/oleksiyu/scratch/DATA/LHCO/small/events_anomalydetection_v2.features_prepCURTAINS.h5")
    print(len(dataset))

if __name__ == "__main__":
    main()