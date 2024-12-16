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
    version_base=None, config_path=str('../config'), config_name="transit_reco_cons0.01_smls0.001_adv3_LHCO_CURTAINS1024b.yaml"
) 
def main(cfg: DictConfig) -> None:
    log.info("Setting up full job config")
    print_config(cfg)
    data = cfg.data.export_latent_all.test_data
    dataset_name="LHCO_transit_CATHODE_addgapmass1024b/export_latent_all"
    data.plotting_path = "plot/user/inspect_data/" + dataset_name
    log.info("Instantiating the data module")
    dataset = hydra.utils.instantiate(data)
    print(dataset)
    print(len(dataset))


if __name__ == "__main__":
    main()