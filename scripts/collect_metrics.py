# This script  just build SIC curves and simmilar plots.

import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")
import logging
import hydra
from omegaconf import DictConfig
import os
from pathlib import Path 
log = logging.getLogger(__name__)
import numpy as np
from datetime import timedelta
import re
import pickle
# TODO pyroot utils will remove the need for ../configs

def parse_time(time_str):
    """Convert a time string (H:M:S.F) to total seconds."""
    time_parts = list(map(float, time_str.split(':')))
    return timedelta(hours=time_parts[0], minutes=time_parts[1], seconds=time_parts[2]).total_seconds()

def get_generation_time(file_path):
    """Extract and sum the generation times from the file."""
    generation_steps = ["Generate template"]
    total_seconds = 0.0
    
    with open(file_path, 'r') as file:
        for line in file:
            for step in generation_steps:
                if step in line:
                    time_str = re.search(r'([0-9]+:[0-9]+:[0-9]+\.[0-9]+)', line)
                    if time_str:
                        total_seconds += parse_time(time_str.group(1))
    
    return total_seconds

def get_training_time(root_directory):
    execution_times = []
    # Traverse directory tree
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename == "execution_time.txt":
                # Construct full file path
                file_path = os.path.join(dirpath, filename)
                with open(file_path, 'r') as file:
                    line = file.readline().strip()
                    # Extract time using regex
                    match = re.search(r"Execution Time:\s*([\d.]+)\s*seconds", line)
                    if match:
                        execution_time = float(match.group(1))
                        execution_times.append(execution_time)
    return np.mean(execution_times)


@hydra.main(
    version_base=None, config_path=str('../config'), config_name="transit_reco_cons0.01_smls0.001_adv3_LHCO_CURTAINS1024b"
)
def main(cfg: DictConfig) -> None:
    results = {}
    print("Starting Collecting Metrics")
    if cfg.get("do_mean_SBtoSB_closure_AUC", False):
        # read first AUC from file
        auc_file = cfg.run_dir + "/cwola_SB2/window_3100_3300__3700_3900/dope_0/standard/seed_0/closure_roc_auc.txt"
        with open(auc_file, "r") as f:
            auc_SB1toSB2 = float(f.read())
        # read second AUC from file
        auc_file = cfg.run_dir + "/cwola_SB1/window_3100_3300__3700_3900/dope_0/standard/seed_0/closure_roc_auc.txt"
        with open(auc_file, "r") as f:
            auc_SB2toSB1 = float(f.read())
        results["mean_SBtoSB_closure_AUC"] = (auc_SB1toSB2 + auc_SB2toSB1) / 2
    
    if cfg.get("do_laSB_closure_AUC", False):
        # read AUC from file
        auc_file = cfg.run_dir + "/latent_SB1_vs_SB2/window_3100_3300__3700_3900/dope_0/standard/seed_0/closure_roc_auc.txt"
        with open(auc_file, "r") as f:
            auc_laSB = float(f.read())
        results["laSB_closure_AUC"] = auc_laSB
    
    if cfg.get("do_mean_SBtoSB_closure_AUC", False) and cfg.get("do_laSB_closure_AUC", False):
        results["mean_all_closures"] = (auc_SB1toSB2 + auc_SB2toSB1 + auc_laSB) / 3
        results["max_all_closures"] = max(auc_SB1toSB2, auc_SB2toSB1, auc_laSB)
    
    if cfg.get("do_generation_time", False):
        # read AUC from file
        results["gen_t"] = get_generation_time(cfg.run_dir + "/summary/runtime.txt")
        results["tra_t"] = get_training_time(cfg.run_dir)
        results["tra_n_gen_t"] = results["gen_t"] + results["tra_t"]
        
    # write down the results
    with open(Path(cfg.run_dir) / "summary/metrics.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    pickle.dump(results, open(Path(cfg.run_dir) / "summary/metrics.pkl", "wb"))
        

if __name__ == "__main__":
    main()