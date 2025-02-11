# This is a script to do one full LHCO analysis. 
# It is a combination of the several steps.

# 1 - Create a separate folder for the experiment save all the relavant configs there 
# 2 - train/create a model that will provide a us with a template (e.g. CATHODE, CURTAINS)
# 3 - generate a template dataset using the model
# 4 - evaluate the performance of the template generation model
# 4 - train cwola
# 5 - evaluate the performance and plot the results
# 6 - produce a set of final plots and tables for one run
# 7 - aggregate the metrics from all the runs
 
import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

import logging
import hydra
from pathlib import Path
import os
import shutil
from omegaconf import DictConfig, OmegaConf, ListConfig
import transit.scripts.full_run as full_run
import copy
import transit.scripts.plot_compare as plot_compare
import matplotlib.pyplot as plt
import sys
log = logging.getLogger(__name__)
import pickle
import numpy as np

def parse_metrics_file(filepath):
    """Parses a metrics file and returns a dictionary of metrics."""
    if filepath.endswith('.txt'):
        metrics = {}
        with open(filepath, 'r') as file:
            for line in file:
                if ':' in line:
                    key, value = line.split(':')
                    metrics[key.strip()] = float(value.strip())
        return metrics
    elif filepath.endswith('.pkl'):
        with open(filepath, 'rb') as file:
            return pickle.load(file)
    else:
        raise ValueError("Unsupported file type. Use a '.txt' or '.pkl' file.")

def find_and_process_metrics(root_folder, output_file, metrics_filename="metrics.txt"):
    """Finds all metrics files, computes averages and standard deviations, and writes results to output_file."""
    all_metrics = {}
    
    # Walk through directory tree to find all specified metrics files
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename == metrics_filename:
                file_path = os.path.join(dirpath, filename)
                metrics = parse_metrics_file(file_path)
                
                for key, value in metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
    
    # Compute average and standard deviation for each metric
    final_metrics = {}
    for key, values in all_metrics.items():
        avg = np.mean(values)
        std = np.std(values, ddof=1)  # Using sample standard deviation (ddof=1)
        final_metrics[key] = (avg, std)
    
    # Write results to output file
    with open(output_file, 'w') as file:
        for key, (avg, std) in final_metrics.items():
            file.write(f"{key}_avg: {avg:.4f}\n")
            file.write(f"{key}_std: {std:.4f}\n")
    
    pickle.dump(final_metrics, open(output_file.with_suffix('.pkl'), 'wb'))
            
def modify_copy_and_submit(path1, path2):
    # Ensure the provided paths exist
    if not os.path.isfile(path1):
        print(f"Error: File '{path1}' does not exist.")
        sys.exit(1)
    if not os.path.isdir(path2):
        print(f"Error: Directory '{path2}' does not exist.")
        sys.exit(1)
    
    # Extract the filename of the .sh file
    sh_filename = os.path.basename(path1)
    
    # Construct the path to the new .sh file in the path2 directory
    new_sh_path = os.path.join(path2, sh_filename)
    
    # Copy the original .sh file to the new location
    shutil.copy(path1, new_sh_path)
    
    # Read the content of the copied .sh file
    with open(new_sh_path, 'r') as file:
        content = file.read()
    
    # Replace the configuration name with full_config.yaml
    updated_content = content.replace(
        "--config-name TRANSITv0v2_LHCO", 
        f"--config-path {path2} --config-name full_config.yaml"
    )
    
    updated_content = updated_content.replace(
        "# SAVE SCONTROL PLACEHOLDER", 
        f"echo \"Running on node: $SLURMD_NODENAME\" > {path2}/node_info.txt"
    )
    
    # Write the updated content back to the new .sh file
    with open(new_sh_path, 'w') as file:
        file.write(updated_content)
    
    print(f"Modified script has been copied to: {new_sh_path}")
    print(f"Updated to use full_config.yaml located in: {path2}")
    
def expand_template_train_seed(config_list, several_template_train_seeds, not_just_seeds=False):
    new_config_list = []
    if several_template_train_seeds is None:
        return config_list
    for cfg in config_list:
        for seed in several_template_train_seeds:
            new_config=copy.deepcopy(cfg)
            new_config.step_train_template.seed = seed
            if not_just_seeds:
                new_config.general.run_dir = cfg.general.run_dir + f"/run-TTS_{seed}"
            else:
                new_config.general.run_dir = cfg.general.run_dir + f"-TTS_{seed}"
            new_config_list.append(new_config)
    return new_config_list

def expand_SBSR(config_list, several_SBSR, check_SBSR):
    new_config_list = []
    if several_SBSR is None:
        return config_list
    for cfg in config_list:
        for SBSR in several_SBSR:
            new_config=copy.deepcopy(cfg)
            replace_specific_name_in_omegacfg(new_config, "intervals", SBSR.SB_set, check_value=check_SBSR.SB_get) 
            replace_specific_name_in_omegacfg(new_config, "intervals", SBSR.SR_set, check_value=check_SBSR.SR_get)
            new_config.general.run_dir = cfg.general.run_dir + f"-SBSR_{SBSR.name}"
            new_config_list.append(new_config)
    return new_config_list

def expand_doping(config_list, several_doping, check_doping):
    new_config_list = []
    if several_doping is None:
        return config_list
    for cfg in config_list:
        for doping in several_doping:
            new_config=copy.deepcopy(cfg)
            replace_specific_name_in_omegacfg(new_config, "n_sig", doping, check_value=check_doping)
            replace_specific_name_in_omegacfg(new_config, "num_signal", doping, check_value=check_doping)
            #find_specific_name_in_omegacfg(new_config, "n_sig")
            #find_specific_name_in_omegacfg(new_config, "num_signal")
            new_config.general.run_dir = cfg.general.run_dir + f"-doping_{doping}"
            new_config_list.append(new_config)
    return new_config_list            

def equal_simple(a, b):
    return a == b

def equal_list_simple(a, b):
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True

def equal_list_list_simple(a, b):
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True

def delete_folder_if_exists(folder_path):
    try:
        # Check if the folder exists
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # Remove the folder and all its contents
            shutil.rmtree(folder_path)
            print(f"Folder '{folder_path}' has been deleted.")
        else:
            print(f"Folder '{folder_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred while deleting the folder: {e}")

def replace_specific_name_in_omegacfg(cfg, search_name, insert_value, check_value=None, check_func=equal_simple):
    # Recursively search and replace search_name in the config
    def replace_in_dict(d):
        if isinstance(d, (list, ListConfig)):
            for item in d:
                if isinstance(item, (dict, DictConfig, list, ListConfig)):
                    replace_in_dict(item) 
            return
        elif isinstance(d, (dict, DictConfig)):
            for key, value in d.items():
                if key == search_name:
                    if check_func is not None:
                        if check_func(value, check_value):
                            d[key] = insert_value
                    else:
                        d[key] = insert_value
                elif isinstance(value, (dict, DictConfig)):
                    replace_in_dict(value)
                elif isinstance(value, (list, ListConfig)):
                    for item in value:
                        if isinstance(item, (dict, DictConfig, list, ListConfig)):
                            replace_in_dict(item)
                elif isinstance(value, (int, float, str, bool, type(None))):
                    pass
                else:
                    raise ValueError(f"Unknown type {type(value)}")
        else:
            raise ValueError(f"Unknown type {type(d)}")
    replace_in_dict(cfg)
    return cfg

# def find_specific_name_in_omegacfg(cfg, search_name):
#     print("started")
#     # Recursively search and replace search_name in the config
#     def find_in_dict(d):
#         if isinstance(d, (list, ListConfig)):
#             for item in d:
#                 if isinstance(item, (dict, DictConfig, list, ListConfig)):
#                     find_in_dict(item) 
#             return
#         elif isinstance(d, (dict, DictConfig)):
#             for key, value in d.items():
#                 if key == search_name:
#                     print("!!!!!!!! FOUND !!!!!!!!")
#                     print(key, value)
#                 elif isinstance(value, (dict, DictConfig)):
#                     find_in_dict(value)
#                 elif isinstance(value, (list, ListConfig)):
#                     for item in value:
#                         if isinstance(item, (dict, DictConfig, list, ListConfig)):
#                             find_in_dict(item)
#                 elif isinstance(value, (int, float, str, bool, type(None))):
#                     pass
#                 else:
#                     pass
#                     #raise ValueError(f"Unknown type {type(value)}")
#         else:
#             pass
#             raise ValueError(f"Unknown type {type(d)}")
#     find_in_dict(cfg)


def replace_specific_name_in_cfg(cfg, search_name, insert_value, check_value=None, check_func=equal_simple):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Replace all 'seeds' elements
    updated_cfg_dict = replace_specific_name_in_omegacfg(cfg, search_name, insert_value, check_value=check_value, check_func=check_func)
    
    # Convert back to OmegaConf if needed
    updated_cfg = OmegaConf.create(updated_cfg_dict)

# TODO pyroot utils will remove the need for ../configs
@hydra.main(
    version_base=None, config_path=str('../config'), config_name="full_run_group_stability_30.yaml"
)
def main(cfg: DictConfig) -> None:
    log.info("<<<START FULL RUN>>>")

    # Create a folder for the run group
    group_dir = Path(cfg.run_dir)
    os.makedirs(group_dir, exist_ok=True)

    # Get the full config
    if hasattr(cfg, "hyperparameters"):
        hyper_overrides = []
        for key, value in cfg.hyperparameters.items():
            hyper_overrides.append(f"hyperparameters.{key}={value}")
        pickle.dump(dict(cfg.hyperparameters), open(cfg.run_dir + "/hyperparameters.pkl", "wb"))
    else:
        hyper_overrides = []
    orig_full_run = hydra.compose(config_name=cfg.full_run_cfg, overrides=hyper_overrides+cfg.overrides)
    config_list = [copy.deepcopy(orig_full_run)]
    config_list[0].general.run_dir = cfg.run_dir + "/run"
    

    if hasattr(cfg, "several_SBSR"):
        config_list = expand_SBSR(config_list, cfg.several_SBSR, cfg.check_SBSR)
    if hasattr(cfg, "several_doping"):
        config_list = expand_doping(config_list, cfg.several_doping, cfg.check_doping)
    if hasattr(cfg, "several_template_train_seeds"):
        config_list = expand_template_train_seed(config_list, cfg.several_template_train_seeds, not_just_seeds=cfg.not_just_seeds)


    # For each config create its directory and save the config
    for i, run_cfg in enumerate(config_list):
        done_file_path = run_cfg.general.run_dir+"/ALL.DONE"
        if os.path.isfile(done_file_path) and not cfg.redo:
            print(f"Run {done_file_path} already exists. Skipping.")
            continue
        log.info(f"Run {done_file_path}")
        run_dir = Path(run_cfg.general.run_dir)
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(run_cfg.step_train_template.paths.full_path, exist_ok=True)
        OmegaConf.save(run_cfg, Path(run_cfg.general.run_dir, "full_config.yaml"), resolve=True)
    
    # Run for ech config in the list
    if cfg.run_sequentially:
        for i, run_cfg in enumerate(config_list):
            done_file_path = run_cfg.general.run_dir+"/ALL.DONE"
            if os.path.isfile(done_file_path) and not cfg.redo:
                print("already done " + done_file_path)
            else:
                delete_folder_if_exists(run_cfg.general.run_dir+"/template")
                full_run.main(copy.deepcopy(run_cfg))
            plt.close("all")
    else:
        print("Running in parallel jobs")
        for i, run_cfg in enumerate(config_list):
            done_file_path = run_cfg.general.run_dir+"/ALL.DONE"
            if os.path.isfile(done_file_path) and not cfg.redo:
                print("already done " + done_file_path)
            else:
                #delete_folder_if_exists(run_cfg.general.run_dir+"/template")
                modify_copy_and_submit(cfg.one_run_sh, os.path.abspath(run_cfg.general.run_dir))
    
    if cfg.do_stability_analysis:
        log.info("Stability analysis")
        if cfg.stability_analysis_cfg.run_dir is None:
            if not cfg.not_just_seeds:
                cfg.stability_analysis_cfg.run_dir = group_dir
                plot_compare.main(cfg.stability_analysis_cfg)
            else:
                for item in os.listdir(group_dir):
                    print("run combination in")
                    print(os.path.join(group_dir, item))
                    item_path = os.path.join(group_dir, item)
                    cfg.stability_analysis_cfg.run_dir = str(item_path)
                    plot_compare.main(cfg.stability_analysis_cfg)
        else:
            plot_compare.main(cfg.stability_analysis_cfg)
    
    if cfg.do_metric_aggregation:
        log.info("Metric aggregation")
        find_and_process_metrics(group_dir, group_dir / "metrics_comb.txt")
        

if __name__ == "__main__":
    main()
    

