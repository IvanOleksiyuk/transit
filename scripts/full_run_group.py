# This is a script to do one full LHCO analysis. 
# It is a combination of the several steps.

# 1 - Create a separate folder for the experiment save all the relavant configs there 
# 2 - train/create a model that will provide a us with a template (e.g. CATHODE, CURTAINS)
# 3 - generate a template dataset using the model
# 4 - evaluate the performance of the template generation model
# 4 - train cwola
# 5 - evaluate the performance and plot the results
# 6 - produce a set of final plots and tables for one run
 
import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

import logging
import hydra
from pathlib import Path
import os
import shutil
from omegaconf import DictConfig, OmegaConf, ListConfig
from transit.src.utils.hydra_utils import instantiate_collection, log_hyperparameters, print_config, reload_original_config, save_config
import transit.scripts.evaluation as evaluation
import transit.scripts.full_run as full_run
import copy
import transit.scripts.plot_compare as plot_compare
log = logging.getLogger(__name__)

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
    version_base=None, config_path=str('../config'), config_name="full_run_group_dopings_3seeds.yaml"
)
def main(cfg: DictConfig) -> None:
    log.info("<<<START FULL RUN>>>")
    orig_full_run = hydra.compose(config_name=cfg.full_run_cfg)
    config_list = [copy.deepcopy(orig_full_run)]
    config_list[0].general.run_dir = cfg.run_dir + "/run"
    

    if hasattr(cfg, "several_SBSR"):
        config_list = expand_SBSR(config_list, cfg.several_SBSR, cfg.check_SBSR)
    if hasattr(cfg, "several_doping"):
        config_list = expand_doping(config_list, cfg.several_doping, cfg.check_doping)
    if hasattr(cfg, "several_template_train_seeds"):
        config_list = expand_template_train_seed(config_list, cfg.several_template_train_seeds, not_just_seeds=cfg.not_just_seeds)
    
    # Create a folder for the run group
    group_dir = Path(cfg.run_dir)
    os.makedirs(group_dir, exist_ok=True)

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
                full_run.main(run_cfg)
                with open(done_file_path, "w") as f:
                    f.write("All done for this run!")
                    f.close()
    
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
        

if __name__ == "__main__":
    main()
    

