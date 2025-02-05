import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")
import os
import logging

import hydra
from omegaconf import DictConfig

import subprocess

log = logging.getLogger(__name__)

# TODO pyroot utils will remove the need for ../configs
@hydra.main(
    version_base=None, config_path=str('../config/step_cwola'), config_name="cwola"
)
def main(cfg: DictConfig) -> None:
    # prepare data for running cwola 
    log.info("<<<START CWOLA SCRIPT>>>")
    os.makedirs(cfg.cwola_path, exist_ok=True)
    datasr = hydra.utils.instantiate(cfg.datasets.datasr)
    print("datasr len:", len(datasr))
    cfg.cwola_subfolders = f"window_{cfg.sideband_1}__{cfg.sideband_2}/dope_{cfg.num_signal}/"
    datasr.write_npy_single(file_path_str=cfg.cwola_path+cfg.cwola_subfolders+"sr.npy", key="data")
    datasr.write_features_txt(file_path_str=cfg.cwola_path+cfg.cwola_subfolders+"features.txt", key="data")
    
    template = hydra.utils.instantiate(cfg.datasets.template)
    print("template len:", len(template))
    template.write_npy_single(file_path_str=cfg.cwola_path+cfg.cwola_subfolders+"template.npy", key="template")

    extra_signal = hydra.utils.instantiate(cfg.datasets.extra_signal)
    if extra_signal is not None:
        print("extra_signal len:", len(extra_signal))
        extra_signal.write_npy_single(file_path_str=cfg.cwola_path+cfg.cwola_subfolders+"extra_signal.npy", key="data")
    
    extra_bkg = hydra.utils.instantiate(cfg.datasets.extra_bkg)
    if extra_bkg is not None:
        print("extra_signal len:", len(extra_bkg))
        extra_bkg.write_npy_single(file_path_str=cfg.cwola_path+cfg.cwola_subfolders+"extra_bkg.npy", key="data")
        
    for seed in cfg.seeds:
        #python libs_snap/linearanomaly/cwola.py --input_path=workspaces/dev/transit_DisCo_LHCO_CWOLA/cwola/ --mode=standard --num_signal=3000 --sideband_1=3100_3300 --sideband_2=3700_3900 --num_folds=5 --max_iter=250 --early_stopping=True --validation_fraction=0.1 --class_weight=balanced --num_ensemble=5 --seed=0
        #python libs_snap/linearanomaly/cwola.py --input_path=/home/users/o/oleksiyu/WORK/hyperproject/workspaces/adv1_gauss_corr_4_gap_transit_usem_addgapmass/transit_reco_cons0.01_smls0.001_adv3_LHCO_CURTAINS1024b/cwola/ --mode=standard --num_signal=3000 --sideband_1=3100_3300 --sideband_2=3700_3900 --num_folds=5 --max_iter=250 --early_stopping=1 --validation_fraction=0.1 --class_weight=balanced --num_ensemble=5 --seed=0
        #python libs_snap/linearanomaly/cwola.py 
        # --input_path=/home/users/o/oleksiyu/WORK/hyperproject/workspaces/adv1_gauss_corr_4_gap_transit_usem_addgapmass/transit_reco_cons0.01_smls0.001_adv3_LHCO_CURTAINS1024b/cwola/ 
        # --mode=standard 
        # --num_signal=3000 
        # --sideband_1=3100_3300 
        # --sideband_2=3700_3900 
        # --num_folds=5 
        # --max_iter=250 
        # --early_stopping=1 
        # --validation_fraction=0.1 
        # --class_weight=balanced 
        # --num_ensemble=5 
        # --seed=0
        command = [
            "python", 
            "transit/cwola.py",
            "--input_path=" + cfg.cwola_path,
            "--mode=" + cfg.mode,
            "--num_signal=" + str(cfg.num_signal),
            "--sideband_1=" + str(cfg.sideband_1),
            "--sideband_2=" + str(cfg.sideband_2),
            "--num_folds=" + str(cfg.num_folds),
            "--max_iter=" + str(cfg.max_iter),
            #"--early_stopping=" + str(cfg.early_stopping),
            "--validation_fraction=" + str(cfg.validation_fraction),
            "--class_weight=" + str(cfg.class_weight),
            "--num_ensemble=" + str(cfg.num_ensemble),
            "--seed=" + str(seed),
            "--title_suffix=" + str(cfg.cwola_eval_name),
            ]
        
        if cfg.get("extra_bkg", None) is not None:
            command.append("--extra_bkg=" + str(cfg.extra_bkg))
        else:
            command.append("--extra_bkg=1")

        if cfg.get("extra_signal", None) is not None:
            command.append("--extra_signal=" + str(cfg.extra_signal))
        else:
            command.append("--extra_signal=1")

        print(" ".join(command))
        # run cwola
        return_code = subprocess.run(command) 
        print(return_code)  
        if return_code.returncode != 0:
            raise Exception("cwola failed") 
    log.info("<<<FINISH CWOLA SCRIPT>>>")
    
    # analyse results of cwola
    
if __name__ == "__main__":
    main()