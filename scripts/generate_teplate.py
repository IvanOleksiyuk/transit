import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

from typing import Union
from transit.src.utils.hydra_utils import reload_original_config

import logging
from pathlib import Path

import hydra
import torch as T
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import h5py
import shutil
import os
log = logging.getLogger(__name__)

def to_np(inpt: Union[T.Tensor, tuple]) -> np.ndarray:
    """More consicse way of doing all the necc steps to convert a pytorch
    tensor to numpy array.

    - Includes gradient deletion, and device migration
    """
    if isinstance(inpt, (tuple, list)):
        return type(inpt)(to_np(x) for x in inpt)
    if inpt.dtype == T.bfloat16:  # Numpy conversions don't support bfloat16s
        inpt = inpt.half()
    return inpt.detach().cpu().numpy()


@hydra.main(
    version_base=None, config_path=str(root / "conf"), config_name="export.yaml"
)
def main(cfg: DictConfig) -> None:
    log.info("Loading run information")
    orig_cfg = reload_original_config(cfg, get_best=cfg.get_best)

    log.info("Loading best checkpoint")
    device = "cuda" if T.cuda.is_available() else "cpu"
    model_class = hydra.utils.get_class(orig_cfg.model._target_)
    model = model_class.load_from_checkpoint(orig_cfg.ckpt_path, map_location=device)

    log.info("Instantiating original trainer")
    trainer = hydra.utils.instantiate(orig_cfg.trainer)

    # Instantiate the datamodule use a different config for data then for training
    if hasattr(orig_cfg, "data"):
        datamodule = hydra.utils.instantiate(cfg.data)
    else:
        datamodule = hydra.utils.instantiate(orig_cfg.datamodule)

    # Cycle through the datasets and create the dataloader
    log.info("Running generation")
    model.eval() #PL should do it but I just do it to be sure
    outputs = trainer.predict(model=model, datamodule=datamodule)
    output_dir = Path(orig_cfg.paths.full_path, "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(cfg, "output_name"):
        output_name = cfg.output_name
    else:
        output_name = "template_sample"
    if isinstance(outputs[0], dict):
        dataset_dict = {var: T.vstack([o[var] for o in outputs]).numpy() for var in outputs[0].keys()}
        log.info("Saving outputs")
        output_dir = Path(orig_cfg.paths.full_path, "outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({k:v.reshape(-1) for k, v in dataset_dict.items()})
        df.to_hdf(output_dir / f"{output_name}.h5", key="template", mode="w")
    else:
        all = T.vstack([o for o in outputs]).numpy()
        jet1 = all[: len(all) // 2] 
        jet2 = all[len(all) // 2 :]
        out_file = output_dir / f"{output_name}.h5"
        with h5py.File(out_file, "a") as file:
            #del file["jet1_locals"]
            file.create_dataset("jet1_locals", data=jet1)
        with h5py.File(out_file, "a") as file:
            #del file["jet2_locals"]
            file.create_dataset("jet2_locals", data=jet2)
        
    print(f"Saved template to {output_dir / f'{output_name}.h5'}")
    
    # Move or copy the file to another location if TRANSIT is a submodule of another project
    if hasattr(cfg, "copy_to_path") and cfg.copy_to_path is not None:
        Path(cfg.copy_to_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(output_dir / f"{output_name}.h5", cfg.copy_to_path)
        print(f"Also copied to {cfg.copy_to_path}")
    elif hasattr(cfg, "move_to_path") and cfg.move_to_path is not None:
        Path(cfg.move_to_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(output_dir / f"{output_name}.h5", cfg.move_to_path)
        print(f"Also moved to {cfg.move_to_path}")
    
    
    

if __name__ == "__main__":
    main()
