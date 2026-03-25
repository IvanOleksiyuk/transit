import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

import transit
import logging
import wandb
import time
import hydra
import pytorch_lightning as pl
import torch as T
import math
import pickle
from omegaconf import DictConfig
from pathlib import Path
import sys
import os

from transit.src.utils.hydra_utils import instantiate_collection, log_hyperparameters, print_config, reload_original_config, save_config
from transit.src.utils.model_visualization import visualize_fx_graph_png

log = logging.getLogger(__name__)


def export_model_visualizations(model, output_dir: Path) -> None:
    """Export FX graph PNGs for model components when available."""
    output_dir.mkdir(parents=True, exist_ok=True)

    modules_to_plot = []
    if hasattr(model, "encoder1") and isinstance(model.encoder1, T.nn.Module):
        modules_to_plot.append(("encoder1", model.encoder1))
    if hasattr(model, "decoder") and isinstance(model.decoder, T.nn.Module):
        modules_to_plot.append(("decoder", model.decoder))
    if hasattr(model, "discriminator") and isinstance(model.discriminator, T.nn.Module):
        modules_to_plot.append(("discriminator", model.discriminator))
    if hasattr(model, "discriminator2") and isinstance(model.discriminator2, T.nn.Module):
        modules_to_plot.append(("discriminator2", model.discriminator2))

    for name, module in modules_to_plot:
        out_file = output_dir / f"fx_graph_{name}.png"
        try:
            saved_path = visualize_fx_graph_png(module, out_file)
            log.info(f"Saved FX graph for {name}: {saved_path}")
        except Exception as err:
            log.warning(f"Could not generate FX graph for {name}: {err}")

def epoch_milestone_list_scale(array, scale):
    new_array = []
    previos_num = 0
    for num in array:
        new_array.append(max(math.ceil(num*scale), previos_num+1))
        previos_num = new_array[-1]
    print("epohch milestones scaled from: ", array, " to: ", new_array)
    return new_array

def update_sheduler_cfgs(cfg, epoch_scale):
    # Skip if no adversarial scheduler is defined (e.g., MMD/energy mode)
    adv = getattr(cfg.model, "adversarial_cfg", None)
    if adv is None or adv.get("scheduler") is None:
        return
    sched = adv.scheduler
    if getattr(sched, "scheduler_g", None) is not None:
        sched.scheduler_g.milestones = epoch_milestone_list_scale(sched.scheduler_g.milestones, epoch_scale)
    if getattr(sched, "scheduler_d", None) is not None:
        sched.scheduler_d.milestones = epoch_milestone_list_scale(sched.scheduler_d.milestones, epoch_scale)
    if getattr(sched, "scheduler_d2", None) is not None:
        sched.scheduler_d2.milestones = epoch_milestone_list_scale(sched.scheduler_d2.milestones, epoch_scale)
    
    cfg.trainer.max_epochs = max(math.ceil(cfg.trainer.max_epochs*epoch_scale), 1)
    cfg.model.adversarial_cfg.warmup = max(math.ceil(cfg.model.adversarial_cfg.warmup*epoch_scale), 1)
    #cfg.trainer.check_val_every_n_epoch = cfg.trainer.check_val_every_n_epoch*epoch_scale
    # if hasattr(cfg.model, "valid_plot_freq"):
    #     cfg.model.valid_plot_freq = max(math.ceil(cfg.model.valid_plot_freq*epoch_scale), 1)

@hydra.main(
    version_base=None, config_path=str('../config'), config_name="train"
)
def main(cfg: DictConfig) -> None:
    
    wandb_key = None
    if cfg.get("wandb_key", False):
        wandb_key = cfg.wandb_key
    elif cfg.get("paths", False) and cfg.paths.get("wandbkey", False):
        wandb_key = open(cfg.paths.wandbkey, "r").read()
    else:
        try:
            wandb_key = os.getenv("WANDB_KEY")
            print(f"Got WANDB_KEY from env variable.")
        except Exception as e:
            print(f"Failed to get WANDB_KEY: {e}. Skipping.")
        try:
            wandb_key = os.getenv("WANDB_API_KEY")
            print(f"Got WANDB_API_KEY from env variable.")
        except Exception as e:
            print(f"Failed to get WANDB_API_KEY: {e}. Skipping.")
    
    if wandb_key:
        wandb.login(key=wandb_key)
        run_id = wandb.util.generate_id()
        wandb.init(project=cfg.project_name, id=run_id, name=cfg.network_name, resume="allow")
        with open(cfg.paths.full_path+"/wandb_id.txt", "w") as f:
            f.write(run_id)
    else:
        print("WANDB_KEY not set. Skipping wandb login.")
    

    
    log.info("Setting up full job config")
    if cfg.full_resume:
        cfg = reload_original_config(cfg)
    print_config(cfg)

    if cfg.seed:
        log.info(f"Setting seed to: {cfg.seed}")
        pl.seed_everything(cfg.seed, workers=True)

    if cfg.precision:
        log.info(f"Setting matrix precision to: {cfg.precision}")
        T.set_float32_matmul_precision(cfg.precision)

    log.info("Instantiating the data module")
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    if hasattr(datamodule, "setup"):
        datamodule.setup(stage="fit")
    
    # get info from the data module if there are quantized vars for padTRANSIT (very hacky solution but it works)
    dequantization_cfg = None
    if hasattr(cfg, "preprocessing_pkl"):
        cwd = Path.cwd()
        src_path = cwd / "src"
        sys.path.append(str(src_path))
        with open(cfg.preprocessing_pkl, "rb") as f:
            preprocessor = pickle.load(f)
            standardiser = preprocessor.features_preprocess.info
            if hasattr(preprocessor, "discrete_indices") and preprocessor.discrete_indices is not None:
                discrete_indices = preprocessor.discrete_indices
                dequantization_cfg = {}
                dequantization_cfg["discrete_indices"] = discrete_indices
                dequantization_cfg["discrete_shift"] = standardiser[0, discrete_indices].numpy()
                dequantization_cfg["discrete_scale"] = standardiser[1, discrete_indices].numpy()
        root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")
        sys.path.remove(str(src_path))
        sys.path.append(str(cwd))
        
    log.info("Scale N epochs with dataseize") #For very small datasets we have to scale the number of epochs
    if cfg.get("do_scale_epochs", False):
        batches_per_epoch_desired = cfg.get("batches_per_epoch_desired", 100)
        train_data_length = len(datamodule.train_dataloader().dataset)
        # get the batch size
        batch_size = datamodule.train_dataloader().batch_size
        # get the number of batches
        num_batches = train_data_length // batch_size
        if cfg.do_scale_epochs=="increase_only":
            if num_batches < batches_per_epoch_desired:
                epoch_scale = batches_per_epoch_desired // num_batches
                update_sheduler_cfgs(cfg, epoch_scale)
        else:
            epoch_scale = batches_per_epoch_desired / num_batches
            update_sheduler_cfgs(cfg, epoch_scale)
    
    log.info("Instantiating the model")
    model = hydra.utils.instantiate(cfg.model, inpt_dim=datamodule.get_dims(), var_group_list=datamodule.get_var_group_list(), seed=cfg.seed, dequantization_cfg=dequantization_cfg)
    log.info(model)

    log.info("Exporting model visualizations")
    export_model_visualizations(model, Path(cfg.paths.full_path))

    log.info("Saving config so job can be resumed")
    save_config(cfg)

    log.info("Instantiating all callbacks")
    callbacks = instantiate_collection(cfg.callbacks)

    log.info("Instantiating the loggers")
    loggers = instantiate_collection(cfg.loggers)

    log.info("Instantiating the trainer")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    log.info("Starting training!")
    start_time = time.time()
    
    if cfg.compile:
        log.info(f"Compiling the model using torch 2.0: {cfg.compile}")
        model = T.compile(model, mode=cfg.compile)

    if loggers:
        log.info("Logging all hyperparameters")
        log_hyperparameters(cfg, model, trainer)

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total trainng time: {elapsed_time:.2f} seconds")
    formatted_time = f"Execution Time: {elapsed_time:.2f} seconds\n"

    # Write the elapsed time to a text file
    with open(cfg.paths.full_path+"/execution_time.txt", "a") as file:  # Use "a" to append to the file
        file.write(formatted_time)

    
        

if __name__ == "__main__":
    main()
