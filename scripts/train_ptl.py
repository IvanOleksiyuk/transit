import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

import transit
import logging
import wandb
import time
import hydra
import pytorch_lightning as pl
import torch as T
from omegaconf import DictConfig

from transit.src.utils.hydra_utils import instantiate_collection, log_hyperparameters, print_config, reload_original_config, save_config

log = logging.getLogger(__name__)

def update_sheduler_cfgs(cfg, epoch_scale):
    cfg.model.adversarial_cfg.scheduler.scheduler_g.milestones = [num*epoch_scale for num in cfg.model.adversarial_cfg.scheduler.scheduler_g.milestones]
    cfg.model.adversarial_cfg.scheduler.scheduler_d.milestones = [num*epoch_scale for num in cfg.model.adversarial_cfg.scheduler.scheduler_d.milestones]
    cfg.model.adversarial_cfg.scheduler.scheduler_d2.milestones = [num*epoch_scale for num in cfg.model.adversarial_cfg.scheduler.scheduler_d2.milestones]
    
    cfg.trainer.max_epochs = cfg.trainer.max_epochs*epoch_scale    
    #cfg.trainer.check_val_every_n_epoch = cfg.trainer.check_val_every_n_epoch*epoch_scale
    if hasattr(cfg.model, "valid_plot_freq"):
        cfg.model.valid_plot_freq *= epoch_scale

@hydra.main(
    version_base=None, config_path=str('../config'), config_name="train"
)
def main(cfg: DictConfig) -> None:
    wandb_key = open(cfg.paths.wandbkey, "r").read()
    wandb.login(key=wandb_key)
    run_id = wandb.util.generate_id()
    wandb.init(project=cfg.project_name, id=run_id, name=cfg.network_name, resume="allow")
    with open(cfg.paths.full_path+"/wandb_id.txt", "w") as f:
        f.write(run_id)
    
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
    
    log.info("Scale N epochs with dataseize") #For very small datasets we have to scale the number of epochs
    if cfg.get("do_scale_epochs", False):
        datamodule.setup(stage="fit")
        train_data_length = len(datamodule.train_dataloader().dataset)
        # get the batch size
        batch_size = datamodule.train_dataloader().batch_size
        # get the number of batches
        num_batches = train_data_length // batch_size
        if num_batches < 100:
            epoch_scale = 100 // num_batches
        update_sheduler_cfgs(cfg, epoch_scale)
    
    log.info("Instantiating the model")
    model = hydra.utils.instantiate(cfg.model, inpt_dim=datamodule.get_dims(), var_group_list=datamodule.get_var_group_list(), seed=cfg.seed)
    log.info(model)

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
