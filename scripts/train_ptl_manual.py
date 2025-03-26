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
import yaml
from omegaconf import DictConfig
from pathlib import Path

from transit.src.utils.hydra_utils import instantiate_collection, log_hyperparameters, print_config, reload_original_config, save_config

log = logging.getLogger(__name__)

def epoch_milestone_list_scale(array, scale):
    new_array = []
    previos_num = 0
    for num in array:
        new_array.append(max(math.ceil(num*scale), previos_num+1))
        previos_num = new_array[-1]
    print("epohch milestones scaled from: ", array, " to: ", new_array)
    return new_array

def update_sheduler_cfgs(cfg, epoch_scale):
    if hasattr(cfg.model.adversarial_cfg.scheduler.scheduler_g, "milestones"):
        cfg.model.adversarial_cfg.scheduler.scheduler_g.milestones = epoch_milestone_list_scale(cfg.model.adversarial_cfg.scheduler.scheduler_g.milestones, epoch_scale)
        cfg.model.adversarial_cfg.scheduler.scheduler_d.milestones = epoch_milestone_list_scale(cfg.model.adversarial_cfg.scheduler.scheduler_d.milestones, epoch_scale)
        cfg.model.adversarial_cfg.scheduler.scheduler_d2.milestones = epoch_milestone_list_scale(cfg.model.adversarial_cfg.scheduler.scheduler_d2.milestones, epoch_scale)
    
    cfg.trainer.max_epochs = max(math.ceil(cfg.trainer.max_epochs*epoch_scale), 1)
    cfg.model.adversarial_cfg.warmup = max(math.ceil(cfg.model.adversarial_cfg.warmup*epoch_scale), 1)
    #cfg.trainer.check_val_every_n_epoch = cfg.trainer.check_val_every_n_epoch*epoch_scale
    # if hasattr(cfg.model, "valid_plot_freq"):
    #     cfg.model.valid_plot_freq = max(math.ceil(cfg.model.valid_plot_freq*epoch_scale), 1)
    

def update_model_parameters(model, config_path):
    # Load the configuration file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Update the model's parameters
    print("updating loss_cfg.reco.w from ", model.loss_cfg.reco.w, " to ", config['loss_weights']['reco'])
    model.loss_cfg.reco.w = config['loss_weights']['reco']
    print("updating loss_cfg.consistency_x.w from ", model.loss_cfg.consistency_x.w, " to ", config['loss_weights']['consistency_x'])
    model.loss_cfg.consistency_x.w = config['loss_weights']['consistency_x']
    
    model.adversarial_cfg.scheduler.scheduler_g.factor = config['lr_g']
    model.adversarial_cfg.scheduler.scheduler_d.factor = config['lr_d']
    #model.adversarial_cfg.sheduler.sheduler_d2.base_lr = config['lr_d2']
    #trainer.lr_scheduler_configs[0].scheduler.base_lr = config['lr_g']
    #trainer.lr_scheduler_configs[2].scheduler.base_lr = config['lr_d']

@hydra.main(
    version_base=None, config_path=str('../config'), config_name="train"
)
def main(cfg: DictConfig) -> None:
    if cfg.get("wandb_key", False):
        wandb_key = cfg.wandb_key
    else:
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
    if hasattr(datamodule, "setup"):
        datamodule.setup(stage="fit")
    
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

    # Training loop with manual intervention
    total_epochs = cfg.trainer.max_epochs
    current_epoch = 0
    checkpoint_path = None

    while current_epoch < total_epochs:
        print("manual training loop")
        # Determine the number of epochs for the next chunk
        chunk_epochs = cfg.get("chunk_epochs", 5)
        next_epoch = min(current_epoch + chunk_epochs, total_epochs)
        
        # Re-instantiate the trainer with the updated max_epochs and resume from checkpoint if available
        trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers, max_epochs=next_epoch)

        # Train the model for the current chunk
        trainer.fit(model, datamodule=datamodule) #, ckpt_path=checkpoint_path
        
        # Update the current epoch
        current_epoch = next_epoch
        
        # Save the checkpoint path
        checkpoint_path = str(Path(trainer.checkpoint_callback.best_model_path).parent / "last.ckpt")
        
        if current_epoch < total_epochs:
            user_input = input("Training paused. Enter 'y' to continue or 'n' to stop: ").strip().lower()
            if user_input in ['y', 'yes']:
                update_model_parameters(model, cfg.config_update_path)
                log.info(f"Model parameters updated from {cfg.config_update_path}")
            elif user_input in ['n', 'no']:
                log.info("Training stopped by user.")
                break
            else:
                log.info("Invalid input. Training stopped by default.")
                break

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")
    formatted_time = f"Execution Time: {elapsed_time:.2f} seconds\n"

    # Write the elapsed time to a text file
    with open(cfg.paths.full_path+"/execution_time.txt", "a") as file:  # Use "a" to append to the file
        file.write(formatted_time)

if __name__ == "__main__":
    main()