"""A collection of misculaneous functions usefull for the lighting/hydra template."""

import logging
import os
from pathlib import Path
from typing import Any, List, Sequence

import hydra
import lightning as L
import rich
import rich.syntax
import rich.tree
import wandb
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@rank_zero_only
def reload_original_config(
    path: str = ".",
    file_name: str = "full_config.yaml",
    set_ckpt_path: bool = True,
    ckpt_flag: str = "*last*",
    set_wandb_resume: bool = True,
) -> OmegaConf:
    """Replace the cfg with the one stored at the checkpoint location.

    Will also set the chkpt_dir to the latest version of the last or best checkpoint
    """

    log.info(f"Looking for previous job config in {path}")
    orig_cfg = OmegaConf.load(Path(path, file_name))

    log.info(f"Looking for checkpoints in folder matching {ckpt_flag}")
    if set_ckpt_path:
        orig_cfg.ckpt_path = str(
            sorted(
                Path(path).glob(f"checkpoints/{ckpt_flag}.ckpt"), key=os.path.getmtime
            )[-1]
        )

    if set_wandb_resume:
        log.info("Attempting to set the same WandB ID to continue logging run")
        if hasattr(orig_cfg, "loggers"):
            if hasattr(orig_cfg.loggers, "wandb"):
                orig_cfg.loggers.wandb.resume = True

    return orig_cfg


@rank_zero_only
def print_config(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "datamodule",
        "model",
        "callbacks",
        "loggers",
        "trainer",
        "paths",
    ),
    resolve: bool = True,
) -> None:
    """Print the content of cfg using Rich library and its tree structure.

    Parameters
    ----------
        cfg:
            Configuration composed by Hydra.
        print_order:
            Determines in what order config components are printed.
        resolve:
            Whether to resolve reference fields of DictConfig.
        save_to_file:
            Whether to export config to the hydra output folder.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.insert(0, field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)


@rank_zero_only
def print_hydra_config(
    cfg: DictConfig,
    resolve: bool = True,
) -> None:
    """Print the content of cfg using Rich library and its tree structure.

    Parameters
    ----------
        cfg:
            Configuration composed by Hydra.
        resolve:
            Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)
    queue = [field for field in cfg]

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)


def save_config(cfg: OmegaConf) -> None:
    """Save the config to the output directory.

    This is necc ontop of hydra's default conf.yaml as it will resolve the entries
    allowing one to resume jobs identically with elements such as ${now:%H-%M-%S}.

    Furthermore, hydra does not allow resuming a previous job from the same dir. The
    work around is reload_original_config but that will fail as hydra overwites the
    default config.yaml file on startup, so this backup is needed for resuming.
    """

    # In order to be able to resume the wandb logger session, save the run id
    if wandb.run is not None:
        if hasattr(cfg, "loggers"):
            if hasattr(cfg.loggers, "wandb"):
                cfg.loggers.wandb.id = wandb.run.id
            else:
                log.warning("WandB is running but cant find config/loggers/wandb!")
                log.warning("This is required to save the ID for resuming jobs.")
                log.warning("Is the name of the logger set correctly")

    # save config tree to file
    OmegaConf.save(cfg, Path(cfg.paths.full_path, "full_config.yaml"), resolve=True)


@rank_zero_only
def log_hyperparameters(
    cfg: DictConfig, model: LightningModule, trainer: Trainer
) -> None:
    """Pass the config dict to the trainer's logger.

    Also calculates and logs the number of parameters
    """

    # Convert the config object to a hyperparameter dict
    hparams = OmegaConf.to_container(cfg, resolve=True)

    # calculate the number of trainable parameters in the model and add it
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    trainer.logger.log_hyperparams(hparams)


def instantiate_collection(cfg_coll: DictConfig) -> List[Any]:
    """Use hydra to instantiate a collection of classes and return a list."""
    objs = []

    if not cfg_coll:
        log.warning("List of configs is empty")
        return objs

    if not isinstance(cfg_coll, DictConfig):
        raise TypeError("List of configs must be a DictConfig!")

    for _, cb_conf in cfg_coll.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating <{cb_conf._target_}>")
            objs.append(hydra.utils.instantiate(cb_conf))

    return objs


def link_best_model(checkpoint_callback: L.pytorch.callbacks.Checkpoint, dest: str):
    """Link the best model of a checkpoint_callback to a destination path."""
    # use os.path.relpath, since Path.relative_to doesn't work, when the second path
    # is not a subpath of the first.
    best_model_path_rel = os.path.relpath(
        checkpoint_callback.best_model_path, Path(dest).parent
    )
    os.symlink(best_model_path_rel, dest)
