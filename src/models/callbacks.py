import os
import pytorch_lightning as pl

class SaveLastEveryEpoch(pl.Callback):
    def __init__(self, dirpath: str, filename: str = "last.ckpt"):
        self.dirpath = dirpath
        self.filename = filename

    def on_train_epoch_end(self, trainer, pl_module):
        os.makedirs(self.dirpath, exist_ok=True)
        path = os.path.join(self.dirpath, self.filename)
        trainer.save_checkpoint(path)
        print(f"Saved checkpoint to {path}")

class ThresholdEarlyStopping(pl.Callback):
    """Stop training once a validation metric has stayed above a threshold (2026-10-29).

    Meant for valid\\d_loss_gen, the reconstruction discriminator's validation loss: it starts at ln 2 = 0.6931 and
    rises above it when the discriminator memorises its training events (the small high-mass windows), while the training
    loss falls. Guards against stopping in the first phase, where the validation loss can also move a lot:
      min_steps  no stop before this optimiser step (counter kept at 0 until then)
      patience   number of consecutive validations above `threshold` needed to stop
    The check runs at the end of every validation (sanity checks excluded); trainer.should_stop ends the training after
    the current epoch, so the epoch-end checkpoint (SaveLastEveryEpoch, last.ckpt) is the stopped model.
    `enabled: false` makes it a no-op.
    """

    def __init__(self, monitor: str = "valid\\d_loss_gen", threshold: float = 0.6950, min_steps: int = 10000,
                 patience: int = 5, enabled: bool = True):
        super().__init__()
        self.monitor = monitor
        self.threshold = threshold
        self.min_steps = min_steps
        self.patience = patience
        self.enabled = enabled
        self.count = 0

    def on_validation_end(self, trainer, pl_module):
        if not self.enabled or trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get(self.monitor)
        if value is None:
            return
        value = float(value)
        if trainer.global_step < self.min_steps:
            self.count = 0
            return
        if value > self.threshold:
            self.count += 1
        else:
            self.count = 0
        if self.count >= self.patience:
            trainer.should_stop = True
            print(f"ThresholdEarlyStopping: {self.monitor} = {value:.5f} > {self.threshold} for {self.count} "
                  f"consecutive validations at step {trainer.global_step}: stopping")


class DLossGapEarlyStopping(pl.Callback):
    """Stop training once the reconstruction discriminator's validation - training loss gap stays large (2026-10-29).

    gap = valid\\d_loss_gen - (mean training d_loss_gen since the previous validation), each averaged over the last
    `smooth` validations. The gap grows when the discriminator memorises its training events: on the release_v2 3 %
    data it stays ~0.0003 in the large low-mass windows and reaches 0.004-0.006 in 112-309 by the end of training;
    gap > 0.003 for 5 validations stops all 3v4 trainings that fail closure and 3 of 27 that pass (replay from Comet,
    user/studies/release_v2_plots/dloss_es_replay/). Guards against the first phase, where the gap can spike:
      min_batches  no stop before this many training batches (the step axis of the training-curve plots; Lightning's
                   global step counts both optimisers and is ~2x larger)
      patience     consecutive validations with the smoothed gap above `threshold` needed to stop
    The training values are summed on the device (no host sync per batch). trainer.should_stop ends the training after
    the current epoch, so the epoch-end checkpoint (SaveLastEveryEpoch, last.ckpt) is the stopped model.
    `enabled: false` makes it a no-op.
    """

    def __init__(self, valid_metric: str = "valid\\d_loss_gen", train_metric: str = "d_loss_gen",
                 threshold: float = 0.003, min_batches: int = 20000, patience: int = 5, smooth: int = 5,
                 enabled: bool = True):
        super().__init__()
        self.valid_metric = valid_metric
        self.train_metric = train_metric
        self.threshold = threshold
        self.min_batches = min_batches
        self.patience = patience
        self.smooth = smooth
        self.enabled = enabled
        self.n_batches = 0
        self.count = 0
        self._train_sum = None
        self._train_n = 0
        self._valid_hist = []
        self._train_hist = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.enabled:
            return
        self.n_batches += 1
        value = trainer.callback_metrics.get(self.train_metric)
        if value is None:
            return
        value = value.detach()
        if self._train_sum is None:
            self._train_sum = value.clone()
        else:
            self._train_sum = self._train_sum + value
        self._train_n += 1

    def on_validation_end(self, trainer, pl_module):
        if not self.enabled or trainer.sanity_checking:
            return
        valid = trainer.callback_metrics.get(self.valid_metric)
        if valid is None or self._train_n == 0:
            return
        train_mean = float(self._train_sum) / self._train_n
        self._train_sum = None
        self._train_n = 0
        self._valid_hist = (self._valid_hist + [float(valid)])[-self.smooth:]
        self._train_hist = (self._train_hist + [train_mean])[-self.smooth:]
        gap = sum(self._valid_hist) / len(self._valid_hist) - sum(self._train_hist) / len(self._train_hist)
        if trainer.logger is not None:  # self.log() is not allowed in on_validation_end
            trainer.logger.log_metrics({"es/dloss_gap": gap}, step=trainer.global_step)
        if self.n_batches < self.min_batches:
            self.count = 0
            return
        if gap > self.threshold:
            self.count += 1
        else:
            self.count = 0
        if self.count >= self.patience:
            trainer.should_stop = True
            print(f"DLossGapEarlyStopping: smoothed gap {gap:.5f} > {self.threshold} for {self.count} consecutive "
                  f"validations at batch {self.n_batches} (global step {trainer.global_step}): stopping")
