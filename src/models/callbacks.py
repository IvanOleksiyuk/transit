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