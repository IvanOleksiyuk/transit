from libs_snap.anomdiff.src.datamodules.cnst_lhco import load_data, train_valid_split, built_kfold_datasets, get_cut_mask
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from functools import partial
from typing import Literal, Mapping
from copy import deepcopy
import pandas as pd
import logging
import h5py

log = logging.getLogger(__name__)

def load_raw_file(
    file_path: str,
    mjj_window: tuple | list | None = None,
    n_events: int | None = None,
    n_csts: int | None = None,
) -> tuple:
    mjj = pd.read_hdf(file_path, key="m_jj").to_numpy(np.float32)[:n_events]
    cut = get_cut_mask(mjj, mjj_window)
    events = pd.read_hdf(file_path, key="events").to_numpy(np.float32)[:n_events, :n_csts][cut]
    mjj = mjj[cut]

    return events, mjj

def load_raw_info_for_discriminator(
    bkg_path: str,
    sig_path: str,
    tem_path: str | None = None,
    n_bkg: int | None = None,
    n_sig: int | None = 3000,
    n_tem: int | None = None,
    n_csts: int | None = 128,
    mjj_window: tuple | list | None = (3300, 3700),
    order_jets: bool = False,
    mode: Literal["standard", "supervised", "idealised"] = "supervised",
) -> tuple:
    # Load all of the data (hlv1, hlv2, jet1, jet2, mjj)
    log.info("Loading all datasets")
    bkg_data = load_raw_file(bkg_path, mjj_window, n_bkg, n_csts)
    sig_data = load_raw_file(sig_path, None, None, n_csts)  # All-split first

    # Load the template data from somewhere. This will be your negative class
    if mode == "standard":
        tem_data = load_raw_file(tem_path, mjj_window, n_tem, n_csts, order_jets)
    elif mode == "idealised":
        bkg_data = [np.array_split(x, 2, axis=0) for x in bkg_data]
        tem_data = [s[0] for s in bkg_data]
        bkg_data = [s[1] for s in bkg_data]
    elif mode == "supervised":  # Empty tensor with right shape but no length
        tem_data = tuple(np.zeros((0, *s.shape[1:]), s.dtype) for s in sig_data)

    # Append the labels 0=background, 1=signal, -1=template
    bkg_data += (np.zeros((len(bkg_data[0]), 1), dtype=np.int64),)
    sig_data += (np.ones((len(sig_data[0]), 1), dtype=np.int64),)
    if mode == "idealised":
        tem_data += (np.zeros((len(tem_data[0]), 1), dtype=np.int64),)
    else:
        tem_data += (-1 * np.ones((len(tem_data[0]), 1), dtype=np.int64),)

    # Append the training targets, signal and template are always the same
    sig_data += (np.ones((len(sig_data[0]), 1), dtype=np.float32),)
    tem_data += (np.zeros((len(tem_data[0]), 1), dtype=np.float32),)
    if mode in ["standard", "idealised"]:
        bkg_data += (np.ones((len(bkg_data[0]), 1), dtype=np.float32),)
    elif mode == "supervised":
        bkg_data += (np.zeros((len(bkg_data[0]), 1), dtype=np.float32),)

    # Filter the signal into what we can train on and what is only for testing
    log.info("Splitting signal into fit and test")
    sig_data = [np.split(x, [n_sig], axis=0) for x in sig_data]
    sig_fit = [s[0] for s in sig_data]
    sig_tst = [s[1] for s in sig_data]

    # Apply the mass cut on the signal data
    cut = get_cut_mask(sig_fit[4], mjj_window)
    sig_fit = [s[cut] for s in sig_fit]
    cut = get_cut_mask(sig_tst[4], mjj_window)
    sig_tst = [s[cut] for s in sig_tst]

    log.info("Dataset sizes:")
    log.info(f"Background:    {len(bkg_data[0])}")
    log.info(f"Signal (Fit):  {len(sig_fit[0])}")
    log.info(f"Signal (Test): {len(sig_tst[0])}")
    log.info(f"Template:      {len(tem_data[0])}")

    # Merge the background, signal (fit), and template into a single set for fitting
    log.info("Merging sources into single dataset")
    fit_set = [np.concatenate(s) for s in zip(bkg_data, sig_fit, tem_data)]

    return fit_set, sig_tst

class LHCORawClassDataset(Dataset):
    """Keeps the jets seperate for the classifier outputs."""

    def __init__(
        self,
        events: np.ndarray,
        mjj: np.ndarray,
        label: np.ndarray,
        target: np.ndarray,
    ):
        super().__init__()
        self.events = events
        self.mjj = mjj
        self.label = label
        self.target = target

        # Get the masks based on the pt
        self.mask1 = self.events[..., 0] > 0
        self.mjj = self.mjj[..., None]  # Must be Bx1 dimension

        # Calculate the weights for the positive class in the loss function
        self.pos_weight = np.reciprocal(np.mean(self.target)) - 1
        self.pos_weight = self.pos_weight[..., None]  # 1 dimension
        log.info(f"Setting pos weight to {self.pos_weight}")

    def __len__(self) -> int:
        return len(self.hlv1)

    def __getitem__(self, index: int) -> np.ndarray:
        return (
            (
                self.events[index],
            ),
            self.target[index],
            self.pos_weight,
            self.label[index],
        )

class LHCORawClassModule(LightningDataModule):
    """Datamodule for the LHCO dataset for training on sidebands."""

    def __init__(
        self,
        *,
        load_fn: partial,
        num_folds: int = 5,
        fold_idx: int = 0,
        loader_kwargs: Mapping,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.num_folds = num_folds
        self.fold_idx = fold_idx

        # Load all the data immediately as this takes by far the longest time
        self.fit_set, self.sig_tst = load_fn()

    def setup(self, stage: str) -> None:
        """Set up the relevant datasets."""

        # Do the splitting
        train, valid, test = built_kfold_datasets(
            self.fit_set, self.sig_tst, self.num_folds, self.fold_idx
        )

        # Build the datasets
        self.train_set = LHCORawClassDataset(*train)
        self.valid_set = LHCORawClassDataset(*valid)
        self.test_set = LHCORawClassDataset(*test)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, **self.hparams.loader_kwargs, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        val_kwargs = deepcopy(self.hparams.loader_kwargs)
        val_kwargs["drop_last"] = False
        return DataLoader(self.valid_set, **val_kwargs, shuffle=True)

    def test_dataloader(self) -> DataLoader:
        test_kwargs = deepcopy(self.hparams.loader_kwargs)
        test_kwargs["drop_last"] = False
        return DataLoader(self.test_set, **test_kwargs, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_dims(self) -> tuple:
        return 3, 1

class LHCORawDatasetTT(Dataset):
    """Combines the jet information to train on both independantly."""

    def __init__(
        self,
        LHCO_path: str,
        m_jj_path: str,
        n_bkg: int | None = None,
        n_sig: int | None = 3000,
        n_csts: int | None = 700,
        mjj_window: tuple | list | None = ((2700, 3300), (3700, 6000)),
    ) -> None:
        super().__init__()

        # Load the data
        events = pd.read_hdf(LHCO_path)
        self.is_signal = events.iloc[:, -1].to_numpy()
        self.events_all = events.to_numpy(dtype=np.float32)[:, :2100].reshape(-1, 700, 3)

        # Simple energy rescaling
        self.events_all[:, :, 0] = np.sqrt(self.events_all[:, :, 0])
        #select upt to 400 constituents per event
        self.events_all = self.events_all[:, :n_csts, :]
        
        masses = pd.read_hdf(m_jj_path)
        self.mjj = masses.to_numpy(dtype=np.float32)
        
        # Select correct signal doping 
        self.events_bkg = self.events_all[self.is_signal == 0]
        self.events_sig = self.events_all[self.is_signal == 1]
        self.mjj_bkg = self.mjj[self.is_signal == 0]
        self.mjj_sig = self.mjj[self.is_signal == 1]
        self.events = np.concatenate([self.events_bkg[:n_bkg], self.events_sig[:n_sig]], axis=0)
        self.masses = np.concatenate([self.mjj_bkg[:n_bkg], self.mjj_sig[n_bkg:n_sig]], axis=0)
        
        self.lowerSB_ids = np.where((self.masses > mjj_window[0][0]) & (self.masses < mjj_window[0][1]))[0]
        self.higherSB_ids = np.where((self.masses > mjj_window[1][0]) & (self.masses < mjj_window[1][1]))[0]
        self.events = np.concatenate([self.events[self.lowerSB_ids], self.events[self.higherSB_ids]], axis=0)
        self.masses = np.concatenate([self.masses[self.lowerSB_ids], self.masses[self.higherSB_ids]], axis=0)
        
        # Combine the leading and subleading jets into one array
        self.mask = np.any(self.events, axis=-1)


    def __len__(self) -> int:
        return len(self.mask)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.events[index], self.mask[index], self.mjj[index], self.mjj[index]
    
class LHCORawModuleTT(LightningDataModule):
    """Datamodule for the LHCO dataset for training on sidebands."""

    def __init__(
        self,
        *,
        dataset: partial,
        loader_kwargs: Mapping,
        val_frac: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str) -> None:
        """Set up the relevant datasets."""

        # Load the full sideband data
        full_set = self.hparams.dataset()

        self.train_set, self.valid_set = train_valid_split(
            full_set, self.hparams.val_frac
        )
        self.test_set = full_set  # For our generative models we give the full set

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, **self.hparams.loader_kwargs, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        val_kwargs = deepcopy(self.hparams.loader_kwargs)
        val_kwargs["drop_last"] = False
        return DataLoader(self.valid_set, **val_kwargs, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        test_kwargs = deepcopy(self.hparams.loader_kwargs)
        test_kwargs["drop_last"] = False
        return DataLoader(self.test_set, **test_kwargs, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_dims(self) -> tuple:
        return 3, 1

    def get_var_group_list(self):
        return ["constituents", "mjj", "mjj"]

