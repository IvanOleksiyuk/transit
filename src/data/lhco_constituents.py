from libs_snap.anomdiff.src.datamodules.cnst_lhco import load_data, train_valid_split, get_cut_mask
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from functools import partial
from typing import Literal, Mapping
from copy import deepcopy
import pandas as pd
from pathlib import Path

class LHCOhlvDatasetTT(Dataset):
    """Only stores the high level features and the mjj."""

    def __init__(
        self,
        bkg_path: str,
        sig_path: str | None = None,
        m_add_path: str | None = None,
        n_bkg: int | None = None,
        n_sig: int | None = None,
        n_csts: int | None = None,
        mjj_window: tuple | list | None = ((2700, 3300), (3700, 6000)),
    ) -> None:
        super().__init__()

        # Load the data
        hlv1, hlv2, _jet1, _jet2, label, mjj = load_data(
            bkg_path, sig_path, n_bkg, n_sig, 0, mjj_window
        )

        # Add a tiny amount of noise to the input number of constituents (dequant)
        hlv1[:, -1] += np.random.randn(len(hlv1))
        hlv2[:, -1] += np.random.randn(len(hlv2))
        self.label = label[..., None]

        # We are generating the joint high level variabels
        self.hlv = np.concatenate([hlv1, hlv2], axis=-1)
        self.mjj = mjj[..., None]  # Must be Bx1 dimension
        
        mjj_add = pd.read_hdf(m_add_path, "mass").to_numpy(np.float32)
        if len(mjj_window) == 1:
            cut = get_cut_mask(mjj_add, [[mjj_window[0][0], mjj_window[0][1]]])
        else:
            cut = get_cut_mask(mjj_add, [[mjj_window[0][0], mjj_window[1][1]]])
        mjj_add = mjj_add[cut.flatten()]
        np.random.shuffle(mjj_add)
        self.mjj_add = mjj_add[:len(mjj)]
        self.data={}
        self.data["data"] = pd.DataFrame(np.concatenate([self.hlv, self.mjj, self.label], -1), columns=["pt1", "eta1", "phi1", "m1", "Ncons1", "pt2", "eta2", "phi2", "m2", "Ncons2", "mjj", "is_signal"])
    def __len__(self) -> int:
        return len(self.mjj)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.hlv[index], self.mjj[index], self.mjj_add[index]

    def write_npy_single(self, file_path_str: str, key):
        filepath = Path(file_path_str)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(file_path_str, self.data[key].to_numpy())
    
    def write_features_txt(self, file_path_str, key):
        filepath = Path(file_path_str)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path_str, "w") as f:
            for feature in self.data[key].columns.tolist():
                f.write("%s " % feature)

class LHCOhlvDatasetTT_export(Dataset):
    """Only stores the high level features and the mjj."""

    def __init__(
        self,
        bkg_path: str,
        sig_path: str | None = None,
        m_add_path: str | None = None,
        n_bkg: int | None = None,
        n_sig: int | None = None,
        n_csts: int | None = None,
        mjj_window: tuple | list | None = ((2700, 3300), (3700, 6000)),
        oversample_x_sr = 5,
        oversample_x_sb = None,
    ) -> None:
        super().__init__()

        if n_sig >= 0:
            # Load the data
            hlv1, hlv2, _jet1, _jet2, _label, mjj = load_data(
                bkg_path, sig_path, n_bkg, n_sig, 0, mjj_window
            )
        elif n_sig < 0:
            # Load the data
            hlv1, hlv2, _jet1, _jet2, _label, mjj = load_data(
                bkg_path, sig_path, n_bkg, None, 0, mjj_window
            )
            hlv1 = hlv1[-n_sig:]
            hlv2 = hlv2[-n_sig:]
            mjj = mjj[-n_sig:]

        # Add a tiny amount of noise to the input number of constituents (dequant)
        hlv1[:, -1] += np.random.randn(len(hlv1))
        hlv2[:, -1] += np.random.randn(len(hlv2))

        # We are generating the joint high level variabels
        self.hlv = np.concatenate([hlv1, hlv2], axis=-1)
        self.mjj = mjj[..., None]  # Must be Bx1 dimension
        
        mjj_add = pd.read_hdf(m_add_path).to_numpy(np.float32)
        cut = get_cut_mask(mjj_add, [[mjj_window[0][1], mjj_window[1][0]]])
        self.mjj_add = mjj_add[cut.flatten()]
        print(f"Number of events in the SR: {len(self.mjj_add)}")
        
        # Count how many we need
        if oversample_x_sr is not None:
            needed = len(self.mjj_add) * oversample_x_sr
        elif oversample_x_sb is not None:
            needed = len(self.mjj) * oversample_x_sb
        else:
            needed = len(self.mjj_add)
        
        # Oversample each array if needed
        if needed > len(self.mjj_add):
            np.random.seed(seed=1)
            self.mjj_add = self.mjj_add[np.random.choice(self.mjj_add.shape[0], needed, replace=True)]
        else:
            self.mjj_add = self.mjj_add[:needed]
            
        if needed > len(self.mjj):
            np.random.seed(seed=2)
            ch = np.random.choice(self.mjj.shape[0], needed, replace=True)
            self.mjj = self.mjj[ch]
            self.hlv = self.hlv[ch]
        else:
            self.mjj = self.mjj[:needed]
            self.hlv = self.hlv[:needed]

    def __len__(self) -> int:
        return len(self.mjj)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.hlv[index], self.mjj[index], self.mjj_add[index]

class LHCOhlvModuleTT(LightningDataModule):
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
        return [10], [1]

    def get_var_group_list(self):
        return [["pt1", "eta1", "phi1", "m1", "Ncons1", "pt2", "eta2", "phi2", "m2", "Ncons2"], ["mjj"], ["mjj"]]

class LHCOLowDatasetTT(Dataset):
    """Combines the jet information to train on both independantly."""

    def __init__(
        self,
        bkg_path: str,
        sig_path: str | None = None,
        n_bkg: int | None = None,
        n_sig: int | None = None,
        n_csts: int | None = None,
        mjj_window: tuple | list | None = ((2700, 3300), (3700, 6000)),
        no_mask = False,
        no_N = False,
        hlvs = None,
    ) -> None:
        super().__init__()

        # Load the data
        hlv1, hlv2, jet1, jet2, _label, _mjj = load_data(
            bkg_path, sig_path, n_bkg, n_sig, n_csts, mjj_window
        )

        # Combine the leading and subleading jets into one array
        if hlvs is not None:
            if len(hlvs)==0:
                hlv1 = hlv1[:]*0
                hlv2 = hlv2[:]*0
            else:
                hlv1 = hlv1[:, np.array(hlvs)]
                hlv2 = hlv2[:, np.array(hlvs)]
        else:
            if no_N:
                hlv1 = hlv1[:, :-1]
                hlv2 = hlv2[:, :-1]
        
        self.hlv = np.concatenate([hlv1, hlv2])
        self.jet = np.concatenate([jet1, jet2])
        if not no_mask:
            self.mask = np.any(self.jet, axis=-1)
        else:
            self.mask = np.any(self.jet>-10000000, axis=-1)

    def __len__(self) -> int:
        return len(self.mask)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.jet[index], self.mask[index], self.hlv[index], self.hlv[index]

class LHCOLowDatasetTT_export(Dataset):
    """Combines the jet information to train on both independantly."""

    def __init__(
        self,
        bkg_path: str,
        sig_path: str | None = None,
        hlv_gen_path: str | None = None,
        n_bkg: int | None = None,
        n_sig: int | None = None,
        n_csts: int | None = None,
        mjj_window: tuple | list | None = ((2700, 3300), (3700, 6000)),
        no_mask = False,
        no_N = False,
        hlvs = None,
    ) -> None:
        super().__init__()

        # Load the data
        hlv1, hlv2, jet1, jet2, _label, _mjj = load_data(
            bkg_path, sig_path, n_bkg, n_sig, n_csts, mjj_window
        )

        # Combine the leading and subleading jets into one array
        if hlvs is not None:
            if len(hlvs)==0:
                hlv1 = hlv1[:]*0
                hlv2 = hlv2[:]*0
            else:
                hlv1 = hlv1[:, np.array(hlvs)]
                hlv2 = hlv2[:, np.array(hlvs)]
        else:
            if no_N:
                hlv1 = hlv1[:, :-1]
                hlv2 = hlv2[:, :-1]

        hlv_gen_df = pd.read_hdf(hlv_gen_path).to_numpy(np.float32)
        mjj_add = hlv_gen_df[:, -1]
        hlv1_gen = hlv_gen_df[:, :5]
        hlv2_gen = hlv_gen_df[:, 5:10]

        needed = len(mjj_add)
        if needed > len(hlv1):
            np.random.seed(seed=2)
            ch = np.random.choice(hlv1.shape[0], needed, replace=True)
            self.hlv1 = hlv1[ch]
            self.hlv2 = hlv2[ch]
            self.jet1 = jet1[ch]
            self.jet2 = jet2[ch]
        else:
            self.hlv1 = hlv1[:needed]
            self.hlv2 = hlv2[:needed]
            self.jet1 = jet1[:needed]
            self.jet2 = jet2[:needed]
            
        self.hlv = np.concatenate([self.hlv1, self.hlv2])
        self.hlv_gen = np.concatenate([hlv1_gen, hlv2_gen])
        self.jet = np.concatenate([self.jet1, self.jet2])
        if not no_mask:
            self.mask = np.any(self.jet, axis=-1)
        else:
            self.mask = np.any(self.jet>-10000000, axis=-1)

    def __len__(self) -> int:
        return len(self.mask)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.jet[index], self.mask[index], self.hlv[index], self.hlv_gen[index]

class LHCOLowModuleTT(LightningDataModule):
    """Datamodule for the LHCO dataset for training on sidebands."""

    def __init__(
        self,
        *,
        dataset: partial,
        loader_kwargs: Mapping,
        val_frac: float = 0.1,
        no_N = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.no_N =  no_N

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
        if self.no_N:
            return [3], [4], [4]
        else:
            return [3], [5], [5]

    def get_var_group_list(self):
        return None

