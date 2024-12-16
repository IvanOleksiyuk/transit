from copy import deepcopy
from typing import Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import os
import functools
import operator
import copy

INVERTED_KEYS = [
    "pxj2",
    "pyj2",
    "pzj2",
    "mj2",
    "tau1j2",
    "tau2j2",
    "tau3j2",
    "pxj1",
    "pyj1",
    "pzj1",
    "mj1",
    "tau1j1",
    "tau2j1",
    "tau3j1",
    "label",
]


def convert_lhco_to_curtain_format(df=pd.DataFrame) -> pd.DataFrame:
    """Return a new dataframe with all variables needed by the curtains project."""

    # Peform reordering such that mj1 is the smaller of the two jets
    jet_order_mask = df["mj1"] < df["mj2"]
    proper_order = df.loc[jet_order_mask]
    improper_order = df.loc[~jet_order_mask]
    improper_order.columns = INVERTED_KEYS
    df = pd.concat((proper_order, improper_order))

    data = pd.DataFrame()
    data["is_signal"] = df["label"].astype("bool")

    # Individual jet kinematics
    for jet in ["j1", "j2"]:
        data[f"px_{jet}"] = df[f"px{jet}"]
        data[f"py_{jet}"] = df[f"py{jet}"]
        data[f"pz_{jet}"] = df[f"pz{jet}"]
        data[f"m_{jet}"] = df[f"m{jet}"]

        data[f"pt_{jet}"] = np.sqrt(data[f"px_{jet}"] ** 2 + data[f"py_{jet}"] ** 2)
        data[f"phi_{jet}"] = np.arctan2(data[f"py_{jet}"], data[f"px_{jet}"])
        data[f"eta_{jet}"] = np.arcsinh(data[f"pz_{jet}"] / data[f"pt_{jet}"])
        data[f"p_{jet}"] = np.sqrt(data[f"pz_{jet}"] ** 2 + data[f"pt_{jet}"] ** 2)
        data[f"e_{jet}"] = np.sqrt(data[f"m_{jet}"] ** 2 + data[f"p_{jet}"] ** 2)

    # Combined jet mass
    data["m_jj"] = calculate_mass(
        np.sum(
            [
                data[[f"e_j{i}", f"px_j{i}", f"py_j{i}", f"pz_j{i}"]].to_numpy()
                for i in range(1, 3)
            ],
            0,
        )
    )

    # Subjettiness ratios
    data["del_m"] = data["m_j2"] - data["m_j1"]
    data["tau21_j1"] = df["tau2j1"] / df["tau1j1"]
    data["tau32_j1"] = df["tau3j1"] / df["tau2j1"]
    data["tau21_j2"] = df["tau2j2"] / df["tau1j2"]
    data["tau32_j2"] = df["tau3j2"] / df["tau2j2"]
    data["m_n"] = data["m_jj"] - data["m_j1"] - data["m_j2"] # Fake variable delete as soon as possible
    
    # Other variables
    phi_1 = data["phi_j1"]
    phi_2 = data["phi_j2"]
    delPhi = np.arctan2(np.sin(phi_1 - phi_2), np.cos(phi_1 - phi_2))
    data["del_R"] = ((data["eta_j1"] - data["eta_j2"]) ** 2 + delPhi**2) ** (0.5)
    data["del_phi"] = abs(delPhi)
    data["del_eta"] = abs(data["eta_j1"] - data["eta_j2"])

    return data.dropna()

def convert_lhco_to_curtain_format_fakes(df=pd.DataFrame) -> pd.DataFrame:
    """Return a new dataframe with all variables needed by the curtains project."""

    # Peform reordering such that mj1 is the smaller of the two jets
    jet_order_mask = df["mj1"] < df["mj2"]
    proper_order = df.loc[jet_order_mask]
    improper_order = df.loc[~jet_order_mask]
    improper_order.columns = INVERTED_KEYS
    df = pd.concat((proper_order, improper_order))

    data = pd.DataFrame()
    data["is_signal"] = df["label"].astype("bool")

    # Individual jet kinematics
    for jet in ["j1", "j2"]:
        data[f"px_{jet}"] = df[f"px{jet}"]
        data[f"py_{jet}"] = df[f"py{jet}"]
        data[f"pz_{jet}"] = df[f"pz{jet}"]
        data[f"m_{jet}"] = df[f"m{jet}"]

        data[f"pt_{jet}"] = np.sqrt(data[f"px_{jet}"] ** 2 + data[f"py_{jet}"] ** 2)
        data[f"phi_{jet}"] = np.arctan2(data[f"py_{jet}"], data[f"px_{jet}"])
        data[f"eta_{jet}"] = np.arcsinh(data[f"pz_{jet}"] / data[f"pt_{jet}"])
        data[f"p_{jet}"] = np.sqrt(data[f"pz_{jet}"] ** 2 + data[f"pt_{jet}"] ** 2)
        data[f"e_{jet}"] = np.sqrt(data[f"m_{jet}"] ** 2 + data[f"p_{jet}"] ** 2)

    # Combined jet mass
    data["m_jj"] = calculate_mass(
        np.sum(
            [
                data[[f"e_j{i}", f"px_j{i}", f"py_j{i}", f"pz_j{i}"]].to_numpy()
                for i in range(1, 3)
            ],
            0,
        )
    )

    # Subjettiness ratios
    data["del_m"] = data["m_j2"] - data["m_j1"]
    data["tau21_j1"] = df["tau2j1"] / df["tau1j1"]
    data["tau32_j1"] = df["tau3j1"] / df["tau2j1"]
    data["tau21_j2"] = df["tau2j2"] / df["tau1j2"]
    data["tau32_j2"] = df["tau3j2"] / df["tau2j2"]


    # Other variables
    phi_1 = data["phi_j1"]
    phi_2 = data["phi_j2"]
    delPhi = np.arctan2(np.sin(phi_1 - phi_2), np.cos(phi_1 - phi_2))
    data["del_R"] = ((data["eta_j1"] - data["eta_j2"]) ** 2 + delPhi**2) ** (0.5)
    data["del_phi"] = abs(delPhi)
    data["del_eta"] = abs(data["eta_j1"] - data["eta_j2"])

    return data.dropna()

def calculate_mass(four_vector: np.ndarray) -> np.ndarray:
    """Calculate the invariant mass of a four vector."""
    return (
        np.clip(
            four_vector[:, 0] ** 2 - np.sum(four_vector[:, 1:4] ** 2, axis=1),
            0,
            None,
        )
    ) ** 0.5


OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "in": np.isin,
    "notin": lambda x, y: ~np.isin(x, y),
}

for i in range(2, 101):
    OPERATORS[f"%{i}=="] = functools.partial(lambda x, y, i: (x % i) == y, i=i)
    OPERATORS[f"%{i}!="] = functools.partial(lambda x, y, i: (x % i) != y, i=i)
    OPERATORS[f"%{i}<="] = functools.partial(lambda x, y, i: (x % i) <= y, i=i)
    OPERATORS[f"%{i}>="] = functools.partial(lambda x, y, i: (x % i) >= y, i=i)

class InMemoryDataset(Dataset):
    """Dataset for training on the side bands of the LHCO.

    Also fits a sampler for generating in the signal region.
    """

    def __init__(
        self,
        file_path: str,
        var_group_list: list,
        to_return_list: list,
        preprocessing_cfg = None,
        scaler_cfg = None,
        selection_cfg = None,
        do_plotting = True,
        plot_before_scale = None,
        plot_after_scale = None,
        do_group_split = True,
    ) -> None:
        super().__init__()

        self.file_path = file_path
        self.var_group_list = var_group_list
        self.variables = []
        for var_group in var_group_list:
            self.variables += var_group
        self.to_return_list = to_return_list

        self.data = self.load(file_path)
        self.data = self.preprocess(self.data, preprocessing_cfg)
        self.data = self.select_features(self.data)
        self.scalers = self.fit_scalers(self.data, scaler_cfg)
        self.data = self.select_data(self.data, selection_cfg)
        
        if plot_before_scale is not None:
            if do_plotting:
                self.plot_data(self.data, plot_before_scale)
        
        # Prepare data to be fed into a model 
        self.data = self.transform(self.data, self.scalers)
        if plot_after_scale is not None:
            if do_plotting:
                self.plot_data(self.data, plot_after_scale)
        if do_group_split:
            self.data = self.var_group_split(self.data, var_group_list, to_return_list)
        else:
            self.data = [self.data]
        
        
    def __getitem__(self, item):
        return [group[item] for group in self.data]
    
    def __len__(self):
        return len(self.data[0])

    def cpu(self):
        self.args = [args.cpu() for args in self.args]
    
    def var_group_split(self, data: pd.DataFrame, var_group_list, to_return_list) -> pd.DataFrame:
        """Split the data into the different variable groups."""
        data_formated = []
        for vars, to_return in zip(var_group_list, to_return_list):
            if to_return:
                data_formated.append(data[vars].to_numpy(dtype=np.float32))
        return data_formated
    
    def get_dim(self):
        return [group[0].shape for group in self.data]
    
    def load(self, file_path: str) -> pd.DataFrame:
        return pd.read_hdf(file_path)

class LHCOInMemoryDataset(InMemoryDataset):

    def preprocess(self, data: pd.DataFrame, preprocessing_cfg) -> pd.DataFrame:
        return convert_lhco_to_curtain_format(data)
    
    def select_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[self.variables]
    
    def fit_scalers(self, data: pd.DataFrame, scaler_cfg) -> pd.DataFrame:
        if hasattr(scaler_cfg, "selection_cfg"):
            data1 = data.copy()
            data1 = self.select_data(data1, scaler_cfg.selection_cfg)
        scalers = {}
        for scale, var in zip(scaler_cfg.scalers, self.variables):
            if scale=="":
                pass
            elif scale=="norm":
                scalers[var] = {"mean": data1[var].mean(), "std": data1[var].std()}
        #print(scalers)
        return scalers

    def transform(self, data: pd.DataFrame, scalers: pd.DataFrame) -> pd.DataFrame:
        for var in self.variables:
            if var in scalers:
                data[var] = (data[var] - scalers[var]["mean"]) / scalers[var]["std"]
        return data
    
    def select_data(self, data: pd.DataFrame, selection_cfg) -> pd.DataFrame:
        # Separate data into signal and background and inject signal to background if needed
        if hasattr(selection_cfg, "n_inject_signal"):
            signal_data = data[data["is_signal"]]
            bkg_data = data[np.logical_not(data["is_signal"])]
            signal_data = signal_data.sample(n=selection_cfg.n_inject_signal, replace=True)
            data = pd.concat([bkg_data, signal_data])
        
        # Apply cuts to the data
        if hasattr(selection_cfg, "cuts"):
            for cut in selection_cfg.cuts:
                cut = cut.split(" ")
                data = data[OPERATORS[cut[1]](data[cut[0]], float(cut[2]))]
        
        # for more selection after preprocessing if needed
        if hasattr(selection_cfg, "n_bkg") or hasattr(selection_cfg, "n_sig"):
            bkg_data = data[np.logical_not(data["is_signal"])]
            sig_data = data[data["is_signal"]]
            if hasattr(selection_cfg, "n_bkg"):
                bkg_data = bkg_data.sample(n=selection_cfg.n_bkg)
            if hasattr(selection_cfg, "n_sig"):
                sig_data = sig_data.sample(n=selection_cfg.n_sig)
            data = pd.concat([bkg_data, sig_data])
        return data
    
    def plot_data(self, data, plot_cfg):
        os.makedirs(plot_cfg.save_path, exist_ok=True)
        for var in self.variables:
            if var in ["is_signal"]:
                pass
            else:
                plt.figure()
                plt.xlabel(var)
                plt.hist(data[var], bins=50, histtype="step", color="green", label="mixed")
                plt.hist(data[var][data["is_signal"]], bins=50, histtype="step", color="gold", label="signal")
                plt.hist(data[var][np.logical_not(data["is_signal"])], bins=50, histtype="step", color="blue", label="background")
                plt.legend()
                plt.grid()
                plt.savefig(plot_cfg.save_path+f"{var}.png")
        plt.close("all")

class ConcatDataset():
    def __init__(self, dataset_list) -> None:
        pass
    
    def __getitem__(self, item):
        return [group[item] for group in self.data]
    
    def __len__(self):
        return len(self.data[0])
    
class ShuffleCombDataset():
    def __init__(self, dataset1_cfg, dataset2_cfg, length=None) -> None:
        dataset1 = LHCOInMemoryDataset(**dataset1_cfg)
        dataset2 = LHCOInMemoryDataset(**dataset2_cfg)
        if length is None:
            length = min(len(dataset1), len(dataset2))
        shuffle_indices = np.random.permutation(length)
        self.data = []
        for group1 in dataset1.data:
            self.data.append(group1[:length])
        for group2 in dataset2.data:
            self.data.append(group2[shuffle_indices][:length])
    
    def __getitem__(self, item):
        return [group[item] for group in self.data]
    
    def __len__(self):
        return len(self.data[0])

class LHCODataModule(LightningDataModule):
    def __init__(self, 
                 loader_kwargs,
                 train_data_conf,  
                 train_frac: float = 0.8,
                 test_data_conf = None,) -> None:
        
        super().__init__()
        self.loader_kwargs = loader_kwargs
        if train_data_conf is None:
            self.train_dataset = None
        else:
            self.train_dataset =  self.instantiate_dataset(train_data_conf)
            n_data = len(self.train_dataset)
            n_train = int(train_frac * n_data)
            self.train_dataset.data = [array[:n_train] for array in self.train_dataset.data]
            self.val_dataset =  self.instantiate_dataset(train_data_conf)
            self.val_dataset.data = [array[n_train:] for array in self.val_dataset.data]
            
        if test_data_conf is None:
            self.test_dataset = self.instantiate_dataset(train_data_conf)
        else:
            self.test_dataset = self.instantiate_dataset(test_data_conf)
    
    def instantiate_dataset(self, train_data_conf):
        if hasattr(train_data_conf, "dataset1_cfg"):
            return ShuffleCombDataset(train_data_conf.dataset1_cfg, train_data_conf.dataset2_cfg)
        return LHCOInMemoryDataset(**train_data_conf)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, **self.loader_kwargs, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, **self.loader_kwargs, shuffle=False)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, **self.loader_kwargs, shuffle=False)
    
    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
    
    def get_dim(self):
        return self.train_dataset.get_dim()


if __name__ == "__main__":
    print("Running the LHCOInMemoryDataset with plots")
    LHCOInMemoryDataset(
        file_path =  "/home/users/o/oleksiyu/scratch/DATA/LHCO/events_anomalydetection_v2.features.h5",
        var_group_list = [["is_signal"], ["m_jj"]],
        to_return_list = [False, True],
        do_plot = True,
    )
    
    pass

