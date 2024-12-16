import copy
import operator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import pickle
from transit.src.data.lhco_curtains import convert_lhco_to_curtain_format
from transit.src.data.cathode_preprocessing import CathodePreprocess
import torch
import os 

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

##############################################
# Here data is always considered to be a list of pandas dataframes other

##############################################
# Collection of functions for data pre-processing
##############################################

class ProcessorSubsample:
    def __init__(self, n_samples, random_state=42, mode="random"):
        self.n_samples = n_samples
        self.random_state = random_state
        self.mode = mode

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.n_samples is not None and self.n_samples > 0:
            for key, value in data.items():
                if self.mode == "random":
                    data[key] = value.sample(
                        n=self.n_samples, random_state=self.random_state
                    )
                elif self.mode == "head":
                    data[key] = value.head(self.n_samples)
                elif self.mode == "tail":
                    data[key] = value.tail(self.n_samples)
                elif self.mode == "head_invert":
                    data[key] = value[self.n_samples :]
                elif self.mode == "tail_invert":
                    data[key] = value[: -self.n_samples]
        return data

class ProcessorIntervals:
    def __init__(self, scalar_df_name, var_name, intervals):
        self.scalar_df_name = scalar_df_name
        self.var_name = var_name
        self.intervals = intervals

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        # find the events that pass the cuts
        for interval in self.intervals:
            min_val, max_val = interval[0], interval[1]
            bool_indices_new = data[self.scalar_df_name][self.var_name] < max_val
            bool_indices_new &= data[self.scalar_df_name][self.var_name] >= min_val
            if interval == self.intervals[0]:
                bool_indices = bool_indices_new
            else:  # combine the intervals
                bool_indices |= bool_indices_new
        # apply the cuts to all the dataframes
        for key, value in data.items():
            data[key] = value[bool_indices]
        return data

class ProcessorApplyCuts:
    def __init__(self, scalar_df_name, cuts):
        self.scalar_df_name = scalar_df_name
        self.cuts = cuts

    @staticmethod
    def process_cut_string(string):
        var_name, operator_s, value_s = string.split(" ")
        return var_name, OPERATORS[operator_s], float(value_s)

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        # find the events that pass the cuts
        for cut in self.cuts:
            var_name, operator, value = self.process_cut_string(cut)
            indices = operator(data[self.scalar_df_name][var_name], value)
        # apply the cuts to all the dataframes
        for key, value in data.items():
            data[key] = value[indices]
        return data
  
class ProcessorSplitDataFrameVars:
    def __init__(self, frame_name, new_df_dict):
        self.frame_name = frame_name
        self.new_df_list = new_df_dict

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data_new = {}
        for key, value in data.items():
            if key == self.frame_name:
                for new_key, var_list in self.new_df_list.items():
                    data_new[new_key] = value[var_list]
            else:
                data_new[key] = value
        return data_new

class ProcessorToFloat32:
    def __init__(self, frame_names):
        self.frame_names = frame_names

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        for name in self.frame_names:
            data[name] = data[name].astype(np.float32)
        return data

class ProcessorShuffle():
    def __init__(self, random_state=42):
        self.random_state = random_state

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        # find the events that pass the cuts
        for key, value in data.items():
            data[key] = value.sample(
                frac=1, random_state=self.random_state
            ).reset_index(drop=True)
        return data

class ProcessorNormalize():
    def __init__(self, frame_names = None, load_normaliser_file= None, save_normaliser_file=None):
        self.frame_names = frame_names
        self.load_normaliser_file = load_normaliser_file
        self.save_normaliser_file = save_normaliser_file

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.load_normaliser_file is not None:
            with open(self.load_normaliser_file, "rb") as f:
                normaliser = pickle.load(f)
            use_loaded = True
        else:
            normaliser = {}
            use_loaded = False
        frame_names = self.frame_names
        if frame_names is None:
            if use_loaded: # if no frame names are provided but normaliser is loaded, use the loaded frame names
                frame_names = normaliser.keys()
            else: # if no frame names are provided and no normaliser is loaded, use all the frames
                frame_names = data.keys()
        for name in frame_names:
            if use_loaded:
                mean = normaliser[name]["mean"]
                std = normaliser[name]["std"]
            else:
                mean = data[name].mean()
                std = data[name].std()
            #Scale the data accordingly 
            data[name] = (data[name] - mean) / std
            normaliser.update({name: {"mean": mean, "std": std}})
        if self.save_normaliser_file is not None:
            with open(self.save_normaliser_file, "wb") as f:
                pickle.dump(normaliser, f)
        return data

class ProcessorLHCOcurtains():
    def __init__(self, frame_name):
        self.frame_name = frame_name
    
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.frame_name] = convert_lhco_to_curtain_format(data[self.frame_name])
        return data
    
class ProcessorSignalContamination():
    def __init__(self, frame_name, var_name=None, n_bkg = None, n_sig = None, invert_bkg=False, invert_sig=False):
        self.frame_name = frame_name
        if var_name is None:
            self.var_name = frame_name
        else:
            self.var_name = var_name
        self.n_bkg = n_bkg
        self.n_sig = n_sig
        self.invert_bkg = invert_bkg
        self.invert_sig = invert_sig

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        indices_bkg = data[self.frame_name].index[data[self.frame_name][self.var_name]==False].tolist()
        indices_sig = data[self.frame_name].index[data[self.frame_name][self.var_name]==True].tolist()
        if self.n_bkg is not None:
            if self.invert_bkg:
                indices_bkg_select = indices_bkg[self.n_bkg:]
            else:
                indices_bkg_select = indices_bkg[:self.n_bkg]
        else:
            indices_bkg_select = indices_bkg
        if self.n_sig is not None:
            if self.invert_sig:
                indices_sig_select = indices_sig[self.n_sig:]
            else:
                indices_sig_select = indices_sig[:self.n_sig]
        else:
            indices_sig_select = indices_sig

        indices_all = indices_bkg_select + indices_sig_select
        # apply the cuts to all the dataframes
        for key, value in data.items():
            data[key] = value.loc[indices_all]
        return data

class ProcessorRemoveFrames():
    def __init__(self, frame_names):
        self.frame_names = frame_names

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        for name in self.frame_names:
            data.pop(name)
        return data

class ProcessorCATHODE():
    def __init__(self, frame_name, save_pkl=None, load_pkl=None):
        self.frame_name = frame_name
        self.load_pkl = load_pkl
        self.save_pkl = save_pkl
        

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:

        np_data = data[self.frame_name].to_numpy()
        if self.load_pkl is not None:
            with open(self.load_pkl, "rb") as f:
                cathode_preprocessor = pickle.load(f)
            print("preprocessor is loadded from a file")
        else:
            cathode_preprocessor = CathodePreprocess()
            cathode_preprocessor.fit(torch.Tensor(np_data))
        
        column_names = data[self.frame_name].columns
        np_processed = cathode_preprocessor.transform(torch.Tensor(np_data)).numpy()
        data[self.frame_name] = pd.DataFrame(np_processed, columns=column_names)
        
        if self.save_pkl is not None:
            with open(self.save_pkl, "wb") as f:
                pickle.dump(cathode_preprocessor, f)
        for key, value in data.items():
            if key != self.frame_name:
                data[key] = value.reset_index(drop=True)
        return data

class ProcessorMergeFrames():
    def __init__(self, frame_names, new_frame_name):
        self.frame_names = frame_names
        self.new_frame_name = new_frame_name

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data_new={}
        data_new[self.new_frame_name] = pd.concat([data[name] for name in self.frame_names], axis=1)
        for key, value in data.items():
            if key not in self.frame_names:
                data_new[key] = value
        return data_new

class ProcessorMergeFrames():
    def __init__(self, frame_names, new_frame_name):
        self.frame_names = frame_names
        self.new_frame_name = new_frame_name

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data_new={}
        data_new[self.new_frame_name] = pd.concat([data[name] for name in self.frame_names], axis=1)
        for key, value in data.items():
            if key not in self.frame_names:
                data_new[key] = value
        return data_new

class ProcessorAddColumn():
    def __init__(self, frame_name, column_name, column_values):
        self.frame_name = frame_name
        self.column_name = column_name
        self.column_values = column_values

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.frame_name][self.column_name] = self.column_values
        return data

class ProcessorLoadInsertDatasets:
    def __init__(self, dataset_files, ignore_length_mismatch=False):
        self.dataset_files = dataset_files
        self.ignore_length_mismatch = ignore_length_mismatch

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        for file_path in self.dataset_files:
            with pd.HDFStore(file_path, "r") as store:
                # Iterate over all the keys (dataset names) in the file
                for key in store:
                    # Read each dataset into a pandas DataFrame
                    # and store in the dictionary
                    if not self.ignore_length_mismatch:
                        assert len(store[key]) == len(data[list(data.keys())[0]]), (
                            f"Length of the dataset {key}",
                            " is not equal to the length of the data",
                        )
                    assert (
                        key[1:] not in data
                    ), "Dataset with the same name already exists"
                    data[key[1:]] = store[key]
        return data

##############################################

class InMemoryDataFrameDictBase(Dataset):
    """Basic class for in-memory datasets.
    data is stored as a dictionary of pandas DataFrames.
    """

    def __getitem__(self, item):
        if self.list_order is None:
            if self.in_numpy:
                return {key: value[item] for key, value in self.data.items()}
            else:
                return {
                    key: value.iloc[item].to_numpy() for key, value in self.data.items()
                }
        else:
            if self.in_numpy:
                return [self.data[it][item] for it in self.list_order]
            else:
                return [self.data[it].to_numpy()[item] for it in self.list_order]

    def __len__(self):
        return len(self.data[list(self.data.keys())[0]])

    def to_np(self):
        self.data_columns = {}
        for key, value in self.data.items():
            self.data_columns[key] = list(value.columns)
            self.data[key] = value.to_numpy()
        self.in_numpy = True
    
    def to_tensor(self, device=None):
        self.data_columns = {}
        for key, value in self.data.items():
            self.data_columns[key] = list(value.columns)
            self.data[key] = torch.tensor(value.to_numpy(), dtype=torch.float32)
            if device is not None:
                self.data[key] = self.data[key].to(device)
        self.in_numpy = False
        self.in_tensor = True
    
    def to_df(self):
        for key, value in self.data.items():
            self.data[key] = pd.DataFrame(value, columns=self.data_columns[key])
        self.in_numpy = False
    
    def get_dims(self):
        if self.in_numpy:
            if self.list_order is None:
                return {key: value[0].shape for key, value in self.data.items()}
            else:
                return [self.data[elem][0].shape for elem in self.list_order]
        else:
            if self.list_order is None:
                return {
                    key: value.iloc[0].to_numpy(dtype=np.float32).shape
                    for key, value in self.data.items()
                }
            return [
                self.data[it].to_numpy(dtype=np.float32)[0].shape for it in self.list_order
            ]

    def load(self, file_path: str) -> pd.DataFrame:
        data = {}

        # Open the HDF5 file
        with pd.HDFStore(file_path, "r") as store:
            # Iterate over all the keys (dataset names) in the file
            for key in store:
                # Read each dataset into a pandas DataFrame and store in the dictionary
                data[key[1:]] = store[key]
        return data

    def save(self, file_path: str):
        with pd.HDFStore(file_path, "w") as store:
            for key, value in self.data.items():
                store.put(key, value)

    def shuffle(self, random_state=42):
        for key, value in self.data.items():
            self.data[key] = value.sample(
                frac=1, random_state=random_state
            ).reset_index(drop=True)
        print("Data shuffled, please mind all later combinations!")

    def split(self, n_train):
        """Very inefficient way to split data."""
        train_data = {key: value.iloc[:n_train] for key, value in self.data.items()}
        val_data = {key: value.iloc[n_train:] for key, value in self.data.items()}
        self.data = train_data
        rest = copy.deepcopy(self)
        rest.data = val_data
        return self, rest

    def plot(self, plot_dir):
        print(f"Plotting the data in {plot_dir}")
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        for key, value in self.data.items():
            plt.figure()
            total_events = len(value)
            if len(value) > 20000:
                sns.pairplot(value.sample(20000, random_state=42))
            else:
                sns.pairplot(value)
            plt.title(key+" distribution, total events: "+str(len(value)))
            plt.savefig(Path(plot_dir) / f"{key}.png")
            plt.close()

    def init_processors(self, processor_cfg):
        self.processors = []
        for processor in processor_cfg:
            self.processors.append(processor)

    def apply_processors(self, data):
        for processor in self.processors:
            data = processor(data)
        return data

    # Functions that you casn call if you really need them not in configs
    def merge_dataframes(self, frame_names, new_frame_name):
        self.data[new_frame_name] = pd.concat([self.data[name] for name in frame_names], axis=1)
        self.list_order.append(new_frame_name)

    def write_npy(self, file_dir: str, keys=None, save_file_names=None):
        if save_file_names is not None:
            assert len(save_file_names) == len(keys)
        os.makedirs(file_dir, exist_ok=True)
        if save_file_names is None:
            if keys is None:
                keys = self.data.keys()
            for i, key in enumerate(keys):
                np.save(file_dir + key + ".npy", self.data[key].to_numpy())
        else:
            if keys is None:
                keys = self.data.keys()
            for i, key in enumerate(keys):
                np.save(file_dir + save_file_names[i] + ".npy", self.data[key].to_numpy())

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
    
    def reset_index(self):
        for key, value in self.data.items():
            self.data[key] = value.reset_index(drop=True)

class InMemoryDataFrameDict(InMemoryDataFrameDictBase):
    """Class for in-memory datasets stored as a dictionary of pandas DataFrames.
    Loaded from an HDF5 file. Preprocessing is applied.
    """

    def __init__(self, file_path: str, processor_cfg=[], list_order=None, plotting_path= None, do_plotting=True, reset_index=False) -> None:
        self.in_numpy = False
        self.file_path = file_path
        self.list_order = list_order
        self.data = self.load(file_path)
        
        self.init_processors(processor_cfg)
        self.data = self.apply_processors(self.data)
        
        if (plotting_path is not None) and do_plotting:
            self.plot(plotting_path)
        print("Data length: ", len(self))

class InMemoryDataMerge(InMemoryDataFrameDictBase):
    """Class for in-memory datasets stored as a dictionary of pandas DataFrames.
    Merged from 2 or more InMemoryDataFrameDict objects.
    """

    def __init__(self, dataset_list, do_shuffle=False) -> None:
        self.in_numpy = False
        self.list_order = dataset_list[0].list_order
        if isinstance(self.list_order, list):
            self.list_order.append("label")
        self.data = {}
        for i, dataset in enumerate(dataset_list):
            if i == 0:
                for key, value in dataset.data.items():
                    self.data[key] = value
            else:
                for key, value in dataset.data.items():
                    self.data[key] = pd.concat([self.data[key], value])
        if do_shuffle:
            self.shuffle()
        print("Data length: ", len(self))

class InMemoryDataMergeClasses(InMemoryDataFrameDictBase):
    """Class for in-memory datasets stored as a dictionary of pandas DataFrames.
    Merged from 2 or more InMemoryDataFrameDict or simmilar objects.
    The lable df is added to them with class_lables.
    """

    def __init__(
        self, dataset_list, class_lables=None, do_shuffle=True, sample_to_min=True, plotting_path= None, processor_cfg=[], 
    ) -> None:
        self.in_numpy = False
        self.list_order = dataset_list[0].list_order
        if isinstance(self.list_order, list):
            self.list_order.append("label")
        if class_lables is None:
            class_lables = list(range(len(dataset_list)))
        self.data = {}
        if sample_to_min:
            min_len = min([len(dataset) for dataset in dataset_list])
            for i, dataset in enumerate(dataset_list):
                if i == 0:
                    for key, value in dataset.data.items():
                        self.data[key] = value.iloc[:min_len]
                    self.data["label"] = pd.DataFrame(
                        np.array([class_lables[i]] * min_len).reshape(-1, 1),
                        columns=["label"],
                    )
                else:
                    for key, value in dataset.data.items():
                        self.data[key] = pd.concat([
                            self.data[key],
                            value.iloc[:min_len],
                        ])
                    self.data["label"] = pd.concat([
                        self.data["label"],
                        pd.DataFrame(
                            np.array([class_lables[i]] * min_len).reshape(-1, 1),
                            columns=["label"],
                        ),
                    ])
        else:
            for i, dataset in enumerate(dataset_list):
                if i == 0:
                    for key, value in dataset.data.items():
                        self.data[key] = value
                    self.data["label"] = pd.DataFrame(
                        np.array([class_lables[i]] * len(value)).reshape(-1, 1),
                        columns=["label"],
                    )
                else:
                    for key, value in dataset.data.items():
                        self.data[key] = pd.concat([self.data[key], value])
                    self.data["label"] = pd.concat([
                        self.data["label"],
                        pd.DataFrame(
                            np.array([class_lables[i]] * len(value)).reshape(-1, 1),
                            columns=["label"],
                        ),
                    ])
        if do_shuffle:
            self.shuffle()

        self.init_processors(processor_cfg)
        self.data = self.apply_processors(self.data)

        if plotting_path is not None:
            self.plot(plotting_path)
        print("Data length: ", len(self))
    
class SimpleDataModule(LightningDataModule):
    def __init__(
        self,
        loader_kwargs,
        train_data = None,
        val_data = None,
        train_frac: float = 0.8,
        test_data=None,
        to_np=False,
    ) -> None:
        super().__init__()
        self.loader_kwargs = loader_kwargs
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        if train_data is None:
            print("No training dataset config")
        else:
            print("Training dataset loaded using cofig")

        if val_data is not None:
            print("Validation dataset loaded using cofig")
        elif self.train_data is not None:
            if train_frac is not None:
                n_data = len(self.train_data)
                n_train = int(train_frac * n_data)
                self.train_data, self.val_data = self.train_data.split(n_train)
                print(
                    "Validation dataset is is split "
                    f"from training with fraction {1 - train_frac}"
                )
            else:
                self.val_data = None
                print("No validation dataset set")
        else:
            self.val_data = None
            print("No validation dataset set")

        if test_data is None:
            print("No test dataset set")
        else:
            print("Test dataset set from config")
        
        if to_np:
            if self.train_data is not None:
                self.train_data.to_np()
            if self.val_data is not None:
                self.val_data.to_np()
            if self.test_data is not None:
                self.test_data.to_np()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, **self.loader_kwargs, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, **self.loader_kwargs, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, **self.loader_kwargs, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_dims(self):
        return self.train_data.get_dims()

    def get_var_group_list(self):
        if self.train_data.in_numpy:
            var_group_list = []
            for key in self.train_data.list_order: 
                var_group_list.append(self.train_data.data_columns[key])  
        else:
            var_group_list = []
            for key in self.train_data.list_order:
                var_group_list.append(self.train_data.data[key].columns.tolist())
        return var_group_list

class CombDataset(InMemoryDataFrameDictBase):
    def __init__(self, dataset1, dataset2, length=None, oversample1=None, oversample2=None, seed=42, plotting_path=None) -> None:
        self.in_numpy = False
        if oversample1 is not None:
            length = oversample1 * len(dataset1)
        if oversample2 is not None:
            length = oversample2 * len(dataset2)
        if length is None:
            length = min(len(dataset1), len(dataset2))
        self.data = {}
        for key, value in dataset1.data.items():
            if len(dataset1)>=length:
                self.data[key]=value[:length]
            else:
                self.data[key]=value.sample(n=length, replace=True, random_state=seed).reset_index(drop=True)
        for key, value in dataset2.data.items():
            if len(dataset2)>=length:
                self.data[key]=value[:length]
            else:
                self.data[key]=value.sample(n=length, replace=True, random_state=seed).reset_index(drop=True)
        self.list_order = dataset1.list_order+dataset2.list_order
        if plotting_path is not None:
            self.plot(plotting_path)