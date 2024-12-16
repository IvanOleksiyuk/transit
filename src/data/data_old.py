import copy
import operator
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

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
### Collection of functions for data pre-processing
##############################################

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
            bool_indices = bool_indices_new if interval == self.intervals[0] else bool_indices | bool_indices_new
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
  
class ProcessorSplitDataFrameVars():
    def __init__(self, frame_name, new_df_dict):
        self.frame_name=frame_name
        self.new_df_list=new_df_dict
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data_new = {}
        for key, value in data.items():
            if key == self.frame_name:
                for new_key, var_list in self.new_df_list.items():
                    data_new[new_key] = value[var_list]
            else:
                data_new[key] = value
        return data_new

class ProcessorToFloat32():
    def __init__(self, frame_names):
        self.frame_names=frame_names
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        for name in self.frame_names:
            data[name] = data[name].astype(np.float32)
        return data

class ProcessorShuffle():
    def __init__(self):
        pass
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        # find the events that pass the cuts
        for key, value in data.items():
            data[key] = value.sample(frac=1).reset_index(drop=True)
        return data

##############################################

class InMemoryDataFrameDictBase(Dataset):
    """Basic class for in-memory datasets stored as a dictionary of pandas DataFrames.
    """

    def __getitem__(self, item):
        if self.list_order is None:
            return {key: value.iloc[item].to_numpy() for key, value in self.data.items()}
        else:
            return [self.data[it].to_numpy()[item] for it in self.list_order]
    
    def __len__(self):
        return len(self.data[list(self.data.keys())[0]])
    
    def get_dim(self):
        if self.list_order is None:
            return {key: value.iloc[0].to_numpy(dtype=np.float32).shape for key, value in self.data.items()}
        else:
            return [self.data[it].to_numpy(dtype=np.float32)[0].shape for it in self.list_order]
    
    def load(self, file_path: str) -> pd.DataFrame:
        data = {}
        
        # Open the HDF5 file
        with pd.HDFStore(file_path, 'r') as store:
            # Iterate over all the keys (dataset names) in the file
            for key in store.keys():
                # Read each dataset into a pandas DataFrame and store in the dictionary
                data[key[1:]] = store[key]
        return data
    
    def shuffle(self, random_state=42):
        for key, value in self.data.items():
            self.data[key] = value.sample(frac=1, random_state=random_state).reset_index(drop=True)

    def split(self, n_train):
        """ Very inefficient way to split data """
        n_data = len(self.data)
        n_val = n_data - n_train
        train_data = {key: value.iloc[:n_train] for key, value in self.data.items()}
        val_data = {key: value.iloc[n_train:] for key, value in self.data.items()}
        self.data = train_data
        rest = copy.deepcopy(self)
        rest.data = val_data
        return self, rest

    def plot_seborn(self, plot_dir):
        os.makedirs(plot_dir, exist_ok=True)
        for key, value in self.data.items():
            sns.pairplot(value)
            plt.savefig(os.path.join(plot_dir, f'{key}.png'))
            plt.close()
    
class InMemoryDataFrameDict(InMemoryDataFrameDictBase):
    """Class for in-memory datasets stored as a dictionary of pandas DataFrames. Loaded from an HDF5 file. Preprocessing is applied.
    """
    def __init__(self, file_path: str, processor_cfg, list_order=None, plotting_path=None) -> None:
        self.file_path = file_path
        self.list_order = list_order
        self.init_processors(processor_cfg)
        self.data = self.load(file_path)  
        self.data = self.apply_processors(self.data) 
        if plotting_path is not None:
            self.plot_seborn(plotting_path)   

    def init_processors(self, processor_cfg):
        self.processors = []
        for processor in processor_cfg:
            self.processors.append(processor)

    def apply_processors(self, data):
        for processor in self.processors:
            data = processor(data)
        return data

class InMemoryDataMerge(InMemoryDataFrameDictBase):
    """Class for in-memory datasets stored as a dictionary of pandas DataFrames. Merged from 2 or more InMemoryDataFrameDict objects"""
    def __init__(self, dataset_list, do_shuffle = False) -> None:
        self.list_order = dataset_list[0].list_order
        if isinstance(self.list_order, list):
            self.list_order.append("label")
        self.data = {}
        for i, dataset in enumerate(dataset_list):
            if i==0:
                for key, value in dataset.data.items():
                    self.data[key] = value
            else:
                for key, value in dataset.data.items():
                    self.data[key] = pd.concat([self.data[key], value])
        if do_shuffle:
            self.shuffle()

class InMemoryDataMergeClasses(InMemoryDataFrameDictBase):
    """Class for in-memory datasets stored as a dictionary of pandas DataFrames. Merged from 2 or more InMemoryDataFrameDict objects and lable them with class_lables"""
    def __init__(self, dataset_list, class_lables=None, do_shuffle = False, plotting_path=None) -> None:
        self.list_order = dataset_list[0].list_order
        if isinstance(self.list_order, list):
            self.list_order.append("label")
        if class_lables is None:
            class_lables = list(range(len(dataset_list)))
        self.data = {}
        for i, dataset in enumerate(dataset_list):
            if i==0:
                for key, value in dataset.data.items():
                    self.data[key] = value
                    self.data["label"] = pd.DataFrame(np.array([class_lables[i]]*len(value)).reshape(-1, 1), columns=["label"])
            else:
                for key, value in dataset.data.items():
                    self.data[key] = pd.concat([self.data[key], value])
                    self.data["label"] = pd.concat([self.data["label"], pd.DataFrame(np.array([class_lables[i]]*len(value)).reshape(-1, 1), columns=["label"])])
        if do_shuffle:
            self.shuffle()
        if plotting_path is not None:
            self.plot_seborn(plotting_path)   
                    
class SimpleDataModule(LightningDataModule):
    def __init__(self, 
                 loader_kwargs,
                 train_data = None,  
                 val_data = None,
                 train_frac: float = 0.8,
                 test_data = None,) -> None:
        
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
        else:
            if self.train_data is not None:
                if train_frac is not None:
                    n_data = len(self.train_data)
                    n_train = int(train_frac * n_data)
                    self.train_data, self.val_data = self.train_data.split(n_train)
                    print(f"Validation dataset is is split from training with fraction {1-train_frac}")
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
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, **self.loader_kwargs, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, **self.loader_kwargs, shuffle=False)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, **self.loader_kwargs, shuffle=False)
    
    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
    
    def get_dim(self):
        return self.train_data.get_dim()
    
    def get_var_group_list(self):
        var_group_list = []
        for key in self.train_data.list_order:
            var_group_list.append(self.train_data.data[key].columns.tolist())
        return var_group_list
    
class CombDataset(InMemoryDataFrameDictBase):
    def __init__(self, dataset1, dataset2, length=None, plotting_path=None) -> None:
        if length is None:
            length = min(len(dataset1), len(dataset2))
        self.data = {}
        for key, value in dataset1.data.items():
            self.data[key]=value[:length]
        for key, value in dataset2.data.items():
            self.data[key]=value[:length]
        self.list_order = dataset1.list_order+dataset2.list_order
        if plotting_path is not None:
            self.plot_seborn(plotting_path)
