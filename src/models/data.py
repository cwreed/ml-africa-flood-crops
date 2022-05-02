from pathlib import Path
from typing import Optional
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.regions import STR2BB
from src.engineers.cropharvest import CropHarvestEngineer, CropHarvestDataInstance
from src.engineers.cloud_to_street_dfo import C2SDFOEngineer, C2SDFODataInstance

class CroplandClassificationDataset(Dataset):
    def __init__(
        self,
        data_folder: Path,
        subset: str
    ):
        self.data_folder = data_folder
        self.features_dir = data_folder / 'features' / CropHarvestEngineer.dataset
        
        assert subset in ['train', 'validation', 'test']
        self.subset = subset

        self.data_files, self.normalizing_dict = self.load_files(self.features_dir, self.subset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        file = self.data_files[index]

        with file.open('rb') as f:
            data_instance = pickle.load(f)
        
        assert isinstance(data_instance, CropHarvestDataInstance)
        array = self.normalize(data_instance.labeled_array)
        crop_label = data_instance.crop_label

        return torch.tensor(array), crop_label

    def __len__(self) -> int:
        return len(self.data_files)

    @staticmethod
    def load_files(features_dir: Path, subset: str) -> tuple[list[Path], Optional[dict]]:
        data_files = list((features_dir / subset).glob('*.pkl'))
        normalizing_dict_file = features_dir / 'normalizing_dict.pkl'

        if normalizing_dict_file.exists():
            with normalizing_dict_file.open('rb') as f:
                normalizing_dict = pickle.load(f)
        else:
            normalizing_dict = None
        
        return data_files, normalizing_dict
    
    def normalize(self, array: np.ndarray) -> np.ndarray:
        if self.normalizing_dict is None:
            return array
        else:
            return (
                (array - self.normalizing_dict['mean']) / self.normalizing_dict['std']
            )
    
    @property
    def num_output_classes(self) -> int:
        return 1

    @property
    def num_input_features(self) -> int:
        assert len(self.data_files) > 0, "No files loaded"
        output_tuple = self[0]
        return output_tuple[0].shape[1]

    @property
    def num_timesteps(self) -> int:
        assert len(self.data_files) > 0, "No files loaded"
        output_tuple = self[0]
        return output_tuple[0].shape[0]

class FloodClassificationDataset(Dataset):
    def __init__(
        self,
        data_folder: Path,
        subset: str
    ):
        self.data_folder = data_folder
        self.features_dir = data_folder / 'features' / C2SDFOEngineer.dataset

        assert subset in ['train', 'validation', 'test']
        self.subset = subset

        self.data_files, self.lookup, self.normalizing_dict = self.load_files(self.features_dir, self.subset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        file, file_index = self.lookup[index]

        filepath = self.features_dir / self.subset / f'{file}.pkl'

        with filepath.open('rb') as f:
            data_instance = pickle.load(f)[file_index]

        assert isinstance(data_instance, C2SDFODataInstance)
        array = self.normalize(data_instance.labeled_array)
        flood_label = data_instance.flood_label
        flood_prior = data_instance.flood_prior

        return torch.tensor(array), flood_label, flood_prior

    def __len__(self) -> int:
        return len(self.lookup)

    @staticmethod
    def load_files(features_dir: Path, subset: str) -> tuple[list[Path], list[tuple[str, int]], Optional[dict]]:
        data_files = list((features_dir / subset).glob('*.pkl'))
        lookup_file = features_dir / subset / 'lookup.ref'
        normalizing_dict_file = features_dir / 'normalizing_dict.pkl'

        assert lookup_file.exists()
        with lookup_file.open('rb') as f:
            lookup = pickle.load(f)
        
        if normalizing_dict_file.exists():
            with normalizing_dict_file.open('rb') as f:
                normalizing_dict = pickle.load(f)
        else:
            normalizing_dict = None
        
        return data_files, lookup, normalizing_dict

    def normalize(self, array: np.ndarray) -> np.ndarray:
        if self.normalizing_dict is None:
            return array
        else:
            return (
                (array - self.normalizing_dict['mean']) / self.normalizing_dict['std']
            )
    
    @property
    def num_output_classes(self) -> int:
        return 1

    @property
    def num_input_features(self) -> int:
        assert len(self.data_files) > 0, "No files loaded"
        output_tuple = self[0]
        return output_tuple[0].shape[1]

    @property
    def num_timesteps(self) -> int:
        assert len(self.data_files) > 0, "No files loaded"
        output_tuple = self[0]
        return output_tuple[0].shape[0]