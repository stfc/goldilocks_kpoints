import numpy as np
import pandas as pd
import torch
import pytorch_lightning as L
import os
import re
import json
from pymatgen.core.structure import Structure
from utils.utils import normalize_formulas
from sklearn.model_selection import train_test_split
from utils.crabnet_utils import get_edm, EDMDataset
from torch.utils.data import DataLoader

data_type_np = np.float32
data_type_torch = torch.float32

class CrabNetDataModule(L.LightningDataModule):
    def __init__(self, root_dir: str,
                 id_prop_csv: str,
                 train_ratio = 0.8,
                 val_ratio = 0.1,
                 test_ratio = 0.1,
                 batch_size = 2**10,
                 random_seed = 42,
                 train_path = 'crabnet_data/train.csv', 
                 val_path = 'crabnet_data/val.csv', 
                 test_path = 'crabnet_data/test.csv',
                 n_elements ='infer',
                 classification = False,
                 stratify = False,     
                 atomic_features = {'atomic_features_strategy': {'atom_feature_file': 'embeddings/atom_init_original.json',
                                                                 'soap_atomic': False}},
                 scale = True,
                 pin_memory = False):
        super().__init__()
        
        self.train_path = os.path.join(root_dir,train_path)
        self.val_path = os.path.join(root_dir,val_path)
        self.test_path = os.path.join(root_dir,test_path)

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.batch_size = batch_size
        self.n_elements=n_elements
        self.pin_memory = pin_memory
        self.scale = scale
        self.classification = classification
        self.random_seed = random_seed
        self.atomic_features = atomic_features

        data = pd.read_csv(os.path.join(root_dir,id_prop_csv),header = None)
        formulas=[]
        for name in data[0].values:
            structure=Structure.from_file(os.path.join(root_dir,str(name)+'.cif'))
            formulas.append(structure.formula)
        data['formula']=formulas
        data = normalize_formulas(data)
        
        data = data.rename(columns={'formula': 'formula', 1: 'target'})
        data=data[['formula', 'target']]

        if self.test_ratio == 1.0:
            train_idx=[]
            val_idx=[]
            test_idx=data.index.values
        elif 0.0 < self.test_ratio < 1.0 and 0.0 < self.train_ratio < 1.0:
            if stratify:
                y=data['target'].values
                train_idx, test_idx, y_train, _ = train_test_split(data.index.values, y, test_size=self.test_ratio, stratify=y, random_state=random_seed)
                train_idx, val_idx, _, _ = train_test_split(train_idx, y_train, train_size=self.train_ratio/(1-self.test_ratio), stratify=y_train,random_state=self.random_seed)
            else:
                train_idx, test_idx = train_test_split(data.index.values, test_size=self.test_ratio, random_state=random_seed)
                train_idx, val_idx = train_test_split(train_idx, train_size=self.train_ratio/(1-self.test_ratio), random_state=self.random_seed)
        else:
            raise ValueError("Invalid test_ratio or train_ratio. Ensure 0 < train_ratio, test_ratio < 1, or test_ratio == 1.0.")

        train = data.iloc[train_idx].reset_index(drop=True)
        val = data.iloc[val_idx].reset_index(drop=True)
        test = data.iloc[test_idx].reset_index(drop=True)

        train.to_csv(self.train_path, index=False,header=None)
        val.to_csv(self.val_path, index=False,header=None)
        test.to_csv(self.test_path, index=False,header=None)

        self.train_main_data = list(get_edm(train, 
                                      n_elements=self.n_elements,
                                      verbose=True,
                                      drop_unary=False,
                                      scale=self.scale))
        
        self.train_len_data = len(self.train_main_data[0])
        self.train_n_elements = self.train_main_data[0].shape[1]//2

        print(f'loading data with up to {self.train_n_elements:0.0f} '
              f'elements in the formula for training')
        
        
        self.val_main_data = list(get_edm(val,
                                      n_elements=self.n_elements,
                                      verbose=True,
                                      drop_unary=False,
                                      scale=self.scale))
        
        self.val_len_data = len(self.val_main_data[0])
        self.val_n_elements = self.val_main_data[0].shape[1]//2

        print(f'loading data with up to {self.val_n_elements:0.0f} '
              f'elements in the formula for validation')
        
        ### loading and encoding testing data
        
        self.test_main_data = list(get_edm(test,
                                      n_elements=self.n_elements,
                                      verbose=True,
                                      drop_unary=False,
                                      scale=self.scale))
        
        self.test_len_data = len(self.test_main_data[0])
        self.test_n_elements = self.test_main_data[0].shape[1]//2

        print(f'loading data with up to {self.test_n_elements:0.0f} '
              f'elements in the formula for testing')

        self.train_dataset = EDMDataset(self.train_main_data, self.train_n_elements)
        self.val_dataset = EDMDataset(self.val_main_data, self.val_n_elements)
        self.test_dataset = EDMDataset(self.test_main_data, self.test_n_elements)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          pin_memory=self.pin_memory, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                        pin_memory=self.pin_memory, shuffle=False)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_len_data,
                        pin_memory=self.pin_memory, shuffle=False)
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_len_data,
                        pin_memory=self.pin_memory, shuffle=False)
