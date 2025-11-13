import pandas as pd
import pytorch_lightning as L
import numpy as np
import os
import shutil
from datamodules.lmdb_dataset import create_lmdb_database, LMDBPyGDataset
from sklearn.model_selection import train_test_split
import pickle as pk
from torch.utils.data import DataLoader
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from utils.compound_features_utils import matminer_composition_features, matminer_structure_features
from utils.compound_features_utils import soap_features, jarvis_features, lattice_features, cgcnn_features
from utils.compound_features_utils import matscibert_features
from utils.atom_features_utils import atom_features_from_structure

class GNNDataModule(L.LightningDataModule):
    """ Datamodule for neural network models (CGCNN, ALIGNN)
    """
    def __init__(self, root_dir: str,
                 model_name: str,
                 id_prop_csv: str,
                 qe_input_files = None,
                 train_ratio = 0.8,
                 val_ratio = 0.1, 
                 test_ratio = 0.1,
                 lmdb_exist = False,
                 lmdb_train_name = 'train_data.lmdb',
                 lmdb_val_name = 'val_data.lmdb',
                 lmdb_test_name = 'test_data.lmdb',
                 batch_size = 64,
                 graph_params = None,
                 pin_memory = False,
                 random_seed = 42,
                 stratify = False,
                 scale_y = False,
                 compound_features = {'additional_compound_features': None,
                                      'checkpoint_path': None,
                                      'data_file': None,
                                      'soap_params': {'r_cut': 10.0, 'n_max': 8, 'l_max': 6, 'sigma': 1.0}},
                 atomic_features = {'atomic_features_strategy': {'atom_feature_file': 'embeddings/atom_init_original.json',
                                                                 'soap_atomic': False}}):
        super().__init__() 

        self.random_seed=random_seed
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.model = model_name
        if graph_params is not None:
            self.max_neighbors = graph_params['max_neighbors']
            self.radius = graph_params['radius']
            self.graph_type = graph_params['graph_type']
        else:
            self.max_neighbors = 12
            self.radius = 10.0
            self.graph_type = 'radius'
        self.lmdb_train_name = lmdb_train_name
        self.lmdb_val_name = lmdb_val_name
        self.lmdb_test_name = lmdb_test_name
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.lmdb_exist = lmdb_exist

        self.atomic_features = atomic_features
        self.compound_features = compound_features

        data = pd.read_csv(os.path.join(root_dir, id_prop_csv), header=None)

        if(self.compound_features['additional_compound_features'] is not None):
            if self.compound_features['data_file'] is not None:
                df=pd.read_pickle(os.path.join(root_dir, self.compound_features['data_file']))
            else:
                structures = []
                formulas = []
                compositions = []
                for i in data[0].values:
                    struct=Structure.from_file(os.path.join(root_dir,str(i)+'.cif'))
                    structures.append(struct)
                    formulas.append(struct.formula)
                    compositions.append(Composition(struct.formula))
                df=pd.DataFrame({'id': data[0].values,
                                 'structure': structures,
                                 'formula': formulas,
                                 'composition': compositions})
            list_of_feat=[]
            if 'composition_features' in self.compound_features['additional_compound_features']:
                specs = [k for k, v in self.compound_features['composition_features'].items() if v]
                composition_features = matminer_composition_features(df, specs)
                list_of_feat.append(composition_features)
            if 'structure_features' in self.compound_features['additional_compound_features']:
                specs = [k for k, v in self.compound_features['structure_features'].items() if v]
                structure_features = matminer_structure_features(df, specs)
                list_of_feat.append(structure_features)
            if 'soap_features' in self.compound_features['additional_compound_features']:
                soap = soap_features(df, soap_params=self.compound_features['soap_params'])
                list_of_feat.append(soap)
            if 'lattice_features' in self.compound_features['additional_compound_features']:
                lattice = lattice_features(df)
                list_of_feat.append(lattice)
            if 'jarvis_features' in self.compound_features['additional_compound_features']:
                jarvis = jarvis_features(df)
                list_of_feat.append(jarvis)
            if 'cgcnn_features' in self.compound_features['additional_compound_features']:
                cgcnn_f = cgcnn_features(self.compound_features['checkpoint_path'], 
                                         root_dir,
                                         lmdb_exist=self.compound_features['feat_lmdb_exist'])
                list_of_feat.append(cgcnn_f)
            if 'matscibert_features' in self.compound_features['additional_compound_features']:
                if qe_input_files is not None:
                    matscibert_f = matscibert_features(df=None, data_path = qe_input_files)
                else:
                    matscibert_f = matscibert_features(df=df)
                list_of_feat.append(matscibert_f)
            additional_features_df=pd.DataFrame(np.concatenate(list_of_feat,axis=1))


        print(f'test_ratio {self.test_ratio}, train_ratio {self.train_ratio}')

        if self.test_ratio == 1.0:
            train_idx=[]
            val_idx=[]
            test_idx=data.index.values
        elif 0.0 < self.test_ratio < 1.0 and 0.0 < self.train_ratio < 1.0:
            if stratify:
                y=data[1].values
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

        train.to_csv(os.path.join(root_dir, 'train.csv'), index=False,header=None)
        val.to_csv(os.path.join(root_dir, 'val.csv'), index=False,header=None)
        test.to_csv(os.path.join(root_dir, 'test.csv'), index=False,header=None)

        if(self.compound_features['additional_compound_features'] is not None):
            train_add_feat=additional_features_df.iloc[train_idx].reset_index(drop=True)
            val_add_feat=additional_features_df.iloc[val_idx].reset_index(drop=True)
            test_add_feat=additional_features_df.iloc[test_idx].reset_index(drop=True)

        list_of_paths=[self.lmdb_train_name, self.lmdb_val_name, self.lmdb_test_name]
        
        if(self.lmdb_exist == False):
            if os.path.exists(os.path.join(root_dir,self.lmdb_train_name)):
                shutil.rmtree(os.path.join(root_dir,self.lmdb_train_name))
            if os.path.exists(os.path.join(root_dir,self.lmdb_val_name)):
                shutil.rmtree(os.path.join(root_dir,self.lmdb_val_name))
            if os.path.exists(os.path.join(root_dir,self.lmdb_test_name)):
                shutil.rmtree(os.path.join(root_dir,self.lmdb_test_name))
            if(self.compound_features['additional_compound_features'] is not None): 
                    create_lmdb_database(train, self.lmdb_train_name, root_dir, self.atomic_features,\
                                 radius=self.radius,max_neighbors=self.max_neighbors, model=self.model,\
                                 graph_type=self.graph_type,additional_compound_features_df=train_add_feat)
                    create_lmdb_database(val,self.lmdb_val_name, root_dir, self.atomic_features,\
                                 radius=self.radius,max_neighbors=self.max_neighbors, model=self.model,\
                                 graph_type=self.graph_type,additional_compound_features_df=val_add_feat)
                    create_lmdb_database(test,self.lmdb_test_name, root_dir, self.atomic_features,\
                                 radius=self.radius,max_neighbors=self.max_neighbors, model=self.model,\
                                 graph_type=self.graph_type,additional_compound_features_df=test_add_feat)
            else:
                    create_lmdb_database(train,self.lmdb_train_name, root_dir, self.atomic_features,\
                                 radius=self.radius,max_neighbors=self.max_neighbors, model=self.model,\
                                 graph_type=self.graph_type)
                    create_lmdb_database(val,self.lmdb_val_name, root_dir, self.atomic_features,\
                                 radius=self.radius,max_neighbors=self.max_neighbors, model=self.model,\
                                 graph_type=self.graph_type)
                    create_lmdb_database(test,self.lmdb_test_name, root_dir, self.atomic_features,\
                                 radius=self.radius,max_neighbors=self.max_neighbors, model=self.model,\
                                 graph_type=self.graph_type)
                     
        elif not all(os.path.exists(os.path.join(root_dir,var)) for var in list_of_paths):
            print("Put lmdb_exist to False or provide train/val/test lmdb files.")
            
        self.train_dataset = LMDBPyGDataset(os.path.join(root_dir, self.lmdb_train_name), model=self.model)
        self.val_dataset = LMDBPyGDataset(os.path.join(root_dir, self.lmdb_val_name), model=self.model)
        self.test_dataset = LMDBPyGDataset(os.path.join(root_dir, self.lmdb_test_name), model=self.model)

        self.train_collate = self.train_dataset.collate_fn
        self.val_collate = self.val_dataset.collate_fn
        self.test_collate = self.test_dataset.collate_fn
  
    def train_dataloader(self,shuffle=True):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=0,collate_fn=self.train_collate, 
                          pin_memory=self.pin_memory, shuffle=shuffle)
    def val_dataloader(self,shuffle=False):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=0, collate_fn=self.val_collate, 
                          pin_memory=self.pin_memory, shuffle=shuffle)
    def test_dataloader(self,shuffle=False):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=0, collate_fn=self.test_collate, 
                          pin_memory=self.pin_memory, shuffle=shuffle)
    def predict_dataloader(self,shuffle=False):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=0, collate_fn=self.test_collate, 
                          pin_memory=self.pin_memory, shuffle=shuffle)