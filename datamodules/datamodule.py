import pandas as pd
import pytorch_lightning as L
import os
import shutil
from .lmdb_dataset import load_atom_features, create_lmdb_database, LMDBPyGDataset
from sklearn.model_selection import train_test_split
import pickle as pk
from torch.utils.data import DataLoader
from .cgcnn_graph import create_magpie_features, create_is_metal_cgcnn_features, create_structure_features


class GNNDataModule(L.LightningDataModule):
    def __init__(self, root_dir: str,
                 model_name: str,
                 id_prop_csv: str,
                 features_file: str,
                 train_ratio = 0.8,
                 val_ratio = 0.1, 
                 test_ratio = 0.1,
                 lmdb_exist = False,
                 lmdb_train_name = 'train_data.lmdb',
                 lmdb_val_name = 'val_data.lmdb',
                 lmdb_test_name = 'test_data.lmdb',
                 batch_size = 64,
                 graph_params = None,
                 pin_memory = True,
                 random_seed = 123,
                 stratify = False,
                 soap_params = None,
                 additional_compound_features = None,
                 additional_atom_features = None,
                 checkpoint_path = None,
                 data_file = None,
                 scale_y = False):
        super().__init__() 

        self.random_seed=random_seed
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.model = model_name
        if graph_params is not None:
            self.max_neighbors = graph_params['max_neighbors']
            self.radius = graph_params['radius']
        else:
            self.max_neighbors = 12
            self.radius = 10.0
        self.additional_compound_features = additional_compound_features
        self.additional_atom_features = additional_atom_features
        self.lmdb_train_name = lmdb_train_name
        self.lmdb_val_name = lmdb_val_name
        self.lmdb_test_name = lmdb_test_name
        self.soap_params = soap_params

        atom_features_dict = load_atom_features(features_file)
        
        if(self.additional_compound_features == 'magpie'):
            df=pd.read_pickle(data_file)
            additional_features_df = create_magpie_features(df, formula_column = 'Formula')
        elif(self.additional_compound_features == 'is_metal_cgcnn'):
            additional_features_df = create_is_metal_cgcnn_features(root_dir, checkpoint_path)
        elif(self.additional_compound_features == 'lattice'):
            df=pd.read_pickle(data_file)
            additional_features_df = create_structure_features(df, structure_column = 'structure')
        
        data = pd.read_csv(os.path.join(root_dir, id_prop_csv), header=None)
        if(test_ratio == 1.0):
            train_idx=[]
            val_idx=[]
            test_idx=data.index.values
        elif(test_ratio<1.0 and test_ratio>0.0 and train_ratio<1.0 and train_ratio>0.0):
            if stratify:
                y=data[1].values
                train_idx, test_idx, y_train, _ = train_test_split(data.index.values, y, test_size=test_ratio, stratify=y, random_state=random_seed)
                train_idx, val_idx, _, _ = train_test_split(train_idx, y_train, train_size=train_ratio/(1-test_ratio), stratify=y_train,random_state=random_seed)
            else:
                train_idx, test_idx = train_test_split(data.index.values, test_size=test_ratio, random_state=random_seed)
                train_idx, val_idx = train_test_split(train_idx, train_size=train_ratio/(1-test_ratio), random_state=random_seed)
        else:
            print('Wrong test_ratio or train_ratio parameters')

        train = data.iloc[train_idx].reset_index(drop=True)
        val = data.iloc[val_idx].reset_index(drop=True)
        test = data.iloc[test_idx].reset_index(drop=True)

        train.to_csv(os.path.join(root_dir, 'train.csv'), index=False,header=None)
        val.to_csv(os.path.join(root_dir, 'val.csv'), index=False,header=None)
        test.to_csv(os.path.join(root_dir, 'test.csv'), index=False,header=None)

        if(self.additional_compound_features is not None):
            train_add_feat=additional_features_df.iloc[train_idx].reset_index(drop=True)
            val_add_feat=additional_features_df.iloc[val_idx].reset_index(drop=True)
            test_add_feat=additional_features_df.iloc[test_idx].reset_index(drop=True)

        list_of_paths=[self.lmdb_train_name, self.lmdb_val_name, self.lmdb_test_name]
        
        if(lmdb_exist == False):
            if os.path.exists(os.path.join(root_dir,'train_data.lmdb')):
                shutil.rmtree(os.path.join(root_dir,'train_data.lmdb'))
            if os.path.exists(os.path.join(root_dir,'val_data.lmdb')):
                shutil.rmtree(os.path.join(root_dir,'val_data.lmdb'))
            if os.path.exists(os.path.join(root_dir,'test_data.lmdb')):
                shutil.rmtree(os.path.join(root_dir,'test_data.lmdb'))
            if(self.additional_compound_features is not None):
                if(self.additional_atom_features is not None):
                    create_lmdb_database(train,self.lmdb_train_name, root_dir, atom_features_dict, \
                                 radius=self.radius,max_neighbors=self.max_neighbors, model=self.model,\
                                 additional_compound_features_df=train_add_feat, soap_params=self.soap_params)
                    create_lmdb_database(val,self.lmdb_val_name, root_dir, atom_features_dict, \
                                 radius=self.radius,max_neighbors=self.max_neighbors, model=self.model,\
                                 additional_compound_features_df=val_add_feat, soap_params=self.soap_params)
                    create_lmdb_database(test,self.lmdb_test_name, root_dir, atom_features_dict, \
                                 radius=self.radius,max_neighbors=self.max_neighbors, model=self.model,\
                                 additional_compound_features_df=test_add_feat, soap_params=self.soap_params)
                else:
                    create_lmdb_database(train,self.lmdb_train_name, root_dir, atom_features_dict, \
                                 radius=self.radius,max_neighbors=self.max_neighbors, model=self.model,\
                                 additional_compound_features_df=train_add_feat)
                    create_lmdb_database(val,self.lmdb_val_name, root_dir, atom_features_dict, \
                                 radius=self.radius,max_neighbors=self.max_neighbors, model=self.model,\
                                 additional_compound_features_df=val_add_feat)
                    create_lmdb_database(test,self.lmdb_test_name, root_dir, atom_features_dict, \
                                 radius=self.radius,max_neighbors=self.max_neighbors, model=self.model,\
                                 additional_compound_features_df=test_add_feat)

            else:
                if(self.additional_atom_features is not None):
                    create_lmdb_database(train,self.lmdb_train_name, root_dir, atom_features_dict, \
                                 radius=self.radius,max_neighbors=self.max_neighbors, model=self.model,\
                                 soap_params=self.soap_params)
                    create_lmdb_database(val,self.lmdb_val_name, root_dir, atom_features_dict, \
                                 radius=self.radius,max_neighbors=self.max_neighbors, model=self.model,\
                                 soap_params=self.soap_params)
                    create_lmdb_database(test,self.lmdb_test_name, root_dir, atom_features_dict, \
                                 radius=self.radius,max_neighbors=self.max_neighbors, model=self.model,\
                                 soap_params=self.soap_params)  
                else:
                    create_lmdb_database(train,self.lmdb_train_name, root_dir, atom_features_dict, \
                                 radius=self.radius,max_neighbors=self.max_neighbors, model=self.model)
                    create_lmdb_database(val,self.lmdb_val_name, root_dir, atom_features_dict, \
                                 radius=self.radius,max_neighbors=self.max_neighbors, model=self.model)
                    create_lmdb_database(test,self.lmdb_test_name, root_dir, atom_features_dict, \
                                 radius=self.radius,max_neighbors=self.max_neighbors, model=self.model)
                     
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