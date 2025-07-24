"""Module to prepare LMDB ALIGNN and CGCNN datasets."""

import os
import numpy as np
import lmdb
import json
import pickle as pk
from torch_geometric.data import Dataset, Batch
import torch
from typing import Dict
from pymatgen.core.structure import Structure
from utils.cgcnn_graph import build_pyg_cgcnn_graph_from_structure
from utils.alignn_graph import build_alignn_graph_with_angles_from_structure
from utils.atom_features_utils import atom_features_from_structure

import lmdb
import pickle as pk
from torch_geometric.data import Dataset


class LMDBPyGDataset(Dataset):
    def __init__(self, lmdb_path, model='cgcnn'):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.model = model

        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin() as txn:
            self.length = txn.stat()['entries']

    def len(self):
        return self.length

    def get(self, idx):
        with self.env.begin() as txn:
            byte_data = txn.get(f"{idx}".encode())
            sample = pk.loads(byte_data)

        if self.model == "alignn":
            data_g, data_lg = sample
            return data_g, data_lg
        elif self.model == "cgcnn":
            return sample  # just a single Data object
        else:
            raise ValueError(f"Unknown model type: {self.model}")
    
    def collate_fn(self, samples):
        if self.model == "cgcnn":
            return Batch.from_data_list(samples)

        elif self.model == "alignn":
            data_g_list, data_lg_list = zip(*samples)
            batch_g = Batch.from_data_list(data_g_list)
            batch_lg = Batch.from_data_list(data_lg_list)
            return batch_g, batch_lg

        else:
            raise ValueError(f"Unknown model type: {self.model}")


def load_atom_features(atom_init_path: str) -> Dict:
    """Load atomic embedding file (traditionally keys are atomic numbers)"""
    with open(atom_init_path, 'r') as f:
        data = json.load(f)
    return data


def create_lmdb_database(data, 
                         file_path, 
                         data_dir, 
                         atomic_features,
                         radius=10.0,
                         max_neighbors=12, 
                         model="cgcnn",
                         additional_compound_features_df=None):
    """Read all the structures, build atomic graphs and dump them into LMDB database
       data: DataFrame from id_prop.csv file
       file_path: location of LMDB database on the local disk
       data_dir: directory in which CIF files are located, CIF file names are 'idx.cif'
       atom_features_dict: dictionary with atomic features, keys are atomic numbers
       model: 'cgcnn' or 'alignn'
       soap_params: if you want to add soap atomic feature
    """

    env = lmdb.open(os.path.join(data_dir, file_path), map_size=int(1e12))

    with env.begin(write=True) as txn:
        for idx in range(len(data)):
            cif_path = os.path.join(data_dir, str(int(data.iloc[idx][0])) + '.cif')
            structure = Structure.from_file(cif_path)
            atom_features = atom_features_from_structure(structure, atomic_features)
            label = torch.tensor(data.iloc[idx][1]).type(torch.get_default_dtype())
            sample_id = torch.tensor(int(data.iloc[idx][0]))

            if model == "alignn":
                data_g, data_lg = build_alignn_graph_with_angles_from_structure(
                    structure, atom_features, radius=radius, max_neighbors=max_neighbors
                )
                data_g.y = label
                data_g.sample_id = sample_id
                data_lg.sample_id = sample_id
                serialized_data = pk.dumps((data_g, data_lg))
            elif model == "cgcnn":
                data_g = build_pyg_cgcnn_graph_from_structure(
                    structure, atom_features, radius=radius, max_neighbors=max_neighbors)
                if additional_compound_features_df is not None:
                    additional_features = additional_compound_features_df.iloc[idx].values
                    additional_features = torch.tensor(additional_features).type(torch.get_default_dtype())
                    data_g.additional_compound_features = additional_features
                data_g.y = label
                data_g.sample_id = sample_id
                serialized_data = pk.dumps(data_g)
            else:
                raise ValueError(f"Unknown model type: {model}")

            txn.put(f"{idx}".encode(), serialized_data)
    env.close()
    return

    

