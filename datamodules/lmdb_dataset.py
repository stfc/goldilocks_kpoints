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
from utils.cgcnn_graph import build_radius_cgcnn_graph_from_structure, build_crystalnn_cgcnn_graph_from_structure
from utils.alignn_graph import build_alignn_graph_with_angles_from_structure
from utils.atom_features_utils import atom_features_from_structure

import lmdb
import pickle as pk
from torch_geometric.data import Dataset


class LMDBPyGDataset(Dataset):
    """PyTorch Geometric dataset stored in LMDB format.
    
    Loads graph data from an LMDB database for efficient I/O during training.
    """
    def __init__(self, lmdb_path, model='cgcnn'):
        """Initialize the LMDB dataset.
        
        Args:
            lmdb_path: Path to the LMDB database.
            model: Model type ('cgcnn' or 'alignn').
        """
        super().__init__()
        self.lmdb_path = lmdb_path
        self.model = model

        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin() as txn:
            self.length = txn.stat()['entries']

    def len(self):
        """Get the length of the dataset.
        
        Returns:
            Number of samples in the dataset.
        """
        return self.length

    def get(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample.
        
        Returns:
            Graph data (single Data object for CGCNN, tuple of (graph, line_graph) for ALIGNN).
        """
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
        """Collate function for batching samples.
        
        Args:
            samples: List of graph samples.
        
        Returns:
            Batched graph data.
        """
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
                         graph_type="radius",
                         additional_compound_features_df=None):
    """Read all the structures, build atomic graphs and dump them into LMDB database.
    
    Args:
        data: DataFrame from id_prop.csv file.
        file_path: Location of LMDB database on the local disk.
        data_dir: Directory in which CIF files are located, CIF file names are 'idx.cif'.
        atomic_features: Dictionary with atomic features configuration.
        radius: Cutoff radius for neighbor search.
        max_neighbors: Maximum number of neighbors per atom.
        model: Model type ('cgcnn' or 'alignn').
        graph_type: Graph construction type ('radius' or 'crystalnn').
        additional_compound_features_df: Optional DataFrame with additional compound features.
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
                if additional_compound_features_df is not None:
                    additional_features = additional_compound_features_df.iloc[idx].values
                    additional_features = torch.tensor(additional_features).type(torch.get_default_dtype())
                    data_g.additional_compound_features = additional_features
                serialized_data = pk.dumps((data_g, data_lg))
            elif model == "cgcnn":
                if graph_type == 'radius':
                    data_g = build_radius_cgcnn_graph_from_structure(
                        structure, atom_features, radius=radius, max_neighbors=max_neighbors)
                elif graph_type == 'crystalnn':
                    data_g = build_crystalnn_cgcnn_graph_from_structure(
                        structure, atom_features, radius=radius)
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

    

