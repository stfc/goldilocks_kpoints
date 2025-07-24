import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital
from matminer.featurizers.base import MultipleFeaturizer
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from typing import List
import torch
from torch_geometric.data import Data
import warnings


def build_pyg_cgcnn_graph_from_structure(structure: Structure, 
                                         atom_features: List, 
                                         radius: float=10.0, 
                                         max_neighbors: int=12) -> Data:
    """Converts a pymatgen Structure to a PyTorch Geometric Data object with atomic features and edge distances."""
    
    x = torch.tensor(atom_features, dtype=torch.float32) 
    # Edge features: collect neighbors
    edge_index = []
    edge_attr = []
    
    all_neighbors = structure.get_all_neighbors(radius, include_index=True)
    disconnected_atoms=[]

    for i, neighbors in enumerate(all_neighbors):
        neighbors = sorted(neighbors, key=lambda x: x[1])[:max_neighbors]  # take closest max_neighbors
        if len(neighbors) == 0:
            disconnected_atoms.append(i)
        for neighbor in neighbors:
            j = neighbor[2]  # neighbor atom index
            dist = neighbor[1]
            edge_index.append([i, j])
            edge_attr.append([dist])
    
    if disconnected_atoms:
        warnings.warn(
            f"{len(disconnected_atoms)} atoms had no neighbors within radius {radius}. "
            f"Disconnected atom indices: {disconnected_atoms}"
        )

    # Convert to tensors
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data




