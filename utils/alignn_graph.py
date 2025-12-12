import numpy as np
from typing import List
from pymatgen.core.structure import Structure
import torch
from torch_geometric.data import Data
import warnings


def build_alignn_graph_with_angles_from_structure(structure: Structure, 
                                                  atom_features: List,
                                                  radius: float = 10.0,
                                                  max_neighbors: int = 12):
    """Generate ALIGNN-style atomic graph and line graph with angle (cosine) features.
    
    Args:
        structure: Pymatgen Structure object.
        atom_features: List of atomic feature arrays.
        radius: Cutoff radius for neighbor search.
        max_neighbors: Maximum number of neighbors per atom.
    
    Returns:
        Tuple of (g, lg):
            g: Atomic PyG graph with node & edge features.
            lg: Line graph with angle cosines as edge features.
    """
    edge_index = []
    edge_attr = []
    edge_vecs = []
    edge_map = {}  # (i, j) -> edge id
    reverse_map = {}  # (j, i) -> reverse edge id
    disconnected_atoms = []

    all_neighbors = structure.get_all_neighbors(radius, include_index=True)

    for i, neighbors in enumerate(all_neighbors):
        neighbors = sorted(neighbors, key=lambda x: x[1])[:max_neighbors]
        if len(neighbors) == 0:
            disconnected_atoms.append(i)
        for neighbor in neighbors:
            j = neighbor[2]
            dist = neighbor[1]
            vec = neighbor[0].coords - structure[i].coords
            edge_index.append([i, j])
            edge_attr.append(dist)
            edge_vecs.append(vec)
            eid = len(edge_vecs) - 1
            edge_map[(i, j)] = eid
            reverse_map[(j, i)] = eid  # for reverse tracking

    if disconnected_atoms:
        warnings.warn(
            f"{len(disconnected_atoms)} atoms had no neighbors within radius {radius}. "
        )

    x = torch.tensor(atom_features, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    edge_vecs = torch.tensor(edge_vecs, dtype=torch.float32)

    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    g.edge_vecs = edge_vecs  # custom attribute

    # === Build line graph manually === #
    bond_pairs = edge_index.t().tolist()
    line_edge_index = []
    angle_features = []
    reverse_bond_ids = []

    for idx_i, (a, b) in enumerate(bond_pairs):
        for idx_j, (c, d) in enumerate(bond_pairs):
            if b == c and a != d:  # shared node, no backtracking
                # reverse_id = reverse_map.get((a, b), -1)  # commented block skips bonds whcih does not have a reverse
                # if reverse_id == -1:
                #     continue  # skip if no reverse exists
                v1 = -edge_vecs[idx_i].numpy()
                v2 = edge_vecs[idx_j].numpy()
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                line_edge_index.append([idx_i, idx_j])
                angle_features.append(cos_angle)
                # track reverse bond if it exists
                reverse_id = reverse_map.get((a, b), -1)
                reverse_bond_ids.append(reverse_id)

    if line_edge_index:
        line_edge_index = torch.tensor(line_edge_index, dtype=torch.long).t().contiguous()
        angle_features = torch.tensor(angle_features, dtype=torch.float32)
        reverse_bond_ids = torch.tensor(reverse_bond_ids, dtype=torch.long)
    else:
        formula=structure.formula
        warnings.warn(
            f"!!! compound {formula} has empty line graph, which will course problems when training the model!!!"
        )
        line_edge_index = torch.empty((2, 0), dtype=torch.long)
        angle_features = torch.empty((0, 1), dtype=torch.float32)
        reverse_bond_ids = torch.empty((0,), dtype=torch.long)

    lg = Data(x=edge_attr, edge_index=line_edge_index, edge_attr=angle_features)
    return g, lg