import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital
from matminer.featurizers.base import MultipleFeaturizer
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from typing import Dict
import torch
from torch_geometric.data import Data
import json
import warnings

def calculate_soap_features(structure, soap_params):
    """Produce SOAP features for a structure assuming that the structure is a single element
       (so it is a lattice representation)
    """
    from dscribe.descriptors import SOAP
    from pymatgen.io.ase import AseAtomsAdaptor
    desc = SOAP(
                    species=['X'],  # or whatever elements you're using
                    r_cut=soap_params['r_cut'],
                    n_max= soap_params['n_max'],
                    l_max= soap_params['l_max'],
                    sigma= soap_params['sigma'],
                    periodic=True,
                    sparse=False
                )
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.set_chemical_symbols(["X"] * len(atoms))
    return desc.create(atoms)


def load_atom_features(atom_init_path: str) -> Dict:
    """Load atomic embedding file (traditionally keys are atomic numbers)"""
    with open(atom_init_path, 'r') as f:
        data = json.load(f)
    return data

def build_pyg_cgcnn_graph_from_structure(structure: Structure, 
                                         atom_features_dict: Dict, 
                                         radius: float=10.0, 
                                         max_neighbors: int=12,
                                         soap_params: Dict = None) -> Data:
    """Converts a pymatgen Structure to a PyTorch Geometric Data object with atomic features and edge distances."""
    num_atoms = len(structure)
    atomic_features = []
    
    if soap_params is not None:
        soap_features = calculate_soap_features(structure, soap_params)
    # Node features 
    for i, site in enumerate(structure):
        number = site.specie.number
        feature = atom_features_dict.get(str(number))
        if feature is None:
            raise ValueError(f"Atomic feature not found for element: {number}")
        if soap_params is not None:
            feature = np.concatenate([feature, soap_features[i]])
        atomic_features.append(feature)

    x = torch.tensor(atomic_features, dtype=torch.float32)
    
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


def normalize_formulas(df: pd.DataFrame, formula_column: str = 'formula') -> pd.DataFrame:
    """Normalize chemical formulas to IUPAC format, removing duplicates due to structural representations.

    Args:
        df (pd.DataFrame): DataFrame containing at least a 'formula' column.
        formula_column"""
    formula=[]
    for form in df[formula_column].values:
        formula.append(Composition(Composition(form).get_integer_formula_and_factor()[0]).iupac_formula)
    df[formula_column]=formula
    return df


def create_magpie_features(df, formula_column: str = 'formula'):
    """Create Magpie features for a dataframe of structures
    """
    df = normalize_formulas(df, formula_column)
    featurizer = MultipleFeaturizer([
                    ElementProperty.from_preset('magpie'),
                    Stoichiometry(),
                    ValenceOrbital()
                ])
    flen=len(featurizer.featurize(Composition(df['Formula'].values[0])))
    features=np.zeros((len(df),flen))
    df['composition']=[Composition(df.iloc[i]['Formula']).fractional_composition for i in range(len(df))]
    for i,comp in enumerate(df['composition']):
        features[i,:]=featurizer.featurize(comp)
    features=np.nan_to_num(features, copy=True, nan=0.0, posinf=None, neginf=None)
    return pd.DataFrame(features)

def create_structure_features(df, structure_column: str = 'structure'):
    """Create Magpie features for a dataframe of structures
    """
    # 7 crystal systems
    crystal_system_map = {
        "triclinic": 0,
        "monoclinic": 1,
        "orthorhombic": 2,
        "tetragonal": 3,
        "trigonal": 4,
        "hexagonal": 5,
        "cubic": 6
    }
    # 14 Bravais lattices (symbols and encodings)
    bravais_map = {
        "aP": 0,  # triclinic primitive
        "mP": 1, "mC": 2,  # monoclinic
        "oP": 3, "oC": 4, "oI": 5, "oF": 6,  # orthorhombic
        "tP": 7, "tI": 8,  # tetragonal
        "hP": 9, "hR": 10,  # hexagonal/trigonal
        "cP": 11, "cI": 12, "cF": 13  # cubic
    }
    # Map to abbreviations
    system_abbr = {
        "triclinic": "a",
        "monoclinic": "m",
        "orthorhombic": "o",
        "tetragonal": "t",
        "trigonal": "h",
        "hexagonal": "h",
        "cubic": "c"
    }
    features=np.zeros((len(df),15))
    for i,structure in enumerate(df[structure_column].values):
        feature=[]
        for x in structure.lattice.abc:
            feature.append(x)
        for x in structure.lattice.angles:
            feature.append(x)
        for x in structure.lattice.reciprocal_lattice.abc:
            feature.append(x)
        for x in structure.lattice.reciprocal_lattice.angles:
            feature.append(x)
        sga = SpacegroupAnalyzer(structure, symprec=0.01)
        spg_symbol = sga.get_space_group_symbol()
        spg_number = sga.get_space_group_number()
        crystal_system = sga.get_crystal_system()
        centering = spg_symbol[0]
        bravais = system_abbr[crystal_system] + centering
        crystal_system_id = crystal_system_map[crystal_system]
        bravais_id = bravais_map.get(bravais, -1)
        feature.append(crystal_system_id)
        feature.append(bravais_id)
        feature.append(spg_number)
        feature=np.array(feature)
        features[i,:]=feature
    features=np.nan_to_num(features, copy=True, nan=0.0, posinf=None, neginf=None)
    return pd.DataFrame(features)


def create_is_metal_cgcnn_features(root_dir: str, checkpoint_path: str):
    """Create a dataframe with embeddings extracted from prevoiusly trained CGCNN model
    """
    from datamodule import GNNDataModule
    from cgcnn_model import CGCNN_PyG

    data = GNNDataModule(root_dir = root_dir,
                         id_prop_csv = 'id_prop.csv',
                         model_name = 'cgcnn',
                         features_file = 'embeddings/atom_init_original.json',
                         train_ratio = 0.0,
                         val_ratio = 0,
                         test_ratio = 1.0,
                         lmdb_exist = True,
                         lmdb_train_name = 'train_data_is_metal.lmdb',
                         lmdb_val_name = 'val_data_is_metal.lmdb',
                         lmdb_test_name = 'test_data_is_metal.lmdb',
                         batch_size = 64,
                         radius = 10.0,
                         max_neighbors = 12,
                         pin_memory = True,
                         random_seed = 123,
                         stratify = False,
                         additional_features = None,
                         data_file = None)
    
    dataset=data.test_dataset
    g=dataset.get(0)
    orig_atom_fea_len = g.x[0].shape[-1]
    model=CGCNN_PyG(orig_atom_fea_len,robust_regression=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_weights = checkpoint["state_dict"]
    for key in list(model_weights):
        model_weights[key.replace("model.", "")] = model_weights.pop(key)
    model.load_state_dict(model_weights)
    model.eval()

    loader=data.test_dataloader()
    df=pd.DataFrame()
    for i,batch in enumerate(loader):
        features=model.extract_crystal_repr(batch)
        features=features.detach().numpy()
        df=pd.concat([df,pd.DataFrame(features)])
    return df

