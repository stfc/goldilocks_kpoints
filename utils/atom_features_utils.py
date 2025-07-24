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


def load_atom_features(atom_init_path: str) -> Dict:
    """Load atomic embedding file (traditionally keys are atomic numbers)"""
    with open(atom_init_path, 'r') as f:
        data = json.load(f)
    return data


def atomic_soap_features(structure, soap_params):
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

def atom_features_from_structure(structure: Structure, atomic_features: Dict):
    """Calculate an array of atomic features for structure
    """
    atom_features_dict=load_atom_features(atomic_features['atom_feature_strategy']['atom_feature_file'])
    
    if atomic_features['atom_feature_strategy']['soap_atomic']:
        soap=atomic_soap_features(structure, atomic_features['soap_params'])

    atom_features=[]
    for i, site in enumerate(structure):
        number = site.specie.number
        feature = atom_features_dict.get(str(number))
        if feature is None:
            raise ValueError(f"Atomic feature not found for element: {number}")
        if atomic_features['atom_feature_strategy']['soap_atomic']:
            feature=np.concatenate([feature,soap[i,:]],axis=0)
        atom_features.append(feature)

    return atom_features