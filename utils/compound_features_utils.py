import sys
from pathlib import Path
# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
import torch
import os
import pickle
from typing import List

import matminer
from matminer.featurizers.structure.composite import JarvisCFID
from dscribe.descriptors import SOAP
from pymatgen.io.ase import AseAtomsAdaptor
from dscribe.descriptors import SOAP
from matminer.featurizers.base import MultipleFeaturizer

from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from utils.utils import normalize_formulas
import warnings


def matminer_composition_features(df: pd.DataFrame, 
                                  list_of_features: List, 
                                  formula_column = 'formula'):
    """Function to calculate composition features
    """
    df = normalize_formulas(df, formula_column)
    df['composition'] = [Composition(form) for form in df[formula_column]]
    
    list_of_feat_meth=[]
    for feat in list_of_features:
        if hasattr(matminer.featurizers.composition, feat):
            if(feat=='ElementProperty'):
                method = getattr(matminer.featurizers.composition , feat).from_preset('magpie')
            else:
                method = getattr(matminer.featurizers.composition , feat)()
            list_of_feat_meth.append(method)
            
            # Use individual featurizers instead of MultipleFeaturizer to avoid argument passing issues
    composition_featurizer = MultipleFeaturizer(list_of_feat_meth)
    
    comp_feat_len = len(composition_featurizer.featurize(df.iloc[0]['composition']))
    features=np.zeros((len(df),comp_feat_len))     
    for i,comp in enumerate(df['composition'].values):
        features[i,:]=composition_featurizer.featurize(comp)
    
    features=np.nan_to_num(features, copy=True, nan=0.0, posinf=None, neginf=None)
    return features

def matminer_structure_features(df: pd.DataFrame,
                                list_of_features: List,
                                structure_column = 'structure'):
    """Function to calculate structure features
       Input:
       df: dataframe with compounds' information 
       list_of_features: list of matminer structure feature methods
       structure_column: column in the dataframe which contains pymatgen structures for compounds.
    """
    list_of_feat_meth=[]
    for feat in list_of_features:
        if(feat=='GlobalSymmetryFeatures'):
            props=["spacegroup_num", "crystal_system_int", "is_centrosymmetric"]
            method = getattr(matminer.featurizers.structure, feat)(props)
        elif(feat=='DensityFeatures'):
            props=["density", "vpa", "packing fraction"]
            method = getattr(matminer.featurizers.structure, feat)(props)
        list_of_feat_meth.append(method)
    
    structure_featurizer = MultipleFeaturizer(list_of_feat_meth)
    struct_feat_len = len(structure_featurizer.featurize(df.iloc[0][structure_column]))
    features=np.zeros((len(df),struct_feat_len))
    for i,struct in enumerate(df[structure_column].values):
        features[i,:]=structure_featurizer.featurize(struct)

    features=np.nan_to_num(features, copy=True, nan=0.0, posinf=None, neginf=None)
    return features


def lattice_features(df: pd.DataFrame, structure_column: str = 'structure'):
    """Create lattice features:
       lattice constants, lattice angles,
       reciprocal lattice constants, reciprocal lattice angles,
       space_group_number, crystal_system, bravais_lattice
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
        try:
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
        except:
            print(f'failed to calculate lattice features for {structure.formula}')
    features=np.nan_to_num(features, copy=True, nan=0.0, posinf=None, neginf=None)
    return features

def soap_features(df: pd.DataFrame,
                  soap_params = {'r_cut': 10.0, 'n_max': 8, 'l_max': 6, 'sigma': 1.0},
                  structure_column = 'structure'):
    """Function to calculate SOAP compound features, all atoms are assumed to be the same
    """
    soap_featurizer = SOAP(species=['X'],  # or whatever elements you're using
                               r_cut=soap_params['r_cut'],
                               n_max=soap_params['n_max'],
                               l_max=soap_params['l_max'],
                               sigma=soap_params['sigma'],
                               periodic=True,
                               sparse=False)

    atoms = AseAtomsAdaptor.get_atoms(df.iloc[0][structure_column])
    atoms.set_chemical_symbols(["X"] * len(atoms))
    soap=soap_featurizer.create(atoms).mean(axis=0)
    soap_feat_len = len(soap)
    features=np.zeros((len(df),soap_feat_len))
    for i,struct in enumerate(df[structure_column].values):
        atoms = AseAtomsAdaptor.get_atoms(struct)
        atoms.set_chemical_symbols(["X"] * len(atoms))
        features[i,:]=soap_featurizer.create(atoms).mean(axis=0)
    
    features=np.nan_to_num(features, copy=True, nan=0.0, posinf=None, neginf=None)
    return features

def jarvis_features(df: pd.DataFrame,
                    structure_column = 'structure'):
    """This function is for calculation of Jarvis features
    """
    jarvis_featurizer = JarvisCFID()
    jarvis_feat_len = len(jarvis_featurizer.featurize(df.iloc[0][structure_column]))
    features=np.zeros((len(df),jarvis_feat_len))
    for i,struct in enumerate(df[structure_column].values):
        features[i,:]=jarvis_featurizer.featurize(struct) 
    features=np.nan_to_num(features, copy=True, nan=0.0, posinf=None, neginf=None)
    return features

def cgcnn_features(checkpoint_path: str, data_path: str, lmdb_exist: bool=True):
    """Create a dataframe with embeddings extracted from prevoiusly trained CGCNN model
    """
    from datamodules.datamodule import GNNDataModule
    from models.cgcnn import CGCNN_PyG

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    data = GNNDataModule(root_dir = data_path,
                         id_prop_csv = 'id_prop.csv',
                         model_name = 'cgcnn',
                         train_ratio = 0.0,
                         val_ratio = 0,
                         test_ratio = 1.0,
                         lmdb_exist = lmdb_exist,
                         lmdb_train_name = 'train_data_feat_model.lmdb',
                         lmdb_val_name = 'val_data_feat_model.lmdb',
                         lmdb_test_name = 'test_data_feat_model.lmdb',
                         batch_size = 64,
                         graph_params=checkpoint['hyper_parameters']['data']['graph_params'],
                         random_seed = checkpoint['hyper_parameters']['data']['random_seed'],
                         compound_features = {'additional_compound_features': None},
                         atomic_features = {'atomic_features_strategy': {'atom_feature_file': 'embeddings/atom_init_original.json',
                                                                         'soap_atomic': False}})

    model=CGCNN_PyG(**checkpoint['hyper_parameters']['model'])

    model_weights = checkpoint["state_dict"]
    for key in list(model_weights):
        model_weights[key.replace("model.", "")] = model_weights.pop(key)
    model.load_state_dict(model_weights)
    # model.load_state_dict(model_weights, strict=False)
    model.eval()

    loader=data.test_dataloader()
    df=pd.DataFrame()
    for i,batch in enumerate(loader):
        features=model.extract_crystal_repr(batch)
        features=features.detach().numpy()
        df=pd.concat([df,pd.DataFrame(features)])
    return np.array(df)