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
from matminer.featurizers.base import MultipleFeaturizer

from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from utils.utils import normalize_formulas
import warnings


def matminer_composition_features(df: pd.DataFrame, 
                                  list_of_features: List, 
                                  formula_column = 'formula'):
    """Calculate composition features using matminer featurizers.
    
    Args:
        df: DataFrame containing chemical formulas.
        list_of_features: List of matminer composition featurizer names to use.
        formula_column: Name of the column containing formulas.
    
    Returns:
        Array of composition features.
    """
    df = normalize_formulas(df, formula_column)
    df['composition'] = [Composition(form) for form in df[formula_column]]
    
    list_of_feat_meth=[]
    for feat in list_of_features:
        if hasattr(matminer.featurizers.composition, feat):
            if(feat=='ElementProperty'):
                method = getattr(matminer.featurizers.composition , feat).from_preset('magpie', impute_nan=True)
            else:
                try:
                    method = getattr(matminer.featurizers.composition , feat)(impute_nan=True)
                except:
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
    """Calculate structure features using matminer featurizers.
    
    Args:
        df: DataFrame with compounds' information.
        list_of_features: List of matminer structure featurizer names to use.
        structure_column: Column in the dataframe which contains pymatgen structures.
    
    Returns:
        Array of structure features.
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
    for i, struct in enumerate(df[structure_column].values):
        try:
            features[i, :] = structure_featurizer.featurize(struct)
        except Exception as e:
            print(f"Warning: structure {struct.formula} at index {i} failed featurization: {e}")
            features[i, :] = 0.0  # fallback: zeros

    features=np.nan_to_num(features, copy=True, nan=0.0, posinf=None, neginf=None)
    return features


def lattice_features(df: pd.DataFrame, structure_column: str = 'structure'):
    """Create lattice features.
    
    Extracts lattice constants, lattice angles, reciprocal lattice constants,
    reciprocal lattice angles, space group number, crystal system, and Bravais lattice.
    
    Args:
        df: DataFrame containing structure information.
        structure_column: Column name containing pymatgen Structure objects.
    
    Returns:
        Array of lattice features.
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
    """Calculate SOAP compound features, all atoms are assumed to be the same.
    
    Args:
        df: DataFrame containing structure information.
        soap_params: Dictionary containing SOAP parameters (r_cut, n_max, l_max, sigma).
        structure_column: Column name containing pymatgen Structure objects.
    
    Returns:
        Array of SOAP features averaged over all atoms.
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
    """Calculate Jarvis CFID features for structures.
    
    Args:
        df: DataFrame containing structure information.
        structure_column: Column name containing pymatgen Structure objects.
    
    Returns:
        Array of Jarvis CFID features.
    """
    jarvis_featurizer = JarvisCFID()
    jarvis_feat_len = len(jarvis_featurizer.featurize(df.iloc[0][structure_column]))
    features=np.zeros((len(df),jarvis_feat_len))
    
    for i,struct in enumerate(df[structure_column].values):
        try:
            features[i,:]=jarvis_featurizer.featurize(struct) 
        except:
            form=struct.formula
            print(f'Compound: {form} failed to calculate JarvisCFID features',form)
    features=np.nan_to_num(features, copy=True, nan=0.0, posinf=None, neginf=None)
    return features

def cgcnn_features(checkpoint_path: str, data_path: str, lmdb_exist: bool=False):
    """Create a dataframe with embeddings extracted from previously trained CGCNN model.
    
    Args:
        checkpoint_path: Path to the trained CGCNN model checkpoint.
        data_path: Path to the data directory.
        lmdb_exist: Whether LMDB database already exists.
    
    Returns:
        Array of CGCNN embeddings.
    """
    from datamodules.gnn_datamodule import GNNDataModule
    from models.cgcnn import CGCNN_PyG

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    data = GNNDataModule(root_dir = data_path,
                         id_prop_csv = 'id_prop.csv',
                         model_name = 'cgcnn',
                         train_ratio = 0.0,
                         val_ratio = 0,
                         test_ratio = 1.0,
                         cal_ratio = 0.0,
                         calibration = False,
                         lmdb_exist = lmdb_exist,
                         lmdb_train_name = 'train_data_feat_model.lmdb',
                         lmdb_val_name = 'val_data_feat_model.lmdb',
                         lmdb_test_name = 'test_data_feat_model.lmdb',
                         lmdb_cal_name = 'cal_data_feat_model.lmdb',
                         batch_size = 64,
                        #  graph_params=checkpoint['hyper_parameters']['data']['graph_params'],
                         graph_params = None,
                         random_seed = checkpoint['hyper_parameters']['data']['random_seed'],
                         compound_features = {'additional_compound_features': None},
                         atomic_features = {'atom_feature_strategy': {'atom_feature_file': '/Users/elena.patyukova/Documents/github/clean-kpoints/goldilocks_kpoints/embeddings/atom_init_original.json',
                                                                         'soap_atomic': False}})

    model=CGCNN_PyG(**checkpoint['hyper_parameters']['model'])

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
    return np.array(df)

def remove_kpoints_section_robust(qe_input_text):
    """Remove kpoints section from QE input file.
    
    Args:
        qe_input_text: Text content of the QE input file.
    
    Returns:
        Text with kpoints section removed.
    """
    import re
    
    # Pattern to match K_POINTS section and everything until next section
    # This handles automatic, crystal, gamma, etc.
    pattern = r'K_POINTS\s+\w*\s*\n.*?(?=\n[A-Z_]+|\n&|\Z)'
    
    # Remove the K_POINTS section
    result = re.sub(pattern, '', qe_input_text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up any extra newlines
    result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
    
    return result.strip()

def matscibert_features(df: pd.DataFrame = None, structure_column = 'structure', data_path: str = None):
    """Create embeddings of QE-input files without k-points with MatSciBert model.
    
    Args:
        df: Optional DataFrame containing structure information.
        structure_column: Column name containing pymatgen Structure objects.
        data_path: Optional path to QE input files directory.
    
    Returns:
        Array of MatSciBert embeddings.
    """
    import math
    import torch
    from transformers import AutoTokenizer, AutoModel
    from tokenizers.normalizers import BertNormalizer
    from robocrys import StructureCondenser, StructureDescriber

    def mean_pooling(hidden_states, attention_mask):
        """Apply mean pooling to get fixed-size embedding"""
        # Expand attention mask to match hidden states dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # Apply mask and compute mean
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
        return sum_embeddings / sum_mask
    
    def create_embedding(chunks, tokenizer, model):
        """Create embedings using model/tokenizer pair"""
        chunk_embeddings = []
        for chunk in chunks:
            tokenized_text = tokenizer(
                    chunk, 
                    max_length=chunk_length, 
                    padding=True, 
                    truncation=True,  # Add truncation
                    return_tensors="pt"  # Return PyTorch tensors directly
                )
            with torch.no_grad():
                outputs = model(**tokenized_text)
                hidden_states = outputs.last_hidden_state  # [1, seq_len, 768]
                # Pool to fixed size [1, 768]
                pooled = mean_pooling(hidden_states, tokenized_text['attention_mask'])
                chunk_embeddings.append(pooled)
                
        chunk_stack = torch.stack(chunk_embeddings)  # [num_chunks, 1, 768]
        chunk_stack = chunk_stack.squeeze(1)  # [num_chunks, 768]
        # Aggregate all chunks into final embedding
        final_embedding = torch.mean(chunk_stack, dim=0)
        return final_embedding
    
    norm = BertNormalizer(lowercase=False, strip_accents=True, clean_text=True, handle_chinese_chars=True)
    tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert')
    model = AutoModel.from_pretrained('m3rg-iitd/matscibert')   
    ############################################################
    # 1. if we to encode input files
    if data_path is not None:
        print('QE input files are used for matscibert input feature...')
        list_of_files = os.listdir(data_path) 
        embeddings=np.zeros((len(list_of_files),768))
        for file_name in list_of_files:
            with open(os.path.join(data_path, file_name),'r') as file:
                text = file.read()
                text=remove_kpoints_section_robust(text)
            norm_text=norm.normalize_str(text)
            chunks = []
            chunk_length = 512
            num_chunks = math.ceil(len(norm_text) / chunk_length)

            for i in range(num_chunks):
                chunks.append(norm_text[i*chunk_length:chunk_length*(1+i)])
            final_embedding = create_embedding(chunks, tokenizer, model)
            i=int(file_name[:-3])
            embeddings[i,:]=final_embedding
    ############################################################
    # 2. encoding robocrystallographer structure description
    if df is not None:
        print('Robocrystallographer descriptions are used for matscibert features...')
        embeddings=np.zeros((len(df),768))
        for i,struct in enumerate(df[structure_column].values):
            try:
                condenser = StructureCondenser()
                describer = StructureDescriber()

                condensed_structure = condenser.condense_structure(struct)
                description = describer.describe(condensed_structure)
                norm_text = norm.normalize_str(description)

                chunks = []
                chunk_length = 512
                num_chunks = math.ceil(len(norm_text) / chunk_length)

                for i in range(num_chunks):
                    chunks.append(norm_text[i*chunk_length:chunk_length*(1+i)])
            except:
                print(f'for structure {i} with formula {struct.formula} cif-file is used')
                text = struct.to_file('.cif')
                norm_text = norm.normalize_str(text)
                chunks = []
                chunk_length = 512
                num_chunks = math.ceil(len(norm_text) / chunk_length)

                for i in range(num_chunks):
                    chunks.append(norm_text[i*chunk_length:chunk_length*(1+i)])

            final_embedding = create_embedding(chunks, tokenizer, model)
            embeddings[i,:]=final_embedding

    return embeddings
