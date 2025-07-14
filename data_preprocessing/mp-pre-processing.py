import numpy as np
import pandas as pd
import math
import json
import re
import os
from typing import List, Dict, Union
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from dotenv import load_dotenv
from pymatgen.core.composition import Composition
from mp_api.client import MPRester


def load_mp_data(fields: List[str], mp_api_key: str) -> pd.DataFrame:
    """Download MP data
       List of availible fields: 
       ['builder_meta', 'nsites', 'elements', 'nelements', 'composition', 'composition_reduced',
        'formula_pretty', 'formula_anonymous', 'chemsys', 'volume', 'density', 'density_atomic', 
        'symmetry', 'property_name', 'material_id', 'deprecated', 'deprecation_reasons', 
        'last_updated', 'origins', 'warnings', 'structure', 'task_ids', 
        'uncorrected_energy_per_atom', 'energy_per_atom', 'formation_energy_per_atom', 
        'energy_above_hull', 'is_stable', 'equilibrium_reaction_energy_per_atom', 'decomposes_to', 
        'xas', 'grain_boundaries', 'band_gap', 'cbm', 'vbm', 'efermi', 'is_gap_direct', 'is_metal', 
        'es_source_calc_id', 'bandstructure', 'dos', 'dos_energy_up', 'dos_energy_down', 
        'is_magnetic', 'ordering', 'total_magnetization', 'total_magnetization_normalized_vol', 
        'total_magnetization_normalized_formula_units', 'num_magnetic_sites', 
        'num_unique_magnetic_sites', 'types_of_magnetic_species', 'bulk_modulus', 
        'shear_modulus', 'universal_anisotropy', 'homogeneous_poisson', 'e_total', 
        'e_ionic', 'e_electronic', 'n', 'e_ij_max', 'weighted_surface_energy_EV_PER_ANG2', 
        'weighted_surface_energy', 'weighted_work_function', 'surface_anisotropy', 
        'shape_factor', 'has_reconstructed', 'possible_species', 'has_props', 
        'theoretical', 'database_IDs']
    """
    with MPRester(mp_api_key) as mpr:
        docs = mpr.materials.summary.search(fields=fields)
    MP_data=pd.DataFrame()
    for field in fields:
        MP_data[field]=[getattr(doc, field) for doc in docs]
    return MP_data

def normalize_formulas(df: pd.DataFrame, formula_column: str = 'formula_pretty') -> pd.DataFrame:
    """Normalize chemical formulas to IUPAC format, removing duplicates due to structural representations.

    Args:
        df (pd.DataFrame): DataFrame containing at least a 'formula' column.
        formula_column"""
    formula=[]
    for form in df[formula_column].values:
        formula.append(Composition(Composition(form).get_integer_formula_and_factor()[0]).iupac_formula)
    df[formula_column]=formula
    return df

def find_structural_duplicates(df: pd.DataFrame, matcher: StructureMatcher, formula_column: str = 'formula_pretty') -> Dict[str, List[str]]:
    """
    Identify structural duplicates in the dataset using StructureMatcher.

    Args:
        df (pd.DataFrame): DataFrame containing 'formula', 'jid', and 'atoms' columns.
        matcher (StructureMatcher): pymatgen StructureMatcher instance.
        formula_column (str): Column name for chemical formulas.

    Returns:
        Dict[str, List[str]]: Mapping of formulas to lists of unique JIDs that have structural duplicates.
    """
    duplicate_pairs = {}

    for formula in df[formula_column].unique():
        subset = df[df[formula_column] == formula]
        ids = subset['material_id'].values
        structure_data = subset['structure'].values

        pairs = []
        for i, struct1 in enumerate(structure_data):
            for j in range(i + 1, len(structure_data)):
                struct2 = structure_data[j]
                if matcher.fit(struct1, struct2):
                    pairs.append((ids[i], ids[j]))
        
        if pairs:
            duplicate_pairs[formula] = pairs

    # Reduce pairs to unique JIDs per formula
    duplicates_reduced = {
        formula: sorted(set(id for pair in pairs for id in pair))
        for formula, pairs in duplicate_pairs.items()
    }

    return duplicates_reduced

def remove_structural_duplicates(df: pd.DataFrame, duplicates: Union[Dict, None] = None, duplicates_file: Union[str, None] = None) -> pd.DataFrame:
    """Remove structural duplicates found previously"""
    if duplicates_file:
        with open(duplicates_file, 'r') as file:
            duplicates=json.load(file)
    
    if not duplicates:
        print('Provide the duplicates dictionary of file with duplicates!')
        return

    no_duplicates_subset=df.loc[~df['formula_pretty'].isin(duplicates.keys())]
    duplicates_subset=df.loc[df['formula_pretty'].isin(duplicates.keys())]
    selected_ids=[]
    for formula,ids in duplicates.items():
        subset=duplicates_subset.loc[duplicates_subset['material_id'].isin(ids)]
        subset=subset.loc[~subset['is_metal'].isna()]
        if(len(subset)>0):
            selected_ids.append(subset['material_id'].values[0])

    duplicates_subset=duplicates_subset.loc[duplicates_subset['material_id'].isin(selected_ids)]
    df_new=pd.concat([duplicates_subset,no_duplicates_subset])
    df_new.reset_index(inplace=True, drop=True)

    return df_new

if __name__ == "__main__":
    # Load and preprocess data
    load_dotenv()
    mp_api_key = os.environ.get('MP_API_KEY')
    fields=["material_id",'formula_pretty', "structure", 'is_metal']
    df = load_mp_data(fields, mp_api_key)
    df = normalize_formulas(df)

    # Initialize StructureMatcher
    matcher = StructureMatcher(attempt_supercell=True)

    # Find duplicates
    duplicate_structures = find_structural_duplicates(df, matcher)

    with open('./mp_metal_nonmetal_duplicates.json','w') as file:
        json.dump(duplicate_structures, file, indent = 4)
    

    # in contrast to jarvis we pickle because the file is too big for json
    mp_no_structural_duplicates = remove_structural_duplicates(df=df, duplicates_file = './mp_metal_nonmetal_duplicates.json')
    mp_no_structural_duplicates.to_pickle('./mp_metal_nonmetal_no_structural_duplicates.pkl')
