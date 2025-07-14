import numpy as np
import pandas as pd
import json

from typing import Dict, List, Union
from jarvis.db.figshare import data
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.analysis.structure_matcher import StructureMatcher

def load_jarvis_data() -> pd.DataFrame:
    """Download jarvis dataset"""
    dft_3d_entries = data('dft_3d')
    df = pd.DataFrame(dft_3d_entries)
    return df

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


def find_structural_duplicates(df: pd.DataFrame, matcher: StructureMatcher, formula_column: str = 'formula') -> Dict[str, List[str]]:
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
        jids = subset['jid'].values
        atoms_data = subset['atoms'].values

        pairs = []
        for i, atoms1 in enumerate(atoms_data):
            structure1 = Structure(
                lattice=atoms1['lattice_mat'],
                species=atoms1['elements'],
                coords=atoms1['coords'],
                coords_are_cartesian=True
            )
            for j in range(i + 1, len(atoms_data)):
                atoms2 = atoms_data[j]
                structure2 = Structure(
                    lattice=atoms2['lattice_mat'],
                    species=atoms2['elements'],
                    coords=atoms2['coords'],
                    coords_are_cartesian=True
                )
                if matcher.fit(structure1, structure2):
                    pairs.append((jids[i], jids[j]))
        
        if pairs:
            duplicate_pairs[formula] = pairs

    # Reduce pairs to unique JIDs per formula
    duplicates_reduced = {
        formula: sorted(set(jid for pair in pairs for jid in pair))
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

    no_duplicates_subset=df.loc[~df['formula'].isin(duplicates.keys())]
    selected_ids=[]
    for formula,ids in duplicates.items():
        subset=df.loc[df['jid'].isin(ids)]
        subset=subset.loc[subset['kpoint_length_unit']!='na']
        subset=subset.loc[subset['optb88vdw_bandgap']!='na']
        if(len(subset)>0):
            selected_ids.append(subset['jid'].values[0])

    duplicates_subset=df.loc[df['jid'].isin(selected_ids)]
    df_new=pd.concat([duplicates_subset,no_duplicates_subset])
    df_new.reset_index(inplace=True, drop=True)

    return df_new

def remove_formula_duplicates(df: pd.DataFrame, duplicate_choice: str = 'lowest_e_form') -> pd.DataFrame:
    """We choose a polymorph with the lowest formation energy here"""
    list_of_formulas=list(set(df['formula']))

    selected_ids=[]
    for formula in list_of_formulas:
        subset=df.loc[df['formula']==formula]
        subset=subset[subset['kpoint_length_unit']!='na']
        if(len(subset)>1):
            if(duplicate_choice == 'lowest_e_form'):
                selected_ids.append(subset.sort_values(by='formation_energy_peratom', ascending=True)['jid'].values[0])
        elif(len(subset)>0):
            selected_ids.append(subset['jid'].values[0])
            
    df_new=df.loc[df['jid'].isin(selected_ids)]
    df_new.reset_index(inplace=True,drop=True)

    return df_new

if __name__ == "__main__":
    # Load and preprocess data
    df = load_jarvis_data()
    df = normalize_formulas(df)

    # Initialize StructureMatcher
    matcher = StructureMatcher(attempt_supercell=True)

    # Find duplicates
    duplicate_structures = find_structural_duplicates(df, matcher)

    with open('./jarvis_duplicates.json','w') as file:
        json.dump(duplicate_structures, file, indent = 4)

    jarvis_no_structural_duplicates = remove_structural_duplicates(df=df, duplicates_file = './jarvis_duplicates.json')
    jarvis_no_formula_duplicates = remove_formula_duplicates(df=df, duplicate_choice = 'lowest_e_form')

    jarvis_no_structural_duplicates.to_json('./jarvis_no_structural_duplicates.json')
    jarvis_no_formula_duplicates.to_json('jarvis_no_formula_duplicates.json')
