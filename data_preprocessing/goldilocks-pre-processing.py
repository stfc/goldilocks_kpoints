import numpy as np
import pandas as pd
import os
import json
import shutil
import argparse
import sys
import ast
from typing import Dict, List, Union
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.analysis.structure_matcher import StructureMatcher

parser = argparse.ArgumentParser(description="Creating data for CGCNN model from Goldilocks's data")
parser.add_argument('--data_file', default='data/upload_version/summary.csv', type=str, help='summary data file')
parser.add_argument('--data_folder', default='data/upload_version/structure_calc_details/',type=str,help="original folder with Goldilocks's data")
parser.add_argument('--target_folder', default='data/goldilocks', type=str, help='target data folder')
args = parser.parse_args(sys.argv[1:])

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

    for formula in df['Formula'].unique():
        subset = df[df[formula_column] == formula]
        ids = subset['source_db_id'].values

        pairs = []
        for i, idx in enumerate(ids):
            structure1 = df.loc[df['source_db_id']==idx]['structure'].values[0]
            for j in range(i + 1, len(ids)):
                structure2 = df.loc[df['source_db_id']==ids[j]]['structure'].values[0]
                if matcher.fit(structure1, structure2):
                    pairs.append((ids[i], ids[j]))    
        if pairs:
            duplicate_pairs[formula] = pairs   

    # Reduce pairs to unique JIDs per formula
    duplicates_reduced = {
        formula: sorted(set(idx for pair in pairs for idx in pair))
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

    no_duplicates_subset=df.loc[~df['Formula'].isin(duplicates.keys())]
    duplicates_subset=df.loc[df['Formula'].isin(duplicates.keys())]
    selected_ids=[]
    for formula,ids in duplicates.items():
        subset=duplicates_subset.loc[duplicates_subset['source_db_id'].isin(ids)]
        subset=subset.loc[~subset['k_dist'].isna()]
        if(len(subset)>0):
            row = subset[subset['k_dist'] == subset['k_dist'].min()]
            selected_ids.append(row['source_db_id'].values[0])

    duplicates_subset=duplicates_subset.loc[duplicates_subset['source_db_id'].isin(selected_ids)]
    df_new=pd.concat([duplicates_subset,no_duplicates_subset])
    df_new.reset_index(inplace=True, drop=True)

    return df_new

if __name__ == "__main__":
    if not os.path.exists(args.target_folder):
        os.makedirs(args.target_folder)
    df=pd.read_csv(args.data_file)
    df=normalize_formulas(df,formula_column='Formula')
    
    structures=[]
    k_distances=[]
    densities=[]
    for ind in range(len(df)):
        da=df.iloc[ind]
        mesh=ast.literal_eval(da['ultra k-mesh'])
        name=da['source_db_id']
        structure = Structure.from_file(os.path.join(args.data_folder+name,name+'.cif'))
        density=structure.density
        k_dist=max(b / n for b, n in zip(structure.lattice.reciprocal_lattice.abc, mesh))
        k_distances.append(k_dist)
        structures.append(structure)
        densities.append(density)
    df['structure']=structures
    df['k_dist']=k_distances
    df['density']=densities
    df=df.loc[df['density']>0.5]
    df.reset_index(inplace=True, drop=True)
    df.to_csv(os.path.join(args.target_folder,'goldilocks_kdist.csv'),index=False)
    

    matcher = StructureMatcher(attempt_supercell=False)
    duplicate_structures = find_structural_duplicates(df, matcher, formula_column = 'Formula')
    with open(os.path.join(args.target_folder,'goldilocks_structural_duplicates.json'),'w') as file:
        json.dump(duplicate_structures, file, indent = 4)

    da_new = remove_structural_duplicates(df=df, duplicates_file = os.path.join(args.target_folder,'goldilocks_structural_duplicates.json'))
    da_new.to_csv(os.path.join(args.target_folder,'goldilocks_no_structural_duplicates.csv'))
    
    prop=da_new[['k_dist']]
    prop.to_csv(os.path.join(args.target_folder,'id_prop.csv'),header=None)

    for ind in da_new.index.values:
        entry=da_new.iloc[ind]
        structure = entry['structure']
        structure.to_file(os.path.join(args.target_folder,str(ind)+'.cif'))
                    
