import numpy as np
import pandas as pd
import os
import json
import shutil
import argparse
import sys
from typing import Dict, List, Union
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.analysis.structure_matcher import StructureMatcher

parser = argparse.ArgumentParser(description="Creating data for CGCNN model from Junwen's data")
parser.add_argument('--data_file', default='./aggregated_v4_analysis.csv', type=str, help='summery data file')
parser.add_argument('--data_folder', default='',type=str,help="original folder with Junwen's data")
parser.add_argument('--target_folder', default='data/cgcnn_junwen_may',type='str',help='target data folder')
parser.add_argument('--property', default='k_number', choices=['k_number', 'convergence', 'convergence_class'], \
                    type=str, help='property we want to predict: k_number, convergence, or convergence_class')
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

    for formula in df[formula_column].unique():
        subset = df[df[formula_column] == formula]
        ids = subset['source_db_id'].values

        pairs = []
        for i, idx in enumerate(ids):
            structure1 = Structure.from_file(os.path.join(path_data+idx,idx+'.cif'))
            for j in range(i + 1, len(ids)):
                structure2 = Structure.from_file(os.path.join(path_data+ids[j],ids[j]+'.cif'))
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
        subset=subset.loc[~subset['k_number'].isna()]
        if(len(subset)>0):
            row = subset[subset['k_number'] == subset['k_number'].max()]
            selected_ids.append(row['source_db_id'].values[0])

    duplicates_subset=duplicates_subset.loc[duplicates_subset['source_db_id'].isin(selected_ids)]
    df_new=pd.concat([duplicates_subset,no_duplicates_subset])
    df_new.reset_index(inplace=True, drop=True)

    return df_new

if __name__ == "__main__":
    df=pd.read_csv(args.data_file)
    df=normalize_formulas(df,formula_column='Formula')

    not_converged=[]

    for ind in range(len(df)):
        da=df.iloc[ind]
        if(pd.isna(da['medium_k_number']) and pd.isna(da['well_k_number']) and pd.isna(da['ultra_k_number'])):
            not_converged.append(ind)

    only_medium=[]

    for ind in range(len(df)):
        da=df.iloc[ind]
        if(not(pd.isna(da['medium_k_number'])) and pd.isna(da['well_k_number']) and pd.isna(da['ultra_k_number'])):
            only_medium.append(ind)

    only_well=[]

    for ind in range(len(df)):
        da=df.iloc[ind]
        if(not(pd.isna(da['well_k_number'])) and pd.isna(da['ultra_k_number'])):
            only_well.append(ind)

    only_ultra=[]

    for ind in range(len(df)):
        da=df.iloc[ind]
        if(not(pd.isna(da['ultra_k_number']))):
            only_ultra.append(ind)

    convergence_class=[]
    k_number=[]

    for ind in range(len(df)):
        if(ind in not_converged):
            convergence_class.append(0)
            k_number.append(np.nan)
        elif(ind in only_medium):
            convergence_class.append(1)
            k_number.append(df.iloc[ind]['medium_k_number'])
        elif(ind in only_well):
            convergence_class.append(2)
            k_number.append(df.iloc[ind]['well_k_number'])
        elif(ind in only_ultra):
            convergence_class.append(3)
            k_number.append(df.iloc[ind]['ultra_k_number'])
    df['convergence_class']=convergence_class
    df['k_number']=k_number

    matcher = StructureMatcher(attempt_supercell=False)
    duplicate_structures = find_structural_duplicates(df, matcher, formula_column = 'Formula')
    with open('./junwen_may_structural_duplicates.json','w') as file:
        json.dump(duplicate_structures, file, indent = 4)

    da_new = remove_structural_duplicates(df=df, duplicates_file = './junwen_may_structural_duplicates.json')
    da_new.to_csv('./junwen_may_no_structural_duplicates.csv')
    
    prop=da_new[['k_number']]
    prop.to_csv(os.path.join(args.target_folder,'id_prop.csv'),header=None)

    for ind in da_new.index.values:
        name=da_new.iloc[ind]['source_db_id']
        structure = Structure.from_file(os.path.join(args.data_folder+name,name+'.cif'))
        structure.to_file(os.path.join(args.target_folder,str(ind)+'.cif'))
                    
