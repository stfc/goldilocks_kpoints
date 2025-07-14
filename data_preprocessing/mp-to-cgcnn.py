import numpy as np
import sys
import argparse
import pandas as pd
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import os
import shutil
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Creating data for CGCNN model from Jarvis')
parser.add_argument('--folder', default="", type=str, help='cgcnn data folder')
parser.add_argument('--property', default="is_metal", type=str, help='specify the mp property name')
args = parser.parse_args(sys.argv[1:])

def create_data_for_cgcnn(df: pd.DataFrame, cgcnn_data_folder: str, property: str) -> None:
    """The data for CGCNN model should contain the folder with name data/cgcnn_data/
       In this folder there should be 
       (1) structure files with names 'ind.cif' for each structure
       (2) 'atom_init.json' file with embeddings for all atom types
       (3) 'id_prop.csv' file with two columns (ind,property) without heading.
    """
    df=df.loc[~df[property].isna()]
    df.reset_index(inplace=True,drop=True)
    

    if(isinstance(df[property].values[0],bool)):
        prop = np.ones(len(df))*df[property].values
        df['prop']=prop
        id_prop=df[['prop']]
    else:
        id_prop=df[['prop']]
    id_prop.to_csv(os.path.join(cgcnn_data_folder,'id_prop.csv'),header=None)

    for ind in tqdm(df.index.values):
        structure=df.iloc[ind]['structure']
        sga = SpacegroupAnalyzer(structure, symprec=1e-2)
        structure = sga.get_primitive_standard_structure()
        structure.to_file(os.path.join(cgcnn_data_folder, str(ind)+'.cif'))

    return

if __name__ == "__main__":
    df=pd.read_pickle("./mp_metal_nonmetal_no_structural_duplicates.pkl")
    
    try:
        os.makedirs(args.folder, exist_ok = False)
    except:
        shutil.rmtree(args.folder)
        os.makedirs(args.folder)
    
    shutil.copy('./atom_init.json', os.path.join(args.folder,'atom_init.json'))

    create_data_for_cgcnn(df,
                          cgcnn_data_folder=args.folder,
                          property=args.property)