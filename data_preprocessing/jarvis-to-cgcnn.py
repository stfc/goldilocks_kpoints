import numpy as np
import sys
import argparse
import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import math
import os
import shutil

parser = argparse.ArgumentParser(description='Creating data for CGCNN model from Jarvis')
parser.add_argument('--folder', default="", type=str, help='cgcnn data folder')
parser.add_argument('--property', default="kpoint_length_unit", type=str, help='specify the jarvis column name')
parser.add_argument('--subset', default="all", choices=['all', 'metals', 'nonmetals'], type=str, help='choose jarvis subset')
args = parser.parse_args(sys.argv[1:])

def create_data_for_cgcnn(df: pd.DataFrame, cgcnn_data_folder: str, property: str, subset: str) -> None:
    """The data for CGCNN model should contain the folder with name data/cgcnn_data/
       In this folder there should be 
       (1) structure files with names 'ind.cif' for each structure
       (2) 'atom_init.json' file with embeddings for all atom types
       (3) 'id_prop.csv' file with two columns (ind,property) without heading.
    """
    if(subset == 'all'):
        pass
    elif(subset == 'metals'):
        df=df.loc[df['optb88vdw_bandgap'] != 'na']
        df=df.loc[df['optb88vdw_bandgap'] == 0.0]
        df.reset_index(inplace=True,drop=True)
    elif(subset == 'nonmetals'):
        df=df.loc[df['optb88vdw_bandgap'] != 'na']
        df=df.loc[df['optb88vdw_bandgap'] > 0]
        df.reset_index(inplace=True,drop=True)

    if(property=='knumber'):
        df=df.loc[df['kpoint_length_unit']!='na']
        df.reset_index(inplace=True,drop=True)
        number_of_kpoints=[]
        for ind in df.index.values:
             atoms=df.iloc[ind]['atoms']
             structure=Structure(lattice=atoms['lattice_mat'],species=atoms['elements'],coords=atoms['coords'],coords_are_cartesian=True)
             klength=df.iloc[ind]['kpoint_length_unit']
             sga = SpacegroupAnalyzer(structure, symprec=1e-2)
             structure = sga.get_primitive_standard_structure()
             mesh = tuple([math.ceil(klength/x) for x in structure.lattice.abc])
             ir_kpoints_weights = sga.get_ir_reciprocal_mesh(mesh=mesh, is_shift=(0, 0, 0))
             number_of_kpoints.append(len(ir_kpoints_weights))
             structure.to_file(os.path.join(cgcnn_data_folder, str(ind)+'.cif'))
        df['knumber']=number_of_kpoints
        id_prop=df[['knumber']]
        id_prop.to_csv(os.path.join(cgcnn_data_folder,'id_prop.csv'),header=None)

    else:
        df=df.loc[df[property]!='na']
        df.reset_index(inplace=True,drop=True)
        id_prop=df[[property]]
        id_prop.to_csv(os.path.join(cgcnn_data_folder,'id_prop.csv'),header=None)

        for ind in df.index.values:
            atoms=df.iloc[ind]['atoms']
            structure=Structure(lattice=atoms['lattice_mat'],species=atoms['elements'],coords=atoms['coords'],coords_are_cartesian=True)
            sga = SpacegroupAnalyzer(structure, symprec=1e-2)
            structure = sga.get_primitive_standard_structure()
            structure.to_file(os.path.join(cgcnn_data_folder, str(ind)+'.cif'))

    return

if __name__ == "__main__":
    df=pd.read_json("./jarvis_no_structural_duplicates.json")
    
    try:
        os.makedirs(args.folder, exist_ok = False)
    except:
        shutil.rmtree(args.folder)
        os.makedirs(args.folder)
    
    shutil.copy('./atom_init.json', os.path.join(args.folder,'atom_init.json'))

    create_data_for_cgcnn(df,
                          cgcnn_data_folder=args.folder,
                          property=args.property,
                          subset=args.subset)

