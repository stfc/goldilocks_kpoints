# Prediction of k-distance
* How do you choose parameters for your DFT calculations? *

This package contains the code to find/train/evaluate the best model to predict k-dist for SCF total energy calculations with plan wave DFT codes for inorganic 3D materials. All models take as input the structure or/and composition of the compound and output k-dist, which is expected to guarantee convergence of total energy calculations and minimize the amount of computational time.

### Implemented models:
1. CGCNN
2. ALIGNN
3. CGCNN + Additional features after GNN layers
4. Random Forrest 
5. Gradient Boosting Trees
6. Under development ...

As features depending on the model, the following can be used as standalone or a part of the feature vector:
1. CGCNN features
2. Energy cutoffs (density cutoffs are determined by the type of pseudo-potential and the value of energy cutoff)
3. Matminer composition features
4. Matminer structure features
5. JarvisCFID features
6. Soap features calculated for the structure with all atoms substituted by one atom type
7. Features extracted from another CGCNN model trained to predict whether the compound is metal/isolator

### Model training/evaluation
To train/evaluate any of the models, use script/train.py and script/predict.py. Parameters of the experiment should be specified in the corresponding config file (configs/cgcnn.yaml, configs/alignn.yaml, configs/ensembles.py)

Usually, models are trained for 300 epochs with early stopping monitoring the loss on the validation dataset. SWA can be turned on.

### Baseline


### Definition of convergence and dataset generation. 
The definition of convergence and dataset generation was provided by Yunwen Yin.

All the data was generated with these parameters fixed:
1. Quantum Espresso code
2. SSSP1.3_PBESol_efficiency library of pseudopotentials
3. Energy cutoffs are those recommended for SSSP1.3_PBESol_efficiency
4. Cold smearing is used for all compounds with degauss=0.1Ry
5. All compounds are treated as non-magnetic

The calculation is considered converged if the total energy change for the 3 consecutive k-meshes with increasing number of points is within 1meV/atom.

