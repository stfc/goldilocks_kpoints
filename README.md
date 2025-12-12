# Goldilocks K-Points: ML Models for Predicting K-Point Density in DFT Calculations

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-CC--BY--4.0-green.svg)](LICENSE)

**How do you choose parameters for your DFT calculations?**

This package provides machine learning models to predict optimal k-point density (`k-dist`) for SCF total energy calculations with plane-wave DFT codes for inorganic 3D materials. All models take as input the structure and/or composition of the compound and output `k-dist`, which is expected to guarantee convergence of total energy calculations while minimizing computational time.

## Overview

The package implements multiple machine learning approaches for predicting k-point density:

- **Graph Neural Networks (GNNs)**: CGCNN and ALIGNN models that learn from crystal structures
- **Transformer Models**: CrabNet for composition-based predictions
- **Ensemble Methods**: Random Forest, Gradient Boosting, and Histogram Gradient Boosting

The models support both regression and classification tasks, with advanced features for uncertainty quantification including robust regression, quantile regression, and conformal prediction.

## Features

### Implemented Models

1. **CGCNN** - Crystal Graph Convolutional Neural Network ([paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301))
2. **ALIGNN** - Atomistic Line Graph Neural Network ([paper](https://www.nature.com/articles/s41524-021-00650-1))
3. **CrabNet** - Transformer-based model for composition-based predictions ([paper](https://www.nature.com/articles/s41524-021-00545-1))
4. **Random Forest** - Ensemble method with quantile regression support ([scikit-learn](https://scikit-learn.org/), [sklearn-quantile](https://sklearn-quantile.readthedocs.io/))
5. **Gradient Boosting Trees** - XGBoost-style gradient boosting with quantile regression support ([scikit-learn implementation](https://scikit-learn.org/))
6. **Histogram Gradient Boosting** - Fast gradient boosting implementation

### Atomic (Node) Features

- **CGCNN features** - Standard atomic embeddings from CGCNN paper
- **CGCNN features modified with energy and density cutoff** - In addition to CGCNN feature the follwoing features added: 1-hot encoding for energy cutoff, 1-hot encoding for density cutoff, type of pseudopotential. PCA is performed on features to remove dimensions with no infromation content.
- **mat2vec features** - Mat2vec embeddings were developed by [Tshitoyan et al.](https://doi.org/10.1038/s41586-019-1335-8) via skip-gram variation of Word2vec method trained on 3.3 million scientific abstracts, and originally used in CrabNet model
- **SOAP features** - Calculated for structures with all atoms substituted by one atom type -- not used as was not effective as atomic features

### Compound-Level Features

1. **Matminer composition features** - Element property, stoichiometry, and valenceorbital descriptors
2. **Matminer structure features** - Global symmetry and  density descriptors
3. **JarvisCFID features** - JARVIS Crystal Fingerprint features, matminer implementation
4. **SOAP features** - Averaged over all atoms in the structure, calculated with [DScribe](https://singroup.github.io/dscribe/latest/)
5. **CGCNN embeddings** - Features extracted from pre-trained CGCNN models. Pre-trained CGCNN model was trained on MP 'is_metal' dataset (Autumn 2025)
6. **MatSciBert embeddings** - Generated from:
   - QE SCF input files with k-points section removed, or
   - Robocrystallographer structure descriptions

### Graph Construction Options

1. **Radius graph** - All atoms within a cutoff radius are considered neighbors
2. **CrystalNN graph** - Uses CrystalNN algorithm to identify nearest neighbors based on chemical environment

### Loss Functions (for Uncertainty Estimation)

1. **RobustL2 Loss** - Gaussian distribution-based robust loss
2. **RobustL1 Loss** - Laplace distribution-based robust loss
3. **StudentT Loss** - Student's t-distribution with configurable degrees of freedom
4. **Quantile Loss** - Single quantile prediction
5. **IntervalScoreLoss** - Interval prediction with coverage guarantees

### Architecture Highlights

For GNN models, atomic features are used as input to the graph neural network, and compound-level features are concatenated to the features produced by the GNN encoder. This hybrid approach enables:

- **Transfer learning**: Leveraging pre-trained models for feature extraction
- **Better structure learning**: Addressing limitations of GNNs in learning certain structural features
- **Domain knowledge integration**: Incorporating metallicity and other important predictors

## Installation

### Prerequisites

- Python 3.11 or 3.12
- Poetry (for dependency management)

### Setup

```bash
# Clone the repository
git clone https://github.com/stfc/goldilocks_kpoints.git
cd goldilocks_kpoints

# Create python environment
python -m venv .venv
source .venv/bin/activate

# Install dependences of pytorch-geometric as described in [here](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). It is needed as torch_scatter, torch_sparse should be installed from binary wheels using pip and can't be installed with poetry.

# Install dependencies using Poetry
poetry install

# Activate the virtual environment
poetry shell
```

## Quick Start

### 1. Prepare Your Data in the CGCNN format (see their paper for details)

Create a CSV file (`id_prop.csv`) with two columns:
- Column 0: Sample IDs (corresponding to `{id}.cif` files)
- Column 1: Target values (k-dist values)

Place your CIF files in the same directory as the CSV file.

### 2. Configure Your Experiment

Create or modify a configuration file in `configs/` directory. Example configurations:
- `configs/cgcnn.yaml` - For CGCNN model
- `configs/alignn.yaml` - For ALIGNN model
- `configs/crabnet.yaml` - For CrabNet model
- `configs/ensembles.yaml` - For ensemble models

### 3. Train a Model

```bash
python scripts/train.py --config_file configs/cgcnn.yaml
```

### 4. Make Predictions

```bash
python scripts/predict.py \
    --config_file configs/cgcnn.yaml \
    --checkpoint_path trained_models/cgcnn/ \
    --output_name output/predictions.csv
```

### 5. To perform Conformalised quantile regression, first train quantile models using quantile loss, or QRF (for ranfom Forest). Then use the notebooks (availibel for RF and ALIGNN, but can be easily modified for other models) to calculate conformal corrections to the intervals.

## Model Training

Models are typically trained for 300 epochs with:
- **Early stopping**: Monitors validation loss/metrics
- **Stochastic Weight Averaging (SWA)**: Optional, can be enabled

### Training Options

- **Classification**: Multi-class classification with class weights
- **Regression**: Standard mean squared error or mean absolute error
- **Robust Regression**: Estimates aleatoric uncertainty (predicts mean and std)
- **Quantile Regression**: Predicts specific quantiles or intervals


## Dataset and Convergence Definition

The dataset and convergence definition were provided by Junwen Yin.

### Calculation Parameters

All data was generated with fixed parameters:
1. **Code**: Quantum Espresso
2. **Pseudopotentials**: SSSP1.3_PBESol_efficiency library
3. **Energy cutoffs**: Recommended values for SSSP1.3_PBESol_efficiency
4. **Smearing**: Cold smearing with `degauss=0.01 Ry`
5. **Magnetism**: All compounds treated as non-magnetic

### Convergence Criterion

A calculation is considered converged if the total energy change for 3 consecutive k-meshes with increasing number of points is within **1 meV/atom**.

## Project Structure

```
goldilocks_kpoints/
├── configs/              # Configuration files for different models
├── data/                 # Data directory (CIF files, CSV files)
├── trained_models/       # The place to store trained models
├── outputs/              # The place to write outputs to
├── embeddings/
|   ├── atom_init_original.json
|   ├── atom_init_with_sssp_cutoffs.json
|   └── mat2vec.json
├── datamodules/          # PyTorch Lightning data modules
│   ├── gnn_datamodule.py
│   ├── crabnet_datamodule.py
│   └── lmdb_dataset.py
├── models/               # Model implementations
│   ├── cgcnn.py
│   ├── alignn.py
|   ├── kingcrab.py
│   ├── crabnet.py
│   ├── ensembles.py
│   └── modelmodule.py
├── utils/                # Utility functions
│   ├── atom_features_utils.py
│   ├── compound_features_utils.py
│   ├── cgcnn_graph.py
|   ├── crabnet_utils.py
│   ├── alignn_graph.py
│   └── utils.py
├── scripts/              # Training and prediction scripts
│   ├── train.py
│   └── predict.py
├── notebooks/
|   ├── Data-exploration.ipynb
|   ├── RF-feature-importance.ipynb
|   ├── Surrogate-models.ipynb
|   ├── ALIGNN-CQR.ipynb
|   ├── RF-CQR.ipynb
|   └── Wall-time.ipynb
└── README.md
```


## Citation

If you use this code in your research, please cite:

```bibtex
@article{goldilocks_kpoints,
  title = {Automatic generation of input files with optimised k-point meshes for Quantum Espresso self-consistent field single point total energy calculations},
  author = {Elena Patyukova, Junwen Yin, Susmita Basak, Jaehoon Cha, Alin Elenaa, and Gilberto Teobaldi},
  year = {2025},
  url = {to be published}
}
```

## License

© 2025 Science and Technology Facilities Council (STFC)

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
