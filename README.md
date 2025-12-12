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

1. **CGCNN** - Crystal Graph Convolutional Neural Network
2. **ALIGNN** - Atomistic Line Graph Neural Network
3. **CrabNet** - Transformer-based model for composition-based predictions ([Nature paper](https://www.nature.com/articles/s41524-021-00545-1))
4. **Random Forest** - Ensemble method with quantile regression support
5. **Gradient Boosting Trees** - XGBoost-style gradient boosting
6. **Histogram Gradient Boosting** - Fast gradient boosting implementation

### Atomic (Node) Features

- **CGCNN features** - Standard atomic embeddings
- **Energy cutoffs** - Density cutoffs determined by pseudo-potential type and energy cutoff
- **SOAP features** - Calculated for structures with all atoms substituted by one atom type

### Compound-Level Features

1. **Matminer composition features** - Element property and stoichiometry descriptors
2. **Matminer structure features** - Global symmetry, density, and structural descriptors
3. **JarvisCFID features** - JARVIS Crystal Fingerprint features
4. **SOAP features** - Averaged over all atoms in the structure
5. **CGCNN embeddings** - Features extracted from pre-trained CGCNN models (transfer learning)
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
git clone https://github.com/ddmms/k_points.git
cd goldilocks_kpoints

# Install dependencies using Poetry
poetry install

# Activate the virtual environment
poetry shell
```

### Key Dependencies

- PyTorch 2.4.0
- PyTorch Geometric 2.7.0
- PyTorch Lightning 2.4+
- pymatgen 2024.11.13
- matminer 0.9.3+
- transformers 4.54.1+ (for MatSciBert)

## Quick Start

### 1. Prepare Your Data

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

## Configuration

Experiments are defined using YAML configuration files in the `configs/` directory. Each model type has its own configuration template:

### Example Configuration Structure

```yaml
model:
  name: cgcnn  # or 'alignn', 'crabnet'
  classification: false
  robust_regression: true
  quantile_regression: false

data:
  root_dir: ./data
  id_prop_csv: id_prop.csv
  batch_size: 64
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1

optim:
  learning_rate: 0.001
  weight_decay: 0.0001
  swa: true
  swa_lr: 0.0001
  swa_start: 200

loss:
  name: RobustL2Loss
  quantile: 0.9  # for quantile regression
```

## Model Training

Models are typically trained for 300 epochs with:
- **Early stopping**: Monitors validation loss/metrics
- **Stochastic Weight Averaging (SWA)**: Optional, can be enabled for improved generalization
- **MLflow logging**: Automatic experiment tracking

### Training Options

- **Classification**: Multi-class classification with class weights
- **Regression**: Standard mean squared error or mean absolute error
- **Robust Regression**: Estimates aleatoric uncertainty (predicts mean and std)
- **Quantile Regression**: Predicts specific quantiles or intervals

## Baseline

As a baseline, we use a dummy model that predicts a constant `k-dist` value. This constant is determined from the training dataset to:
- Minimize the number of compounds for which predicted `k-dist` is larger than the true value (minimize underconverged calculations)
- Minimize MAE on the training dataset (minimize compute time)

In high-throughput calculations, a fixed low `k-dist` is usually used for all compounds to guarantee convergence. This package suggests using ML to predict `k-dist` values that are:
- Closer to the optimal values
- Minimize the number of underconverged calculations
- Reduce computational cost through optimized k-point sampling

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
├── datamodules/          # PyTorch Lightning data modules
│   ├── gnn_datamodule.py
│   ├── crabnet_datamodule.py
│   └── lmdb_dataset.py
├── models/               # Model implementations
│   ├── cgcnn.py
│   ├── alignn.py
│   ├── crabnet.py
│   ├── ensembles.py
│   └── modelmodule.py
├── utils/                # Utility functions
│   ├── atom_features_utils.py
│   ├── compound_features_utils.py
│   ├── cgcnn_graph.py
│   ├── alignn_graph.py
│   └── utils.py
├── scripts/              # Training and prediction scripts
│   ├── train.py
│   └── predict.py
└── README.md
```

## Advanced Features

### Uncertainty Quantification

The package supports multiple approaches for uncertainty estimation:

- **Aleatoric Uncertainty**: Using robust loss functions (RobustL1, RobustL2, StudentT)
- **Quantile Prediction**: Direct quantile regression
- **Conformal Prediction**: Coverage-guaranteed prediction intervals

### Transfer Learning

- Extract features from pre-trained CGCNN models
- Use MatSciBert embeddings from text descriptions
- Combine multiple feature types for improved performance

### Custom Features

Easily add custom features by:
1. Implementing feature extraction functions in `utils/compound_features_utils.py`
2. Adding feature configuration to your YAML file
3. The datamodule will automatically compute and concatenate features

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{goldilocks_kpoints,
  title = {Goldilocks K-Points: ML Models for Predicting K-Point Density in DFT Calculations},
  author = {Patyukova, Elena},
  year = {2024},
  url = {https://github.com/ddmms/k_points}
}
```

## References

1. **CGCNN Limitations**: Sheng Gong et al., "Examining graph neural networks for crystal structures: Limitations and opportunities for capturing periodicity." *Sci. Adv.* 9, eadi3245 (2023). [DOI:10.1126/sciadv.adi3245](https://doi.org/10.1126/sciadv.adi3245)

2. **MatSciBert**: Gupta, T., Zaki, M., Krishnan, N.M.A. et al. "MatSciBERT: A materials domain language model for text mining and information extraction." *npj Comput Mater* 8, 102 (2022). [DOI:10.1038/s41524-022-00784-w](https://doi.org/10.1038/s41524-022-00784-w)

3. **Robocrystallographer**: Ganose, A., & Jain, A. "Robocrystallographer: Automated crystal structure text descriptions and analysis." *MRS Communications* 9(3), 874-881 (2019). [DOI:10.1557/mrc.2019.94](https://doi.org/10.1557/mrc.2019.94)

4. **CrystalNN**: Hillary Pan, et al. "Benchmarking Coordination Number Prediction Algorithms on Inorganic Crystal Structures." *Inorganic Chemistry* 60(3), 1590-1603 (2021). [DOI:10.1021/acs.inorgchem.0c02996](https://doi.org/10.1021/acs.inorgchem.0c02996)

5. **CrabNet**: Wang, A.Y.T., et al. "Machine learning for materials discovery." *Nature Computational Materials* (2021). [DOI:10.1038/s41524-021-00545-1](https://www.nature.com/articles/s41524-021-00545-1)

## License

© 2025 Science and Technology Facilities Council (STFC)

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/ or see the [LICENSE](LICENSE) file in this repository.

### What this means

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

## Contact

For questions or issues, please contact:
- Elena Patyukova: elena.patyukova@stfc.ac.uk

---

**Note**: This package is under active development. Please report any issues or suggestions for improvement.
