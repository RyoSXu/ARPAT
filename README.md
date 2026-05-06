# ARPAT: Phonon Density of States Prediction from Atomic Relative Position Encoding Attention Transformer

ARPAT is a deep learning framework designed to predict the **Phonon Density of States (PhDOS)** of crystalline materials. It leverages a Transformer architecture with a specialized **Atomic Relative Position Encoding (ARPAT)** mechanism to capture both chemical identities and the complex geometric relationships between atoms in a crystal lattice.

![Model Architecture](https://raw.githubusercontent.com/your-username/ARPAT/main/FIG/architecture_placeholder.png) *(Note: Update this URL with your actual FIG path if hosted)*

## Core Features

- **Relative Position Encoding**: Incorporates interatomic distances and unit directions into the Attention mechanism using RBF and Spherical Harmonics.
- **PBC-Aware**: Correctly handles Periodic Boundary Conditions (PBC) to compute real-space Cartesian relative vectors from fractional coordinates.
- **Hybrid Embedding**: Combines discrete atom type embeddings with continuous numerical atomic features.
- **Multi-task Ready**: Optimized for PhDOS regression with built-in metrics (MAE, MSE, etc.).

## Repository Structure

```text
.
├── ARPAT_MP/           # Core implementation of the model and datasets
├── configs/            # Configuration files (YAML) for model hyperparameters
├── data/               # Data processing scripts (csv2npy.py)
├── datasets/           # PyTorch dataset implementation
├── model/              # ARPAT Transformer architecture (transformer.py, model.py)
├── utils/              # Geometric features, metrics, and encoding helpers
├── train.py            # Main training script
└── test.py             # Evaluation script
```

## Getting Started

### 1. Installation

Ensure you have Python 3.8+ and PyTorch installed.

```bash
pip install -r requirements.txt
# Requirements: torch, numpy, yaml, e3nn, pandas
```

### 2. Data Preparation

The model expects atomic positions, lattice parameters, and target PhDOS in `.npy` format. You can use the provided script to convert CSV data:

```bash
python data/csv2npy.py
```

### 3. Training

To start the training process using the default configuration:

```bash
bash train_script.sh
# Or directly:
python train.py --cfg configs/transformer.yaml --outdir ./output
```

### 4. Evaluation

To test the model performance on a test dataset:

```bash
bash test_script.sh
# Or directly:
python test.py --cfg configs/transformer.yaml --resume True
```

## Model Highlights

### Atomic Relative Position Encoding (RPEncoding)
Located in `utils/rp_encoding.py`, this module expands relative vectors using:
- **Radial Basis Functions (RBF)** for distance information.
- **Spherical Harmonics** for angular/directional information.

### Transformer Architecture
The `model/transformer.py` implements a customized Transformer where the attention scores are augmented by the encoded relative position bias, allowing the model to be rotationally and translationally aware.

## Citation

If you use this code in your research, please cite our paper:
> **Phonon Density of States Prediction from Atomic Relative Position Encoding Attention Transformer.**

---
*Developed for research at SHU.*
