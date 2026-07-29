# BDIP-Net: Dual-Interaction Graph Learning for Property Prediction of  Bilayer Materials


![Overview of the Workflow](workflow1.jpg)

## Structural Optimization

Structural optimization is performed using MatterSim.

### Environment Setup

```bash
conda create -n mattersim python=3.10 -y
conda activate mattersim
```

### Install PyTorch

```bash
pip install torch==2.5.1 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
```

### Install Required Packages

```bash
pip install mattersim ase pymatgen pandas numpy scipy dftd3
```

### Run Structural Optimization

```bash
python structural_optimization.py
```
