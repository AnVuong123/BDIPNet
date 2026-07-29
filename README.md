# BDIP-Net: Dual-Interaction Graph Learning for Property Prediction of  Bilayer Materials


![Overview of the Workflow](workflow1.jpg)

## Structural Optimization

Structural optimization is performed using MatterSim. Create and activate the environment:

```bash
conda create -n mattersim python=3.10 -y
conda activate mattersim
```

Install PyTorch with CUDA 12.1 and the required packages:

```bash
pip install torch==2.5.1 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

pip install mattersim ase pymatgen pandas numpy scipy dftd3
```

Run the structural optimization script:

```bash
python structural_optimization.py
```
