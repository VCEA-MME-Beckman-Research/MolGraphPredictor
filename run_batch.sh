#!/bin/bash
#SBATCH -J MolecularGNN
#SBATCH -o logs/outputs/MolecularGNN-%j.out
#SBATCH -e logs/outputs/MolecularGNN-%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Load necessary modules
module load python3/3.9.5
module load cuda/11.8.0
module load cudnn/8.9.4_cuda11.8

# Install Python dependencies
pip install --user -r requirements.txt

# Run the training script
python MolecularGNN.py --epochs 200
