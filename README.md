# Introduction
abfeflow: non-equilibrium absolute binding free energies with GROMACS
An end-to-end workflow for absolute binding free energies (ABFE) of a neutral ligand, using non-equilibrium (fast-growth) alchemical transitions in GROMACS. Analysis is pmx-compatible: BAR, Crooks Gaussian intersection and Jarzynski, validated against pmx analyse to ±0.01 kcal/mol.
User provides the bound-state production run, its topology and its .mdp, plus one YAML file. abfeflow does the rest!

# Installation
conda create -n abfe python=3.11 -y
conda activate abfe    
pip install -e abfeflow              
abfeflow --version                         # 0.2.0
