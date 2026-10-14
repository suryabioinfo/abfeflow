# Introduction
abfeflow: non-equilibrium absolute binding free energies with GROMACS
An end-to-end workflow for absolute binding free energies (ABFE) of a neutral ligand, using non-equilibrium (fast-growth) alchemical transitions in GROMACS. Analysis is pmx-compatible: BAR, Crooks Gaussian intersection and Jarzynski, validated against pmx analyse to ±0.01 kcal/mol.
User provides the bound-state production run, its topology and its .mdp, plus one YAML file. abfeflow does the rest!

# Installation
1. conda create -n abfe python=3.11 -y

2. conda activate abfe    
3. pip install -e abfeflow              
4. abfeflow --version

# Using ABFEFLOW

1. abfeflow -c abfe.yaml setup
2. abfeflow -c abfe.yaml check
3. abfeflow -c abfe.yaml run
4. abfeflow -c abfe.yaml analyze
5. abfeflow -c abfe.yaml report                        


# The ABFE estimated for T4 -lysozyme bound to 2-propylephenol (PDB code: 3HTB), using the ABFEFLOW. Results are as below:

# Thermodynamic Cycle
<img width="1895" height="1049" alt="fig1_cycle" src="https://github.com/user-attachments/assets/0ca929bc-c518-463a-a8ed-b3cb48488115" />

# Comparison of Binding Free Energies
<img width="1358" height="734" alt="fig5_comparison" src="https://github.com/user-attachments/assets/4b3d3ee3-74ba-4f4d-92b7-888bcc08a9d0" />

# Convergence

<img width="2075" height="756" alt="fig3_convergence" src="https://github.com/user-attachments/assets/08672a18-8a02-481b-a071-d87c425948a8" />

# Work Distribution

<img width="2075" height="803" alt="fig2_work_distributions" src="https://github.com/user-attachments/assets/a481607b-d615-4675-aecb-f257315c9793" />
