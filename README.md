# X2-GNN
X2-GNN is an attention based graph neural network architecture designed for utilizing the symmetrized one electron integral molecule feature. It's highlighted by the generalization ability.
## Requirements
the following python packages are used, and the environment used for experiments are provided in 'requirements.txt':\
numpy==1.23.5\
pyscf==2.2.1\
sympy==1.11.1\
torch==1.12.1\
torch-geometric==2.1.0\
torch-scatter==2.1.0
## How to use
the following content will help you to reproduce the results presented in X2-GNN paper.
### download dataset
 simply run the notebook 'datapre.ipynb' and it will download and unzip QM9 dataset and reorganize it into a single xyz file. These files can be found in directory './raw/'. Other datasets mention in the article are also provided in this dir in .xyz or .extxyz(supported by ASE) format.
### represent molecules as graph
 The graph representation of molecules and calculation of integrals are handled by 'qm9_allprop.py'. It stores the obtaining results in torch_geometric 'Data' objects and forms an 'InMemoryDataset' object. The processed dataset will be stored at './processed/'. To use deffirent dataset, just specify the corresponding input xyz file.
### train an energy model
 After the dataset is prepared, run 'train_ema.ipynb' and it will start training, hyperparameters are specified by a json file 'config.json'. the trainer will automatically generate log file in directory './log/'. Model checkpoint will be saved in directory './modelsaves/'
### load trained model
 A notebook 'load.ipynb' is provided to use model trained on QM9. In the notebook we test a trained model for U0 on curated OCELOT dataset. Other properties and dataset can be used by adjusting the config file.
