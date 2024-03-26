# DynGAT
code for 'A Dynamic Graph Convolutional Network for Anti-money Laundering', ICIC 2023


Authors: Tianpeng Wei, Biyang Zeng, Wenqi Guo, Zhenyu Guo, Shikui Tu, Lei Xu

# Directory description:
/datasets/raw: hold the raw data


/datasets/processed: hold the proccsed graph data generated from the raw version


/save: save log information and trained models


You can customize these settings through changing 'param' in 'config.py'.

# Datasets
The datasets are used in our paper.


--[Bitcoin-Elliptic]: (https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

--[AMLSim]: You may generate your own version from the beginning through (https://github.com/IBM/AMLSim), or download official samples from (https://github.com/IBM/AMLSim/wiki/Download-Example-Data-Set).


Place the raw datasets in /datasets/raw/{datasetname}


We also provide processed versions in ()

# Train and Test
The corresponding hyperparameters are defined in 'config.py'.


To begin training or testing, you may change the 'param['task]' in 'config.py' and use the following commands:

'''
python run.py
'''

The trained model are provided in the link above.
