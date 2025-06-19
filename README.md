MODA: A Graph Convolutional Network-Based Multi-Omics Integration Framework
Author: Jinhui Zhao, Yanyan Zhou, Han Bao, Xinjie Zhao, Xinxin Wang, Chunxia Zhao, Wangshu Qin, Xin Lu, Guowang Xu

Abstract
Advances in omics technologies provide unprecedented opportunities for systems biology, yet integrating multi-omics data remains challenging due to its complexity, heterogeneity, and the sparsity of prior knowledge networks. Here, we introduce MODA (Multi-Omics Data Integration Analysis), a framework that:

Incorporates prior knowledge to identify hub molecules and pathways.

Uses graph convolutional networks (GCNs) with attention mechanisms to capture molecular relationships.

Ranks key molecules via a feature-selective layer.

Detects core functional modules via an overlapping community detection algorithm.

Results:
✔ Outperforms 7 existing multi-omics integration methods in classification.
✔ Achieves superior stability in pan-cancer datasets.
✔ Identifies carnitine and palmitoylcarnitine (regulated by BBOX1) as key players in prostate cancer.

Applications: Precision medicine, disease mechanism discovery.

MODA User Manual
1. Introduction
MODA is a free platform for integrating and analyzing multi-omics data, leveraging:

Genome-Scale Metabolic Network (GSMN)

Machine learning coefficients

Copy Number Profile Matching (CPM) algorithm

Key Features:
🔹 Deep learning-based novel molecule discovery
🔹 Extraction of key functional modules
🔹 Identification of hub molecules with significant expression fluctuations

2. Manual
2.1 Predicted Metabolic Flux
2.1.1 Set Parameters
yaml
output_folder: 'PRADall'  
input_folder: 'PRADall'  
2.1.2 Input Files
File	Description
_OPTIONS_.xlsx	Example parameter file
_TEMPLATE_.xls	Example template file
PRAD_RNAseq_FPKM all data.txt	RNA-seq profile
2.2 Multi-Omics Data Integration
2.2.1 Set Parameters
Embedding
yaml
depth: 2               # Default: 2, Suggested: 1-4  
dataSet: 'PRAD'        # Your project name  
agg_func: 'MEAN'       # Default  
epochs: 30             # Default: 30, Suggested: 20-60  
b_sz: 50               # Default: 50, Suggested: 32-128  
cuda: False            # Default: False, Options: True/False  
learn_method: 'sup'    # Default  
Module Detection
yaml
project: 'PRAD'        # Your project name  
method: 'CPM'          # Default  
threshold: 0.01        # Default  
spectral_k: 40         # Default  
2.2.2 Input Files
File	Description
seed node.csv	Seed nodes file
node_feature.csv	Node features file
Output Files
📂 Predicted Metabolic Flux → PRADall_product rate_all.csv
📂 Embedding Results → PRAD_Embed_score.csv
📂 Module Detection → PRAD_CPM_community.csv
