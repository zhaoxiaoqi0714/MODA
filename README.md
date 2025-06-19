# MODA: A Graph Convolutional Network-Based Multi-Omics Integration Framework for Unraveling Hub Molecules and Disease Mechanisms

**Author:** Jinhui Zhao, Yanyan Zhou, Han Bao, Xinjie Zhao, Xinxin Wang, Chunxia Zhao, Wangshu Qin, Xin Lu, Guowang Xu

## Abstract

Advances in omics technologies provide unprecedented opportunities for systems biology, yet integrating multi-omics data remains challenging due to its complexity, heterogeneity, and the sparsity of prior knowledge networks. Here, we introduce a multi-omics data integration analysis (MODA) framework that fully incorporates prior knowledge to identify hub molecules and pathways and elucidate biological mechanisms. By leveraging multiple machine learning approaches, MODA transforms raw omics data into a feature importance matrix that is mapped onto a biological knowledge graph to mitigate omics data noise. Then, it uses graph convolutional networks with attention mechanisms to capture intricate molecular relationships and rank key molecules via a feature-selective layer. Ultimately, MODA transcends the limitations of predefined pathway annotations by employing an overlapping community detection algorithm to extract core functional modules that are involved in multiple pivotal disease pathways. Systematic evaluations show that MODA outperforms seven existing multi-omics integration methods in classification performance while maintaining biological interpretability. Moreover, MODA achieves superior stability in pan-cancer datasets. Application to the multi-omics datasets of prostate cancer reveals a key role for carnitine and palmitoylcarnitine, regulated by BBOX1 in the progression of prostate cancer. Population samples and in vitro experiments further validate these findings. With high data utilization efficiency and low computational cost, MODA serves as a robust tool for uncovering novel disease mechanisms and advancing precision medicine.

## MODA User Manual

### 1. Introduction

Multi-Omics Data Integration and Analysis (MODA) offers a free platform for researchers to seamlessly integrate and analyze multi-omics data. Leveraging Genome-Scale Metabolic Network (GSMN), machine learning coefficients, and the Copy Number Profile Matching (CPM) algorithm, our tool not only facilitates a more comprehensive integration of metabolomics with other omics but also uncovers novel molecules based on deep learning. With a focus on extracting key modules and identifying hub molecules with significant expression fluctuations, our tool provides researchers with an advanced and robust solution for in-depth analysis and interpretation of multi-omics data.

### 2. Manual

#### 2.1 Predicted Metabolic Flux

##### 2.1.1 Set Parameter

- **output_folder:** 'PRADall'
- **input_folder:** 'PRADall'

##### 2.1.2 Input files

- **Parameters (Example: _OPTIONS_.xlsx)**: Downloaded example data and revised information based on your project.
- **Template (Example: _TEMPLATE_.xls)**: Downloaded example data and revised information based on your project.
- **RNA-seq profile (Example: PRAD_RNAseq_FPKM all data.txt)**: Downloaded example profile and tidied your data.

#### 2.2 Integration Multi-omics data

##### 2.2.1 Set Parameter

###### 2.2.1.1 Embedding

- **depth:** 2 (default: 2, suggestion: 1-4)
- **dataSet:** 'PRAD' (Your project name)
- **agg_func:** 'MEAN' (default)
- **epochs:** 30 (default: 30, suggestion: 20-60)
- **b_sz:** 50 (default: 50, suggestion: 32-128)
- **seed:** 777 (default)
- **cuda:** False (default: False, suggestion: True/False)
- **learn_method:** 'sup' (default)
- **unsup_loss:** 'normal' (default)
- **name:** 'debug' (default)
- **num_layers:** 2 (default)
- **hidden_emb_size:** 128 (default)
- **gcn:** False (default)
- **loss_mean_inital:** 100 (default)
- **weight:** 3 (default)

###### 2.2.1.2 Module detection

- **project:** 'PRAD' (Your project name)
- **community_top:** 8 (default)
- **over_community_num:** 8 (default)
- **step_size:** 0.01 (default)
- **threshold:** 0.01 (default)
- **method:** 'CPM' (default)
- **dimension:** 4 (default)
- **numIter:** 2 (default)
- **power:** 2 (default)
- **inflation:** 2 (default)
- **c_num:** 2 (default)
- **spectral_k:** 40 (default)

##### 2.2.2 Input files

- **Seed nodes (Example: seed node.csv)**: Downloaded example profile and tidied your data.
- **Node Features (Example: node_feature.csv)**: Downloaded example profile and tidied your data.
