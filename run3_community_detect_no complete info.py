import numpy as np
import pandas as pd
import json
import networkx

from PRAD.tools.tools import *
from PRAD.model.Louvain import *
from PRAD.model.Bigclam import *
from PRAD.tools.agm import *
from PRAD.model.GN import *
from PRAD.model.Markov import *
from PRAD.model.CPM import *

# parameter
project = 'PRAD_no_complete'
# file_path = 'D:/anaconda3/envs/torch/Lib/site-packages/PRAD/example_data/processed/'
file_path = 'E:/PRAD/Step 1 bioinformatics process/Step 15. network analysis/1. network analysis/processed_data/'
save_path = 'E:/PRAD/Step 1 bioinformatics process/Step 15. network analysis/1. network analysis/community/no_complete/'
net_path = 'E:/PRAD/Step 1 bioinformatics process/Step 15. network analysis/1. network analysis/biological network/'
community_top = 8
over_community_num = 8
step_size = 0.01
threshold = 0.01
method = 'CPM' # ['louvain','GN','Markov','CPM']
dimension = 4
numIter = 2
power=2
inflation=2
c_num=2

## load data
edge = pd.read_csv(net_path+'neighbor_edges.csv', encoding='utf-8')
# feat_data = pd.read_csv(file_path+'node_feature.csv', encoding='utf-8', index_col=0)
feat_data = pd.read_csv(file_path+'2022-04-20_Embed_30.csv', encoding='utf-8', index_col=0)

## extract labeled node
label = pd.read_csv(net_path + 'neighbor_node_label.csv', encoding='utf-8')
label_node = list(label['node'][label['feature'] == 1])
edge = edge[edge['Source'].isin(label_node) | edge['Target'].isin(label_node)]
feat_data = feat_data[feat_data.index.isin(label_node)]

edge, node_map, node_map_trans, feat_data, label_map, label_dict = tidy_input_data(edge, feat_data)
edge_weighted,G,G_bigclam,G_adj = construct_weighted_network(edge=edge, feat_data=feat_data, norm=True,save_path=save_path)

if __name__ == '__main__':
    if method != 'CPM':
        ## non-overlap community
        if method == 'louvain':
            algorithm = Louvain(G)
            communities = sorted(algorithm.execute(), key=lambda b: -len(b))
            # save
            communities_csv = save_community(communities, node_map_trans, save_path, project, method)

        elif method == 'GN':
            algorithm = GN_w(G_bigclam)
            algorithm.run()
            algorithm.add_group()
            communities = algorithm.partition
            communities_csv = save_community(algorithm.partition, node_map_trans, save_path, project, method)

        else:
            communities = markovCluster(adjacencyMat=G_adj, dimension=dimension, numIter=numIter,power=power,
                                        inflation=inflation,save_path=save_path, c_num=c_num,node_map_trans=node_map_trans,
                                        project=project, method=method)

        ## BigCLAM
        nodes = range(len(feat_data))
        for c in range(community_top):
            over_community = []
            for n in range(community_top):
                over_community.append(community(communities[n], 0.8))
        G = AGM(nodes, over_community)
        F = bigclam(G, over_community_num, step_size, threshold)
        big_community = save_bigclam_community(F, node_map_trans, over_community_num, save_path, project)

        print('-----------Finished!-----------')

    else:
        communities = get_percolated_cliques(G_bigclam,c_num,node_map_trans,save_path, project, method)
        print('-----------Finished!-----------')
