import numpy as np
import pandas as pd
import json
import networkx as nx

from PRAD.tools.tools import *
from PRAD.model.Louvain import *
from PRAD.model.Bigclam import *
from PRAD.tools.agm import *
from PRAD.model.GN import *
from PRAD.model.Markov import *
from PRAD.model.CPM import *
from PRAD.model.Spectral import *

import yaml
import os

# parameter
filepath = os.path.abspath(os.path.join(os.path.dirname('__file__'),os.path.pardir))
def par_yaml():
    file = filepath +'/运行代码/run3config.yaml'     # 文件路径,这里需要将a.yaml文件与本程序文件放在同级目录下
    with open(file, 'r') as f:     # 用with读取文件更好
        configs = yaml.load(f, Loader=yaml.FullLoader) # 按字典格式读取并返回
    return configs
par = par_yaml()

project = par['project']
save_path = filepath + '/示例数据/Run3_output/'
net_path = filepath + '/示例数据/Run1_output/'
community_top = par['community_top']
over_community_num = par['over_community_num']
step_size = par['step_size']
threshold = par['threshold']
method = par['method'] # ['louvain','GN','Markov','CPM',’spectral]
dimension = par['dimension']
numIter = par['numIter']
power=par['power']
inflation=par['inflation']
c_num=par['c_num']
spectral_k = par['spectral_k']

## load data
edge = pd.read_csv(net_path+'neighbor_edges.csv', encoding='utf-8')
feat_data = pd.read_csv(filepath + '/示例数据/Run2_output/'+project+'_Embed.csv', encoding='utf-8', index_col=0)
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

        elif method == 'spectral':
            sc_com = partition(G_bigclam, spectral_k)
            communities = pd.DataFrame(list(sc_com), columns=['Community'])
            communities['node_index'] = list(G_bigclam.nodes)
            communities['node'] = communities['node_index'].map(node_map_trans)
            communities_map = [[] for k in range(spectral_k)]
            for c in range(spectral_k):
                communities_map[c] = np.array(communities['node'][communities['Community'] == c])
            communities_csv = pd.DataFrame(communities_map)
            communities_csv.index = ['Community_' + str(i) for i in range(1, spectral_k+1)]
            communities_csv.to_csv(save_path + project + '_' + method + '_community.csv', index=True, header=False)

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

