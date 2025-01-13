import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import yaml
import os

from PRAD.tools.tools import k_neighbor_search

# parameter
file_path = os.path.abspath(os.path.join(os.path.dirname('__file__'),os.path.pardir))
def par_yaml():
    filepath = file_path+'/运行代码/run1config.yaml'     # 文件路径,这里需要将a.yaml文件与本程序文件放在同级目录下
    with open(filepath, 'r') as f:     # 用with读取文件更好
        configs = yaml.load(f, Loader=yaml.FullLoader) # 按字典格式读取并返回
    return configs
par = par_yaml()

### load file
edges = pd.read_csv(file_path+'/示例数据/Raw_biological_network/edges_raw.csv', encoding='utf-8')
seed_node = pd.read_csv(file_path+'/示例数据/Input_file/seed node.csv', encoding='utf-8')

### remove mirna
edges_pro = edges[(edges['Source_Type'] != 'miRNA') & (edges['Target_Type'] != 'miRNA')]
seed_node = seed_node[seed_node['type'] != 'miRNA']

### construct network
G = nx.from_pandas_edgelist(edges_pro, 'Source', 'Target')
seed = np.array(seed_node['node'])

### remove the node which their degree > 100
degree = dict(nx.degree(G))
node_t = seed_node['node'].tolist()
for k in degree:
    if degree[k] < 100:
        if k not in node_t:
            node_t.append(k)

print('The final targeted node in biological network:{}'.format(len(node_t)))

edges_final = edges[(edges['Source'].isin(node_t)) & (edges['Target'].isin(node_t))]
G = nx.from_pandas_edgelist(edges_final, 'Source', 'Target')

### extract the 5-neighbor for each seed node, and extract sub-graph based on the target node (seed + neighbor)
t_node = k_neighbor_search(seed, par['depth'], G)
print('target node: {}'.format(len(t_node)))
edges_neighbor = edges_final[(edges_final['Source'].isin(t_node)) | (edges_final['Target'].isin(t_node))]
print('edges_neighbor: {}'.format(edges_neighbor.shape))
# G = nx.from_pandas_edgelist(edges_neighbor, 'Source', 'Target')
# pos = nx.layout.spring_layout(G)
# nx.draw(G,pos=pos,node_size=12)
# plt.show()

seed_node.to_csv(file_path+'/示例数据/Run1_output/targeted_node.csv', index=False)
edges_neighbor.to_csv(file_path+'/示例数据/Run1_output/neighbor_edges.csv', index=False)
