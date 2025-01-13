import networkx as nx
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from PRAD.tools.tools import _split_data
from sklearn.preprocessing import MinMaxScaler

class DataCenter(object):

    def load_data(self, dataSet, weight, file_path):
        file_path = r'E:\前端开发\北京瑞恩科软科技中心\功能项面需求\功能项面1需求'
        # parameter
        save_path = file_path+'/示例数据/Run2_output/processed_data/'
        feat_data = []
        labels = []

        ## load file
        features = pd.read_csv(file_path+'/示例数据/Input_file/node_feature.csv', encoding='utf-8', index_col=0)
        edge = pd.read_csv(file_path+'/示例数据/Run1_output/neighbor_edges.csv', encoding='utf-8')

        ## tidying node label
        allnode = list(set(pd.concat([edge['Source'],edge['Target']])))
        nolabelnode = []
        for n in allnode:
            if n not in list(features.index):
                nolabelnode.append(n)
        print(len(nolabelnode))

        nofeatures = pd.DataFrame(np.zeros((len(nolabelnode),features.shape[1])), columns=features.columns, index=nolabelnode)
        node_label = pd.concat([
            pd.DataFrame({
                'node':nolabelnode,
                'feature':0
            }),
            pd.DataFrame({
                'node': list(features.index),
                'feature': 1
            })
        ])
        features = pd.concat([features,nofeatures])

        ## calculated topo
        import networkx
        G = nx.from_pandas_edgelist(edge, source='Source', target='Target')
        degree = nx.degree_centrality(G)
        eigenvector = nx.eigenvector_centrality(G)
        close = nx.closeness_centrality(G)
        between = nx.betweenness_centrality(G)
        print('Finished!')

        features['degree'] = features.index.map(degree)
        features['eigenvector'] = features.index.map(eigenvector)
        features['close'] = features.index.map(close)
        features['between'] = features.index.map(between)

        features = features.fillna(0)

        # scaling
        scaler = MinMaxScaler()
        scaler = scaler.fit(features)
        result_ = scaler.fit_transform(features)
        features = pd.DataFrame(result_,columns=features.columns,index=features.index)
        assert ~pd.DataFrame.isnull(features).values.any(),'-------The faature data exists NaN-------'

        ## tidy node
        # edge
        edge['link'] = edge['Source'] + "_" + edge['Target']
        edge = edge.drop_duplicates('link',keep='first').reset_index(drop=True)
        features.iloc[:,0:4] = features.iloc[:,0:4]*weight
        s_node = edge.iloc[:,[0,2]]
        s_node.columns = ['node', 'type']
        t_node = edge.iloc[:,[1,3]]
        t_node.columns = ['node', 'type']
        node = pd.concat([s_node, t_node], axis=0)
        node.drop_duplicates(subset=['node'], keep='first', inplace=True)
        node = node.reset_index(drop=True)

        ## valid data
        valid_node = set(node['node']).intersection(features.index)
        num_node = len(valid_node)
        node = node[node['node'].isin(valid_node)]
        node_label = node_label[node_label['node'].isin(valid_node)]
        edge = edge[(edge['Source'].isin(valid_node) & edge['Target'].isin(valid_node))]
        features = features[features.index.isin(valid_node)]

        assert sum(features.index.isin(node['node'])) == num_node == node_label.shape[0]

        ## node map
        node_map = dict(zip(node['node'], node.index))
        node_label.index = node_label['node'].map(node_map)
        node_label = node_label.sort_index()
        features.index = features.index.map(node_map)
        features = features.sort_index()
        feat_map = dict(zip(range(features.shape[1]), features.columns))
        label_map = dict(zip(node_label.index, node_label['feature']))
        label = np.array(node_label['feature'])

        num_f_node = sum(label == 1)
        num_n_f_node = sum(label == 0)
        num_node = label.shape[0]
        assert num_f_node + num_n_f_node == num_node
        print('feature_node:{}, non_feature_node:{}, features_num:{}'.format(num_f_node, num_n_f_node, features.shape[0]))

        ## features info, the f_node: true profile; non_f_node: zero
        feat_data = np.array(features)
        f_labels = label

        print('feat_shape:{}, labels:{}'.format(feat_data.shape, f_labels.shape))

        ## construct adj_matrix
        adj_lists = defaultdict(set)

        for l in edge.index:
            info = np.array(edge.loc[l])
            source = node_map[info[0]]
            target = node_map[info[1]]
            adj_lists[source].add(target)
            adj_lists[target].add(source)

        assert feat_data.shape[0] == f_labels.shape[0] == num_node == len(adj_lists)

        ## split train, valid and test
        test_index, valid_index, train_index = _split_data(feat_data.shape[0], test_split=0, val_split=3)
        print('Train_node:{}, Valid_Node:{}, Test_Node:{}'.format(len(train_index), len(valid_index), len(test_index)))

        setattr(self, dataSet + '_train', train_index)
        setattr(self, dataSet + '_valid', valid_index)
        setattr(self, dataSet + '_test', test_index)

        setattr(self, dataSet + '_feats', feat_data)
        setattr(self, dataSet + '_labels', f_labels)
        setattr(self, dataSet + '_adj_lists', adj_lists)
        setattr(self, dataSet + '_node_map', node_map)
        setattr(self, dataSet + '_label_map', label_map)
        setattr(self, dataSet + '_node', node)
        setattr(self, dataSet + '_feat_map', feat_map)
