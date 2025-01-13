import networkx as nx
import numpy as np
import pandas as pd
import time
import collections
import random

from PRAD.model.Bigclam import Reshape

def tidy_input_data(edge, feat_data):
    ## generate node index map and check data
    # check self-loop
    edge = edge[edge['Source'] != edge['Target']]
    assert sum(edge['Source'] == edge['Target']) == 0,'---------This data had self-loop! Please check data!---------'

    # valid data
    s_node = edge.iloc[:, [0, 2]]
    s_node.columns = ['node', 'type']
    t_node = edge.iloc[:, [1, 3]]
    t_node.columns = ['node', 'type']
    node = pd.concat([s_node, t_node], axis=0)
    node.drop_duplicates(subset=['node'], keep='first', inplace=True)
    node = node.reset_index(drop=True)

    if node.shape[0] == feat_data.shape[0]:
        print('---------The nodes in edges all had vaild features!---------')
    else:
        print('---------Not all nodes in edges had vaild features! The node should be delete or please check your data---------')
        valid_node = list(feat_data.index)
        edge = edge[(edge['Source'].isin(valid_node)) & (edge['Target'].isin(valid_node))]
        s_node = edge.iloc[:, [0, 2]]
        s_node.columns = ['node', 'type']
        t_node = edge.iloc[:, [1, 3]]
        t_node.columns = ['node', 'type']
        node = pd.concat([s_node, t_node], axis=0)
        node.drop_duplicates(subset=['node'], keep='first', inplace=True)
        node = node.reset_index(drop=True)
        feat_data = feat_data[feat_data.index.isin(list(node['node']))]

    assert node.shape[0] == feat_data.shape[0]

    # construct node map
    node_map = dict(zip(node['node'], node.index))
    node_map_trans = dict(zip(node.index, node['node']))
    label_dict = dict(zip(set(node['type']), range(len(set(node['type'])))))
    label_map = dict(zip(node.index, node['type'].map(label_dict)))
    feat_data.index = feat_data.index.map(node_map)
    feat_data = np.array(feat_data.sort_index()).tolist()
    edge['Source'] = edge['Source'].map(node_map)
    edge['Target'] = edge['Target'].map(node_map)
    edge = edge.iloc[:, [0, 1]]

    return edge, node_map, node_map_trans, feat_data, label_map, label_dict

def k_neighbor_search(seed, depth, G):
    node_list = []
    for s_node in seed:
        if s_node in G:
            output = {}
            layers = dict(nx.bfs_successors(G, source=s_node, depth_limit=depth))
            nodes = [s_node]
            for i in range(1 ,depth+1):
                output[i] = []
                for n in nodes:
                    output[i].extend(layers.get(n ,[]))
                nodes = output[i]
            all_node_list = []
            for k in range(1 ,depth +1):
                k_node_list = output[k]
                [all_node_list.append(n) for n in k_node_list]
            node_list.append(all_node_list)

    all_node = np.array(node_list[0])
    for l in range(1, len(node_list)):
        all_node = np.hstack((all_node, np.array(node_list[l])))

    return set(all_node)


def _split_data(num_nodes, test_split, val_split):
    rand_indices = np.random.permutation(num_nodes)

    if test_split != 0:
        test_size = num_nodes // test_split
        val_size = num_nodes // val_split
        train_size = num_nodes - (test_size + val_size)

        test_indexs = rand_indices[:test_size]
        val_indexs = rand_indices[test_size:(test_size + val_size)]
        train_indexs = rand_indices[(test_size + val_size):]
    else:
        val_size = num_nodes // val_split
        train_size = num_nodes - val_size

        val_indexs = rand_indices[:val_size]
        train_indexs = rand_indices[val_size:]

        test_indexs = rand_indices

    return test_indexs, val_indexs, train_indexs

def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]

def info_save_result(dataCenter, ds, loss_result, valid_result, save_path, epoch):
    localtime = time.localtime(time.time())
    times = time.strftime("%Y-%m-%d", time.localtime())
    # save result
    loss_final = pd.DataFrame(loss_result, columns=['Loss'])
    loss_final['Batch'] = epoch + 1
    valid_final = pd.DataFrame(np.array(valid_result).reshape(-1), columns=['Loss'])
    valid_final['Batch'] = epoch + 1

    loss_final.to_csv(save_path + '{}_Train_Loss_{}.csv'.format(times,epoch+1), index=False)
    valid_final.to_csv(save_path + '{}_Valid_Loss_{}.csv'.format(times,epoch+1), index=False)

    return loss_final, valid_final

def emb_save_result(dataCenter, ds, embed_result, save_path, features, epoch,dataSet):
    localtime = time.localtime(time.time())
    times = time.strftime("%Y-%m-%d", time.localtime())
    # save result
    embed_final = np.zeros(shape=features.shape)
    for l in range(len(embed_result)):
        if sum(embed_result[l]) != 0:
            embed_final[l] = embed_result[l].cpu().detach().numpy()
    embed_final = pd.DataFrame(embed_final)
    embed_final.index = np.array([get_key(getattr(dataCenter, ds+'_node_map'), i) for i in embed_final.index])[:,0]
    embed_final.columns = embed_final.columns.map(getattr(dataCenter, ds+'_feat_map'))
    embed_final_score = embed_final
    embed_final_score['score'] = embed_final_score.apply(lambda x:x.sum(), axis=1)
    embed_final_score = embed_final_score[['score']]

    embed_final.to_csv(save_path + '{}_Embed.csv'.format(dataSet), index=True)
    embed_final_score.to_csv(save_path + '{}_Embed_score.csv'.format(dataSet), index=True)
    return embed_final

def save_community(communities, node_map_trans, save_path, project, method):
    communities_map = [[] for k in range(len(communities))]
    for l in range(len(communities)):
        t_community = pd.DataFrame(communities[l]).iloc[:, 0].map(node_map_trans)
        communities_map[l] = np.array(t_community)
        print('Community_{}:{}'.format(l, communities_map[l]))
    communities_csv = pd.DataFrame(communities_map)
    communities_csv.index = ['Community_' + str(i) for i in range(1, communities_csv.shape[0] + 1)]
    communities_csv.to_csv(save_path + project + '_'+ method + '_community.csv', index=True, header=False)

    return communities_map

def save_bigclam_community(F, node_map_trans, over_community_num, save_path, project):
    big_community = pd.DataFrame(Reshape(F))
    big_community.index = big_community.index.map(node_map_trans)
    big_community.columns = ['Over_Community_' + str(r) for r in range(1, over_community_num + 1)]
    big_community.to_csv(save_path + project + '_BigCLAM_community.csv', index=True, header=True)
    for c in range(big_community.shape[1]):
        print("{}:{}".format(big_community.columns[c], big_community.index[big_community.iloc[:, c] != 0]))

    return big_community

def cosine_similarity(x, y, norm):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    return 0.5 * cos + 0.5 if norm else cos


def construct_weighted_network(edge, feat_data, norm, save_path):
    G = collections.defaultdict(dict)
    G_bigclam = nx.Graph()
    edge['weight'] = 0

    print('Start to construct weighted-network')
    for l in range(edge.shape[0]):
        line_info = edge.iloc[l, :]
        feat_s = feat_data[int(line_info[0])]
        feat_t = feat_data[int(line_info[1])]
        edge.iloc[l, 2] = cosine_similarity(x=feat_s, y=feat_t, norm=norm)  # norm, linear scale
        # add info into Graph
        G[int(line_info[0])][int(line_info[1])] = edge.iloc[l, 2]
        G[int(line_info[1])][int(line_info[0])] = edge.iloc[l, 2]
        G_bigclam.add_weighted_edges_from([(int(line_info[0]),int(line_info[1]),edge.iloc[l, 2])])

    edge['Source'] = edge['Source'].astype('int')
    edge['Target'] = edge['Target'].astype('int')
    edge['weight'] = edge['weight'].astype('float')
    edge.to_csv(save_path+'edge_weight.csv', index=False)

    # output weighted-adj_matrix
    adj = np.array(nx.adjacency_matrix(G_bigclam).todense())

    print('Finished!')
    return edge,G,G_bigclam, adj

