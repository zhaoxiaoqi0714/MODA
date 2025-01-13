import numpy as np
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt

from PRAD.tools.tools import *

def get_percolated_cliques(G, k,node_map_trans, save_path, project, method):
    cliques = list(frozenset(c) for c in nx.find_cliques(G) if len(c) >= k)  # 找出所有大于k的最大k-派系
    #     print(cliques)
    matrix = np.zeros((len(cliques), len(cliques)))  # 构造全0的重叠矩阵
    #     print(matrix)
    for i in range(len(cliques)):
        for j in range(len(cliques)):
            if i == j:  # 将对角线值大于等于k的值设为1，否则设0
                n = len(cliques[i])
                if n >= k:
                    matrix[i][j] = 1
                else:
                    matrix[i][j] = 0
            else:  # 将非对角线值大于等于k的值设为1，否则设0
                n = len(cliques[i].intersection(cliques[j]))
                if n >= k - 1:
                    matrix[i][j] = 1
                else:
                    matrix[i][j] = 0

    #     print(matrix)
    #     for i in matrix:
    #         print(i)

    #     l = [-1]*len(cliques)
    l = list(range(len(cliques)))  # l（社区号）用来记录每个派系的连接情况，连接的话值相同
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] == 1 and i != j:  # 矩阵值等于1代表，行派系与列派系相连，让l中的行列派系社区号变一致
                l[j] = l[i]  # 让列派系与行派系社区号相同（划分为一个社区）
    #     print(l)
    q = []  # 用来保存有哪些社区号
    for i in l:
        if i not in q:  # 每个号只取一次
            q.append(i)
    #     print(q)

    p = []  # p用来保存所有社区
    for i in q:
        print(frozenset.union(*[cliques[j] for j in range(len(l)) if l[j] == i]))  # 每个派系的节点取并集获得社区节点
        p.append(list(frozenset.union(*[cliques[j] for j in range(len(l)) if l[j] == i])))

    # save
    community_csv = save_community(p, node_map_trans, save_path, project, method)

    return p


def cal_Q(partition, G):  # 计算Q
    m = len(G.edges(None, False))  # 如果为真，则返回3元组（u、v、ddict）中的边缘属性dict。如果为false，则返回2元组（u，v）
    # print(G.edges(None,False))
    # print("=======6666666")
    a = []
    e = []
    for community in partition:  # 把每一个联通子图拿出来
        t = 0.0
        for node in community:  # 找出联通子图的每一个顶点
            t += len([x for x in G.neighbors(node)])  # G.neighbors(node)找node节点的邻接节点
        a.append(t / (2 * m))
    #             self.zidian[t/(2*m)]=community
    for community in partition:
        t = 0.0
        for i in range(len(community)):
            for j in range(len(community)):
                if (G.has_edge(community[i], community[j])):
                    t += 1.0
        e.append(t / (2 * m))

    q = 0.0
    for ei, ai in zip(e, a):
        q += (ei - ai ** 2)
    return q


def add_group(p, G):
    num = 0
    nodegroup = {}
    for partition in p:
        for node in partition:
            nodegroup[node] = {'group': num}
        num = num + 1
    nx.set_node_attributes(G, nodegroup)


def setColor(G):
    color_map = []
    color = ['red', 'green', 'yellow', 'pink', 'blue', 'grey', 'white', 'khaki', 'peachpuff', 'brown']
    for i in G.nodes.data():
        if 'group' not in i[1]:
            color_map.append(color[9])
        else:
            color_map.append(color[i[1]['group']])
    return color_map
