import numpy as np

def getCE(node,edgeList):
    ce_list = []
    node_set = set()
    edge_set = set()
    #找当前node的直接邻接点，存入node_set的集合中
    for edge in edgeList:
        if edge[0] == node:
            node_set.add(edge[1])
        elif edge[1] == node:
            node_set.add(edge[0])
    #找邻接点集合中的点所构成的边的数目
    for edge in edgeList:
        if edge[0] in node_set and edge[1] in node_set:
            s = edge[0]+edge[1]
            edge_set.add(s)

    neighbourNodeNum = len(node_set)  #邻接点结点个数
    neighbouredgeNum = len(edge_set)  #邻接点构成的边的条数
    print("neighbour node Num:", neighbourNodeNum)
    print("neighbour edge Num:", neighbouredgeNum)
    ceNum = 0
    #求聚类系数的公式
    if neighbourNodeNum > 1:
        ceNum = 2*neighbouredgeNum/((neighbourNodeNum-1)*neighbourNodeNum) #无向图要乘2，有向图不需要乘2
    ce_list.append(ceNum)
    node_set.clear()
    edge_set.clear()

    return ce_list

def getAverageCE(ce_list):
    total = 0
    for ce in ce_list:
        total += ce[0]
    return total/len(ce_list)

def get_ce_list(edge,save_path,name):
    node_list = []
    edge_list2 = []
    ce_list = []

    for index in edge.index:
        splitlist = list(edge.loc[index])[0:2]
        edge_list2.append(splitlist)
    node_s = np.array(edge['Source'])
    node_t = np.array(edge['Target'])
    node = set(np.hstack((node_s,node_t)))
    node_list = list(node)

    for node in node_list:
        print(node)
        ce = getCE(node, edge_list2)
        ce_list.append(ce)
    assert len(ce_list) == len(node_list)
    f = open(save_path+name, 'w')
    for l in range(len(ce_list)):
        f.writelines(node_list[l]+ ' ' + str(ce_list[l][0]) +'\n')
    f.close()

    return ce_list










