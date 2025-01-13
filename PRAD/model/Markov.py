import numpy as np
import pandas as pd

from PRAD.tools.tools import *

def markovCluster(adjacencyMat, dimension, numIter, power, inflation,save_path, c_num, node_map_trans, project, method):
    columnSum = np.sum(adjacencyMat, axis=0)
    probabilityMat = adjacencyMat / columnSum

    # Expand by taking the e^th power of the matrix.
    def _expand(probabilityMat, power):
        expandMat = probabilityMat
        for i in range(power - 1):
            expandMat = np.dot(expandMat, probabilityMat)
        return expandMat

    expandMat = _expand(probabilityMat, power)

    # Inflate by taking inflation of the resulting
    # matrix with parameter inflation.
    def _inflate(expandMat, inflation):
        powerMat = expandMat
        for i in range(inflation - 1):
            powerMat = powerMat * expandMat
        inflateColumnSum = np.sum(powerMat, axis=0)
        inflateMat = powerMat / inflateColumnSum
        return inflateMat

    inflateMat = _inflate(expandMat, inflation)

    for i in range(numIter):
        expand = _expand(inflateMat, power)
        inflateMat = _inflate(expand, inflation)
    print(inflateMat)
    inflateMat_select = inflateMat[sum(inflateMat!=0) >= c_num]
    community = []
    for l in range(inflateMat_select.shape[0]):
        c_number = list(np.argwhere(inflateMat_select[l] != 0).reshape(-1))
        if c_number not in community:
            community.append(c_number)
    community_csv = save_community(community, node_map_trans, save_path, project, method)
    inflateMat_csv = pd.DataFrame(inflateMat)
    inflateMat_csv.to_csv(save_path + project + '_'+ method + '_inflateMat_community.csv', index=True, header=False)

    return community




