import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
import scipy.linalg as linalg
# G = G_bigclam
# k = 35
def partition(G, k, normalized=False):
    A = nx.to_numpy_array(G)
    D = degree_matrix(G)
    L = D - A
    Dn = np.power(np.linalg.matrix_power(D, -1), 0.5)
    L = np.dot(np.dot(Dn, L), Dn)
    if normalized:
        pass
    eigvals, eigvecs = linalg.eig(L)
    n = len(eigvals)

    dict_eigvals = dict(zip(eigvals, range(0, n)))
    k_eigvals = np.sort(eigvals)[0:k]
    eigval_indexs = [dict_eigvals[k] for k in k_eigvals]
    k_eigvecs = eigvecs[:, eigval_indexs]
    k_eigvecs = np.asarray(k_eigvecs).astype(float)
    result = KMeans(n_clusters=k).fit_predict(k_eigvecs)
    return result


def degree_matrix(G):
    n = G.number_of_nodes()
    V = [node for node in G.nodes()]
    D = np.zeros((n, n))
    for i in range(n):
        node = V[i]
        d_node = G.degree(node)
        D[i][i] = d_node
    return np.array(D)


if __name__ == '__main__':
    filepath = r'.\football.gml'

    G = nx.read_gml(filepath)
    k = 12
    a = partition(G, k)
