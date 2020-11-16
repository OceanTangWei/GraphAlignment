import scipy.sparse as ss
import scipy
import numpy as np
import networkx as nx
import time
from sklearn.metrics.pairwise import cosine_similarity
# 仅适用于存在节点特征的情况
class FINAL(object):
    def __init__(self,g1,g2,node_attr1,node_attr2,edge_attr1,edge_attr2,prior_alignment,tol,iters):
        self.g1 = g1
        self.g2 = g2
        self.node_attr1 = node_attr1
        self.node_attr2 = node_attr2
        self.edge_attr1 = edge_attr1
        self.edge_attr2 = edge_attr2
        self.prior = prior_alignment
        self.N1, self.N2 = nx.number_of_nodes(self.g1), nx.number_of_nodes(self.g2)
        self.tol = tol
        self.iters = iters
    def initial_S(self):
        if self.prior is None:
            if self.node_attr1 is None:
                self.prior = np.ones((self.N1*self.N2,1))
            else:
                self.prior = cosine_similarity(self.node_attr1,self.node_attr2).reshape((self.N1*self.N2,1))
                
        else:
            self.prior = self.prior.reshape((self.N1*self.N2,1))
        return self.prior

    def align(self,):
        S = self.initial_S()
        N1,N2 = self.N1, self.N2
        # 获取邻接矩阵
        A1 = nx.to_scipy_sparse_matrix(self.g1,nodelist=range(N1))
        A2 = nx.to_scipy_sparse_matrix(self.g2,nodelist=range(N2))
        # 假设n个节点特征有K个维度，那么可以将其写成K个对角矩阵
        N = ss.csc_matrix((N1*N2,N2*N1))
        K = self.node_attr1.shape[-1]
        for k in range(K):
            k1 = ss.csc_matrix(np.diag(self.node_attr1[:,k]))
            k2 = ss.csc_matrix(np.diag(self.node_attr2[:,k]))

            N = N + ss.kron(k1,k2)
        # 求解D
        D = ss.csc_matrix((N1 * N2, 1))
        for k in range(K):
            k1 = ss.csc_matrix(np.diag(self.node_attr1[:, k]))
            k2 = ss.csc_matrix(np.diag(self.node_attr2[:, k]))

            left = np.dot(np.dot(k1,A1),k1).sum(axis=-1)
            right = np.dot(np.dot(k2,A2),k2).sum(axis=-1)
            D = D + ss.kron(left,right)
        D = D.todense().getA().reshape(-1)
        DD = 1. / (np.sqrt(D)+1e-12)
        DD[DD == np.inf] = 0
        DD = ss.diags(DD)
        DD = np.dot(N,DD)

        iter = 0
        while iter < self.iters:
            Q = np.dot(DD,S).reshape(N1,N2)
            print(Q.shape)








G1 = nx.erdos_renyi_graph(1000,0.0009)
G2 = nx.erdos_renyi_graph(1000,0.0008)
f1 = np.random.random((1000,10))
f2 = np.random.random((1000,10))
s = time.time()
a = FINAL(G1,G2,f1,f2,None,None,None,0.0,2)
a.align()
e = time.time()
print(e-s)