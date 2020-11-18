
import numpy as np

from time import time

import scipy.sparse as sparse




import networkx as nx

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from utils.evaluate import evaluate
from utils.load_data import split_embedding


def node_to_degree(G_degree, SET):
    SET = list(SET)
    SET = sorted([G_degree[x] for x in SET])
    return SET

def svd_deepwalk_matrix(X, dim):
    u, s, v = sparse.linalg.svds(X, dim)
    # return U \Sigma^{1/2}
    print(s)

    res  = sparse.diags(np.sqrt(s)).dot(u.T).T

    return res

def cal_degree_dict(G_list, G, layer):
    G_degree = G.degree()
    degree_dict = {}
    degree_dict[0] = {}
    for node in G_list:
        degree_dict[0][node] = {node}
    for i in range(1, layer + 1):
        degree_dict[i] = {}
        for node in G_list:
            neighbor_set = []
            for neighbor in degree_dict[i - 1][node]:
                neighbor_set += nx.neighbors(G, neighbor)
            neighbor_set = set(neighbor_set)
            for j in range(i - 1, -1, -1):
                neighbor_set -= degree_dict[j][node]
            degree_dict[i][node] = neighbor_set
    for i in range(layer + 1):
        for node in G_list:
            if len(degree_dict[i][node]) == 0:
                degree_dict[i][node] = [0]
            else:
                degree_dict[i][node] = node_to_degree(G_degree, degree_dict[i][node])
    return degree_dict

def structing(layers, G1, G2, G1_degree_dict, G2_degree_dict, attribute, alpha, c,c_num):
    G1_nodes = list(G1.nodes())
    G2_nodes = list(G2.nodes())

    k1 = k2 = 1
    pp_dist_matrix = {}
    pp_dist_df = pd.DataFrame(np.zeros((G1.number_of_nodes(), G2.number_of_nodes())),
                              index=G1_nodes, columns=G2_nodes)

    for layer in range(layers + 1):
        L1 = [np.log(k1 * np.max(G1_degree_dict[layer][x]) + np.e) for x in G1_nodes]
        L2 = [np.log(k2 * np.max(G2_degree_dict[layer][x]) + np.e) for x in G2_nodes]
        pp_dist_matrix[layer, 0] = pd.DataFrame(
            np.transpose(np.array(L1 * G2.number_of_nodes()).reshape(-1, G1.number_of_nodes())),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_matrix[layer, 1] = pd.DataFrame(
            np.array(list(L2 * G1.number_of_nodes())).reshape(-1, G2.number_of_nodes()),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_df += abs(pp_dist_matrix[layer, 0] - pp_dist_matrix[layer, 1])
    for layer in range(layers + 1):
        L1 = [np.log(k1 * np.min(G1_degree_dict[layer][x]) + 1) for x in G1_nodes]
        L2 = [np.log(k2 * np.min(G2_degree_dict[layer][x]) + 1) for x in G2_nodes]
        pp_dist_matrix[layer, 0] = pd.DataFrame(
            np.transpose(np.array(L1 * G2.number_of_nodes()).reshape(-1, G1.number_of_nodes())),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_matrix[layer, 1] = pd.DataFrame(
            np.array(list(L2 * G1.number_of_nodes())).reshape(-1, G2.number_of_nodes()),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_df += abs(pp_dist_matrix[layer, 0] - pp_dist_matrix[layer, 1])
    pp_dist_df /= 2
    pp_dist_df = np.exp(-alpha * pp_dist_df)
    if attribute is not None:
        pp_dist_df = c * pp_dist_df + attribute * (1 - c)
    struc_neighbor1 = {}
    struc_neighbor2 = {}
    struc_neighbor_sim1 = {}
    struc_neighbor_sim2 = {}

    struc_neighbor_sim1_score = {}
    struc_neighbor_sim2_score = {}
    for i in range(G1.number_of_nodes()):
        pp = pp_dist_df.iloc[i, np.argsort(-pp_dist_df.iloc[i, :])]
        struc_neighbor1[G1_nodes[i]] = list(pp.index[:c_num])
        struc_neighbor_sim1[G1_nodes[i]] = np.array(pp[:c_num])
        # tmp = [math.pow(i,2) for i in struc_neighbor_sim1[G1_nodes[i]]]
        # struc_neighbor_sim1[G1_nodes[i]] = tmp / np.sum(tmp)
        struc_neighbor_sim1_score[G1_nodes[i]] = struc_neighbor_sim1[G1_nodes[i]]
        struc_neighbor_sim1[G1_nodes[i]] /= np.sum(struc_neighbor_sim1[G1_nodes[i]])
    pp_dist_df = pp_dist_df.transpose()
    for i in range(G2.number_of_nodes()):
        pp = pp_dist_df.iloc[i, np.argsort(-pp_dist_df.iloc[i, :])]
        struc_neighbor2[G2_nodes[i]] = list(pp.index[:c_num])
        struc_neighbor_sim2[G2_nodes[i]] = np.array(pp[:c_num])
        # tmp = [math.pow(i,2)for i in struc_neighbor_sim2[G2_nodes[i]]]
        # struc_neighbor_sim2[G2_nodes[i]] = tmp / np.sum(tmp)
        struc_neighbor_sim2_score[G2_nodes[i]] = struc_neighbor_sim2[G2_nodes[i]]
        struc_neighbor_sim2[G2_nodes[i]] /= np.sum(struc_neighbor_sim2[G2_nodes[i]])

    return struc_neighbor1, struc_neighbor2, struc_neighbor_sim1, struc_neighbor_sim2,struc_neighbor_sim1_score, struc_neighbor_sim2_score

def get_random_walk_matrix(TRANS_P, DIS_D_R, window_size, b=3.0):
    M = np.eye(TRANS_P.shape[0])
    S = np.zeros_like(TRANS_P)
    s = time()
    M = sparse.coo_matrix(M)
    S = sparse.coo_matrix(S)
    for r in range(window_size):
        print(r)
        M = M.dot(TRANS_P)
        S = S + M
    e = time()
    print(e-s)
    S = S.dot(DIS_D_R) + DIS_D_R.dot(S.T)

    # 选用b=3
    S = S / (2 * window_size * b)
    # S所有元素均大于0
    S[S == 0] = 1e-12
    S = np.log(S)
    return S

def DCMF(g1,g2,original_to_new,layer=3,q=0.2,alpha=5,c=0.5):

    Node1 = nx.number_of_nodes(g1)
    Node2 = nx.number_of_nodes(g2)
    attribute = None
    adj1 = nx.to_numpy_array(g1, nodelist=list(range(Node1)))
    adj2 = nx.to_numpy_array(g2, nodelist=list(range(Node2)))

    G1_degree_dict = cal_degree_dict(list(g1.nodes()), g1, layer)
    G2_degree_dict = cal_degree_dict(list(g2.nodes()), g2, layer)
    struc_neighbor1, struc_neighbor2, struc_neighbor_sim1, struc_neighbor_sim2, \
    struc_neighbor_sim1_score, struc_neighbor_sim2_score = \
        structing(layer, g1, g2, G1_degree_dict, G2_degree_dict, attribute, alpha, c, 10)

    # 构造转移矩阵
    P_TRANS1 = np.zeros((Node1 + Node2, Node1 + Node2))
    P_TRANS2 = np.zeros((Node1 + Node2, Node1 + Node2))
    D1 = np.sum(adj1, axis=1).reshape(-1, 1)
    D2 = np.sum(adj2, axis=1).reshape(-1, 1)
    adj1_hat = adj1 / D1
    adj2_hat = adj2 / D2

    for edge in g1.edges():
        P_TRANS1[edge[0], edge[1]] = adj1_hat[edge[0], edge[1]]
        P_TRANS1[edge[1], edge[0]] = adj1_hat[edge[1], edge[0]]
    for edge in g2.edges():
        P_TRANS1[edge[0] + Node1, edge[1] + Node1] = adj2_hat[edge[0], edge[1]]
        P_TRANS1[edge[1] + Node1, edge[0] + Node1] = adj2_hat[edge[1], edge[0]]
    for key in struc_neighbor_sim1.keys():

        for index, neighbor in enumerate(struc_neighbor1[key]):
            P_TRANS2[key, neighbor + Node1] = struc_neighbor_sim1[key][index]

    for key in struc_neighbor_sim2.keys():

        for index, neighbor in enumerate(struc_neighbor2[key]):
            P_TRANS2[key + Node1, neighbor] = struc_neighbor_sim2[key][index]

    cross_switch_alpha = np.zeros_like(P_TRANS2)
    for key in struc_neighbor_sim1.keys():
        if struc_neighbor_sim1[key][0] - struc_neighbor_sim1[key][1] >= 0.15:
            new_q = min(struc_neighbor_sim1[key][0] + q, 1)
        else:
            new_q = q
        cross_switch_alpha[key, :Node1] = 1 - new_q
        cross_switch_alpha[key, Node1:] = new_q

    for key in struc_neighbor_sim2.keys():
        if struc_neighbor_sim2[key][0] - struc_neighbor_sim2[key][1] >= 0.15:
            new_q = min(struc_neighbor_sim2[key][0] + q, 1)
        else:
            new_q = q
        cross_switch_alpha[key + Node1, :Node1] = new_q
        cross_switch_alpha[key + Node1, Node1:] = 1 - new_q

    P_TRANS = (P_TRANS1 + P_TRANS2) * cross_switch_alpha

    #
    P_TRANS = np.maximum(P_TRANS, P_TRANS.T)
    # P_TRANS = (P_TRANS + P_TRANS.T)/2.0

    P_TRANS = P_TRANS / np.sum(P_TRANS, axis=1, keepdims=True)
    # 计算平稳分布：
    tmp = (P_TRANS.sum(axis=0) / np.sum(P_TRANS)).tolist()

    D_R = np.zeros_like(P_TRANS)
    Node = Node1 + Node2
    # 初始采样概率，应该按照
    for i in range(len(tmp)):
        D_R[i, i] = 1 / tmp[i]

    # windows sizw小于5
    M = get_random_walk_matrix(P_TRANS, D_R, 20, 5)

    print("M计算完成")
    # 对于cora
    res = svd_deepwalk_matrix(M, dim=64)

    print("得到向量")
    e1, e2 = split_embedding(res, Node1)
    sim = cosine_similarity(e1, e2)
    acc = evaluate(sim, ans_dict=original_to_new)
    print(acc)