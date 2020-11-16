import copy
import random
import networkx as nx
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import random

def create_align_graph(g, remove_rate, add_rate):
    max_deree = max([g.degree[i] for i in g.nodes()])
    edges = list(g.edges())
    nodes = list(g.nodes())
    remove_num = int(len(edges) * remove_rate)
    add_num = int(len(edges) * add_rate)
    random.shuffle(edges)
    random.shuffle(nodes)
    max_iters = (len(edges) + len(nodes)) * 2

    new_g = copy.deepcopy(g)
    while remove_num and max_iters:
        candidate_edge = edges.pop()
        if new_g.degree[candidate_edge[0]] > 1 and new_g.degree[candidate_edge[1]] > 1:
            new_g.remove_edge(candidate_edge[0], candidate_edge[1])
            remove_num -= 1
        max_iters -= 1

    max_iters = (len(edges) + len(nodes)) * 2
    while add_num and max_iters:
        n1 = random.choice(nodes)
        n2 = random.choice(nodes)
        if n1 != n2 and n1 not in new_g.neighbors(n2):
            if new_g.degree[n1] < max_deree - 1 or new_g.degree[n2] < max_deree - 1:
                new_g.add_edge(n1, n2)
                add_num -= 1
        max_iters -= 1
    return new_g


def shuffle_graph(g):
    original_nodes = list(g.nodes())
    new_nodes = copy.deepcopy(original_nodes)
    random.shuffle(new_nodes)
    original_to_new = dict(zip(original_nodes, new_nodes))
    new_graph = nx.Graph()
    for edge in g.edges():
        new_graph.add_edge(original_to_new[edge[0]], original_to_new[edge[1]])
    return new_graph, original_to_new


def get_node_degree_feature(g, node):
    # 获取度特征有很多种方法，这里就实现两种：
    # 取周围邻点的度，然后分别取它的最大值，最小值，均值，方差作为特征
    degrees = [g.degree[i] for i in g.neighbors(node)]
    min_v = min(degrees)
    max_v = max(degrees)
    mean_v = sum(degrees)/len(degrees)
    std_v = sum([(i-mean_v)**2 for i in degrees])/len(degrees)
    node_feat = np.array([min_v, max_v, mean_v, std_v]).reshape(1,-1)

    return node_feat


def get_degree_feature(g):
    node_num = nx.number_of_nodes(g)
    feature_vector = np.zeros((node_num, 4))
    for i in range(node_num):
        feature_vector[i, :] = get_node_degree_feature(g, i)
    return feature_vector


def find_top_k_candidate_with_degree_feature(g1, g2, topk):
    feature_vector1 = get_degree_feature(g1)
    feature_vector2 = get_degree_feature(g2)
    feature_vector1 = normalize(feature_vector1, norm="max",axis=0)
    feature_vector2 = normalize(feature_vector2, norm="max", axis=0)
    return find_top_k_candidate_with_embedding(feature_vector1, feature_vector2,topk)


def find_top_k_candidate_with_embedding(embedding1, embedding2, topk):
    tree = KDTree(embedding1)
    candidate_list2 = tree.query(embedding2, k=topk, return_distance=False)

    tree = KDTree(embedding2)
    candidate_list1 = tree.query(embedding1, k=topk, return_distance=False)

    return candidate_list1.tolist(), candidate_list2.tolist()


def combine_graph(g1, g2, candidate_list1, candidate_list2):
    new_g = nx.Graph()
    node_num1 = nx.number_of_nodes(g1)
    node_num2 = nx.number_of_nodes(g2)
    node_num = node_num1 + node_num2
    edges1 = list(g1.edges())
    edges2 = list(g2.edges())
    edges2 = [(pair[0]+node_num1, pair[1]+node_num1) for pair in edges2]
    edges = edges1 + edges2
    # 添加原本的边
    new_g.add_edges_from(edges)

    # 添加candidate links
    for node, candidate_group in enumerate(candidate_list1):
        candidate_edge_group = [(node, i+node_num1) for i in candidate_group]
        new_g.add_edges_from(candidate_edge_group)
    # for node, candidate_group in enumerate(candidate_list2):
    #     candidate_edge_group = [(node+node_num1, i) for i in candidate_group]
    #     new_g.add_edges_from(candidate_edge_group)

    return new_g


def split_embedding(embeddings, node1):
    e1 = embeddings[:node1,:]
    e2 = embeddings[node1:,:]
    return e1, e2


def get_prediction_alignment(e1, e2):
    cosine_sim = cosine_similarity(e1, e2)
    preds = np.argmax(cosine_sim, axis=1).tolist()
    nodes1 = [i for i in range(e1.shape[0])]
    preds_dict = dict(zip(nodes1, preds))
    return preds_dict


def evaluate(sim, ans_dict):
    # preds_dict = get_prediction_alignment(e1,e2)

    preds_dict = greedy_match(sim)
    acc = 0.0
    cnt = 0.0
    for key in ans_dict.keys():
        cnt += 1.0
        if preds_dict[key] == ans_dict[key]:
            acc += 1.0
    return acc/cnt

def greedy_match(X):
    G1_nodes = [i for i in range(X.shape[0])]
    G2_nodes = [i for i in range(X.shape[1])]
    m, n = X.shape
    x = np.array(X.flatten()).reshape(-1, )
    minSize = min(m, n)
    usedRows = np.zeros(n)
    usedCols = np.zeros(m)
    maxList = np.zeros(minSize)
    row = np.zeros(minSize)
    col = np.zeros(minSize)
    ix = np.argsort(-np.array(x))
    matched = 0
    index = 0
    while (matched < minSize):
        ipos = ix[index]
        jc = int(np.floor(ipos / n))
        ic = int(ipos - jc * n)
        if (usedRows[ic] != 1 and usedCols[jc] != 1):
            row[matched] = G1_nodes[ic]
            col[matched] = G2_nodes[jc]
            maxList[matched] = x[index]
            usedRows[ic] = 1
            usedCols[jc] = 1
            matched += 1
        index += 1
    row = row.astype(int)
    col = col.astype(int)
    return dict(zip(col, row))



