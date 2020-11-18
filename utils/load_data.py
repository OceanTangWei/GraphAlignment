
import pickle as pkl
import scipy.sparse as sp
import sys
import copy

import networkx as nx
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import random

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):  # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("D:/pythonProject/GraphAlignmentLibrary/data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("D:/pythonProject/GraphAlignmentLibrary/data/ind.{}.test.index".format(dataset_str))
    # test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]


    return adj, features


def load_random_data(size):
    adj = sp.random(size, size, density=0.002)  # density similar to cora
    features = sp.random(size, 1000, density=0.015)
    int_labels = np.random.randint(7, size=(size))
    labels = np.zeros((size, 7))  # Nx7
    labels[np.arange(size), int_labels] = 1

    train_mask = np.zeros((size,)).astype(bool)
    train_mask[np.arange(size)[0:int(size / 2)]] = 1

    val_mask = np.zeros((size,)).astype(bool)
    val_mask[np.arange(size)[int(size / 2):]] = 1

    test_mask = np.zeros((size,)).astype(bool)
    test_mask[np.arange(size)[int(size / 2):]] = 1

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # sparse NxN, sparse NxF, norm NxC, ..., norm Nx1, ...
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def preprocess_features2(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def preprocess_adj_bias(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack(
        (adj.col, adj.row)).transpose()  # This is where I made a mistake, I used (adj.row, adj.col) instead
    # return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
    return indices, adj.data, adj.shape


def preprocess_adj_bias2(adj):
    adj = sp.csr_matrix(adj)
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack(
        (adj.col, adj.row)).transpose()  # This is where I made a mistake, I used (adj.row, adj.col) instead
    # return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
    return indices, adj.data, adj.shape




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


def shuffle_graph(g,features=None,shuffle=True):

    original_nodes = list(g.nodes())
    new_nodes = copy.deepcopy(original_nodes)
    if shuffle:
        random.shuffle(new_nodes)
    original_to_new = dict(zip(original_nodes, new_nodes))
    new_graph = nx.Graph()
    for edge in g.edges():
        new_graph.add_edge(original_to_new[edge[0]], original_to_new[edge[1]])

    if features is not None:
        new_to_original = {original_to_new[i]:i for i in range(nx.number_of_nodes(g))}
        new_order = [new_to_original[i] for i in range(nx.number_of_nodes(g))]
        features = features[new_order,:]
        return new_graph, original_to_new, features
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





