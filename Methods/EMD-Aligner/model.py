from abc import ABC
import networkx as nx
import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from sklearn.neighbors import KDTree
from collections import defaultdict
import numpy as np
import random
import os
from utils.load_data import load_data, preprocess_features
from utils.utils import create_align_graph, shuffle_graph, evaluate
import ot
torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class config(object):
    def __init__(self):
        self.hidden_dim1 = 128
        self.hidden_dim2 = 128
        self.top_candidates = 10
        self.topk = 2

        # training
        self.epoches = 100
        self.lr = 1e-3
        self.batch_size = 5

class GCN(nn.Module):
    def __init__(self,in_feats,out_feats1):
        super().__init__()
        self.gcn1 = GraphConv(in_feats=in_feats,out_feats=out_feats1)
        self.gcn2 = GraphConv(in_feats=out_feats1,out_feats=out_feats1)
        self.a = nn.ReLU(inplace=False)
    def forward(self,g,feats):

        h1 = self.a(self.gcn1(g, feats))
        h2 = self.a(self.gcn2(g, h1))
        return h2

class EMDAligner(nn.Module):
    def __init__(self,in_feats,out_feats1,out_feats2):
        super().__init__()
        self.GCNLayer = GCN(in_feats,out_feats1)
        self.MLPLayer1 = nn.Linear(in_features=out_feats1,out_features=out_feats2)
        self.MLPLayer2 = nn.Linear(in_features=out_feats2, out_features=out_feats2)
        self.a = nn.ReLU(inplace=False)
    def calculate_cosine(self,v1,v2):

        v1_ = F.normalize(v1, p=2, dim=-1)
        v2_ = F.normalize(v2,p=2,dim=-1)
        res = torch.mm(v1_,v2_.T)
        res = torch.clamp(res,min=0.0,max=1.0)

        return res

    def forward(self,g1,g2,feat1,feat2):

        c1 = self.GCNLayer(g1,feat1)
        c2 = self.GCNLayer(g2,feat2)

        x11 = self.a(self.MLPLayer1(c1))
        x12 = self.a(self.MLPLayer1(c2))
        x21 = self.a(self.MLPLayer2(x11))
        x22 = self.a(self.MLPLayer2(x12))

        return c1,c2,x21,x22

class DGA(object):
    def __init__(self,g1, g2, node_feat1, node_feat2,config):
        super().__init__()
        self.g1 = g1
        self.g2 = g2
        self.node_feat1 = node_feat1
        self.node_feat2 = node_feat2
        self.model = EMDAligner(node_feat1.shape[-1], config.hidden_dim1, config.hidden_dim2)
        self.config = config

    def get_sub_graph_nodes(self):
        sub_g1 = defaultdict(list)
        sub_g2 = defaultdict(list)
        for node in self.g1.nodes():
            sub_g1[node] = list(nx.single_source_shortest_path_length(self.g1, node, self.config.topk).keys())
        for node in self.g2.nodes():
            sub_g2[node] = list(nx.single_source_shortest_path_length(self.g2, node, self.config.topk).keys())
        return sub_g1, sub_g2

    @staticmethod
    def batch(_, batch_size,Random=True):
        if Random:
            random.shuffle(_)

        L = len(_)
        if L%batch_size == 0:
            iters = L//batch_size
        else:
            iters = L//batch_size + 1
        for iter in range(iters):
            yield _[iter*batch_size:(iter+1)*batch_size]


    def get_train_data(self):
        g1 = dgl.from_networkx(self.g1)
        g2 = dgl.from_networkx(self.g2)

        feat1 = torch.from_numpy(self.node_feat1).float()
        feat2 = torch.from_numpy(self.node_feat2).float()

        # 对于无监督对齐场景，先使用随机初始化的GCN找到对齐的seed
        h1 = self.model.GCNLayer(g1, feat1)
        h2 = self.model.GCNLayer(g2, feat2)
        h1 = h1.cpu().detach().numpy()
        h2 = h2.cpu().detach().numpy()

        h1 =preprocessing.normalize(h1, norm='l2')
        h2 = preprocessing.normalize(h2,norm='l2')

        rough_similarity = cosine_similarity(h1,h2)*cosine_similarity(self.node_feat1,self.node_feat2)
        # 挑选出前K大
        candidates = np.argpartition(-rough_similarity,kth=self.config.top_candidates,axis=1)
        candidates = candidates[:,:self.config.top_candidates]
        # train_indices
        train_indices = defaultdict(list)
        for i in range(candidates.shape[0]):
            train_indices[i] = candidates[i,:].tolist()
        return train_indices,g1,g2,feat1,feat2

    def get_optimal_distance_matrix(self):
        pass

    def train(self,true_dict):
        train_indices, g1, g2, feat1, feat2 = self.get_train_data()
        print(g1)
        adj1 = nx.to_numpy_array(self.g1,nodelist=range(nx.number_of_nodes(self.g1)))
        adj2 = nx.to_numpy_array(self.g2, nodelist=range(nx.number_of_nodes(self.g2)))
        adj1 = torch.from_numpy(adj1).float().cuda()
        adj2 = torch.from_numpy(adj2).float().cuda()
        self.model.cuda()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        feat1_c = feat1.cuda()
        feat2_c = feat2.cuda()
        sub_g1, sub_g2 = self.get_sub_graph_nodes()
        g1 = g1.to('cuda:0')
        g2 = g2.to('cuda:0')


        # 两个阶段
        # 固定GCN,即固定成本函数C，利用POT，求出运输方案，距离矩阵X
        for epoch in range(self.config.epoches):

            c1,c2,x1,x2 = self.model(g1,g2,feat1_c,feat2_c)

            # Phase1
            C = 1.0-self.model.calculate_cosine(c1,c2)
            C_numpy=C.detach().cpu().numpy().astype(np.float64)


            c2_mean = torch.mean(c2,dim=0,keepdim=True)
            c1_mean = torch.mean(c1,dim=0,keepdim=True)
            s1 = torch.clamp(torch.mm(c1, c2_mean.T),min=0) #(N1,1)
            s2 = torch.clamp(torch.mm(c2, c1_mean.T), min=0) #(N2,1)

            s1_hat = s1 / torch.sum(s1)
            s2_hat = s2 / torch.sum(s2)

            s1_hat = s1_hat.reshape(-1,).detach().cpu().numpy().astype(np.float64)
            s2_hat = s2_hat.reshape(-1,).detach().cpu().numpy().astype(np.float64)

            # this is a bug of POT.....
            s1_hat = s1_hat / s1_hat.sum()
            s2_hat = s2_hat / s2_hat.sum()

            X = ot.emd(s1_hat,s2_hat,C_numpy)

            # X = torch.from_numpy(X).float().cuda()
            # acc = evaluate((1 - C).detach().cpu().numpy(), true_dict)
            # print(acc)
            # Phase2
             # 转置矩阵P
            for i in range(10):
                c1, c2, x1, x2 = self.model(g1, g2, feat1_c, feat2_c)


                C = 1.0 - self.model.calculate_cosine(c1, c2)
                # P = torch.mul(1-C,X)+1e-12

                P = torch.clamp(1-C, min=1e-24)
                P = P/P.sum(dim=1,keepdim=True)

                P = doubly_stochastic(P,2,10)

                loss = get_loss(adj1,adj2,P)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            acc = evaluate(P.detach().cpu().numpy(), true_dict)
            print(acc)
            acc = evaluate((1-C).detach().cpu().numpy(), true_dict)
            print(acc)
            print("########3")







            #
            # for batch_list in DGA.batch(list(train_indices.keys()), self.config.batch_size):
            #     loss = torch.tensor(0).cuda()
            #     self.model.train()
            #     for key in batch_list:
            #         key_neighbors = sub_g1[key]
            #         c_source = c1[key_neighbors, :]
            #         x_source = x1[key_neighbors, :]
            #         for neighbor in train_indices[key]:
            #             n_neighbors = sub_g2[neighbor]
            #             c_target = c2[n_neighbors, :]
            #             x_target = x2[n_neighbors, :]
            #
            #             C = 1 - self.model.calculate_cosine(c_source, c_target)
            #             X = self.model.calculate_cosine(x_source, x_target)
            #
            #             loss = loss + torch.sum(torch.mul(C, X)) + \
            #                     torch.abs(torch.sum(X, dim=1) - 1).sum() + \
            #                     torch.abs(torch.sum(X, dim=0) - 1).sum()
            #
            #     loss = loss/(len(train_indices.keys())*len(batch_list))
            #
            #     optimizer.zero_grad()
            #     loss.backward(retain_graph=True)
            #     optimizer.step()
            #
            #     loss_list.append(loss.detach().cpu().numpy())
            #
            #     print("epoch:{}---loss:{}".format(epoch,sum(loss_list)/len(loss_list)))


def doubly_stochastic(P, tau, it):
    """Uses logsumexp for numerical stability."""

    A = P / tau
    for i in range(it):
        A = A - A.logsumexp(dim=1, keepdim=True)
        A = A - A.logsumexp(dim=0, keepdim=True)
    return torch.exp(A)
def get_loss(adj1, adj2, P):
    return -torch.trace(P.T@adj1@P@adj2.T)
def main():
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data("cora")
    features,_ = preprocess_features(features)
    features = features[:, :100]
    g1 = nx.from_scipy_sparse_matrix(adj)
    g2 = create_align_graph(g1,remove_rate=0.2,add_rate=0)
    g2, original_to_new = shuffle_graph(g2)
    C = config()
    features1 = features
    new2original = {original_to_new[i]: i for i in range(nx.number_of_nodes(g2))}
    cols = [new2original[i] for i in range(nx.number_of_nodes(g2))]
    features2 = features[cols]
    model = DGA(g1,g2,features1,features2,C)
    model.train(original_to_new)








if __name__ == '__main__':
    main()







