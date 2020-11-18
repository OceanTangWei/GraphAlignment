from utils.load_data import load_data,create_align_graph,shuffle_graph, cosine_similarity
import networkx as nx
from Methods.FINAL.final import FINAL
from Methods.DCMF.src import DCMF
from utils.evaluate import evaluate
from Methods.CENALP.CENALP import CENALP
from Methods.GOT.src import got_align
import numpy as np
if __name__ == '__main__':
    adj,feature = load_data("cora")

    feature = feature[:,:50]
    feature = feature.todense()
    feature = np.ones_like(feature)

    g1 = nx.from_scipy_sparse_matrix(adj)
    g2 = create_align_graph(g1,remove_rate=0.00,add_rate=0.0)


    g2,ans_dict,feature2 = shuffle_graph(g2,feature,True)
    attribute = cosine_similarity(feature, feature2)
    # S, precision, seed_l1, seed_l2 = CENALP(g1, g2,0.2, attribute, {},
    #                                         {},
    #                                         3, 0.0, 5, 0.5, True)
    model = FINAL(g1,g2,feature,feature2,alpha=0.82,maxiter=30)
    sim = model.align()
    acc = evaluate(sim, ans_dict)
    print(acc)