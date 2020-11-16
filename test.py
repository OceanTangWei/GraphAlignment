import numpy as np
import scipy.sparse as ss
import networkx as nx
import ot
a = np.array([0.5,0.4,0.1]).reshape(-1,)
b = np.array([0.8,0.1,0.1]).reshape(-1,)
M = np.array([
    [1,2,3],
    [2,1,3],
    [1,1,1]
    ]
)
d = ot.emd(a,b,M)
print(d)