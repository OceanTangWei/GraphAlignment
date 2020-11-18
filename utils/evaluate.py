
import numpy as np
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
