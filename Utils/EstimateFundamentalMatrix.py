import numpy as np

def ComputeFundamentalMatrix(RandMatches):
    A = []
    for i in range(RandMatches.shape[0]):
        x1, y1, x2, y2 = RandMatches[i,:]#[i, 3:]
        A.append([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])
    A = np.array(A)
    
    U, s, Vt = np.linalg.svd(A)
    V = Vt.T
    F = V[:,-1].reshape(3,3)

    Uf, sf, Vft = np.linalg.svd(F)

    # rank 2 constraint
    sf[-1] = 0
    F_final = Uf @ np.diag(sf) @ Vft
    F_final /= F_final[2,2]
    return F_final