import numpy as np

def EssentialMatrixFromFundamentalMatrix(K, F):
    E_hat = K.T @ F @ K
    U, s, Vt = np.linalg.svd(E_hat)

    s_hat = np.diag([1, 1, 0])
    EssentialMatrix = U @ s_hat @ Vt
    return EssentialMatrix