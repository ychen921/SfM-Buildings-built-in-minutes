import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def NonLinearPnP(x, X, K, R, C):
    if R is not None and C is not None:
        Q_ = Rotation.from_matrix(R)
        Q = Q_.as_quat()

        QC = [Q[0], Q[1], Q[2], Q[3], C[0], C[1], C[2]]

        optim_result = least_squares(fun=CostFunction, method='trf', x0=QC, args=[X, x, K])
        X = optim_result.x
        optim_q = X[:4]
        C = X[4:]
        R_ = Rotation.from_quat(optim_q)
        R = R_.as_matrix()

        return R, C
    
    else:
        return 0, 0
    

def CostFunction(X, pts_3d, pts_2d, K):
    pts_3d = np.concatenate((pts_3d, np.ones(pts_3d.shape[0]).reshape(-1,1)), axis=1)
    Q = X[:4]
    C = X[4:].reshape(-1,1)

    R_ = Rotation.from_quat(Q)
    R = R_.as_matrix()
    P = np.dot(K, np.dot(R, np.concatenate((np.eye(3), -C), axis=1)))

    errors = []

    for pt_2d, pt_3d in zip(pts_2d, pts_3d):
        P1_row1, P1_row2, P1_row3 = P[0,:], P[1,:], P[2,:]
        u, v = pt_2d
        error = np.square(u - ((P1_row1 @ pt_3d) / (P1_row3 @ pt_3d))) + np.square(v - ((P1_row2 @ pt_3d) / (P1_row3 @ pt_3d)))
        errors.append(error)

    errors = np.squeeze(errors)
    mean_error = np.mean(errors)

    return mean_error