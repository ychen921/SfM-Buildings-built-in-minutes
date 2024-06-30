import numpy as np

def LinearPnP(X, x, K):
    x = np.concatenate((x, np.ones(x.shape[0]).reshape(-1,1)), axis=1)
    x_norm = np.dot(np.linalg.inv(K), x.T).T # Normalized by K-^1 x

    A = []
    for i in range(X.shape[0]):
        z = np.zeros((1,4))
        X_ = X[i,:].reshape((1,4))
        u, v, _ = x_norm[i,:]

        row_1 = np.concatenate((X_, z, z), axis=1)
        row_2 = np.concatenate((z, X_, z), axis=1)
        row_3 = np.concatenate((z, z, X_), axis=1)

        homogeneouos = np.concatenate((row_1, row_2, row_3), axis=0)
        uv = np.array([[0, -1, v],
                       [1, 0, -u],
                       [-v, u, 0]])
        a = np.dot(uv, homogeneouos)
        if i == 0:
            A = a
        else:
            A = np.vstack((A, a))

        _, _, Vt = np.linalg.svd(A)
        # V = Vt.T
        P = Vt[:,-1].reshape((3,4)) # projection matrix
        R, t = P[:, :3], P[:, 3] # R and t matrix from projection matrix P
        
        # Corrected Rotation matrix
        U, S, Vtr = np.linalg.svd(R)
        R = np.dot(U, Vtr)

        if np.linalg.det(R) < 0:
            R = -R

        C = -np.dot(R.T, t)

        return R, C
    
def ProjectionError(R, C, x, X, K, threshold):
    I = np.eye(3)
    P = K @ R @ np.concatenate((I, -C.reshape(3,1)), axis=1)
    Pr1 = P[0,:]
    Pr2 = P[1,:]
    Pr3 = P[2,:]

    S = []
    for i in range(X.shape[0]):
        u, v = x[i,:]
        X_ = X[i,:]

        u_reproj = np.dot(Pr1, X_) / np.dot(Pr3, X_)
        v_reproj = np.dot(Pr2, X_) / np.dot(Pr3, X_)

        error = np.sqrt(np.square(u - u_reproj) + np.square(v - v_reproj))
        if error < threshold:
            S.append(i)
    
    return len(S)
        

def PnPRANSAC(Points3D, Points2D, K):
    max_inlier_count = 0
    MaxIteration = 5000
    iter_count = 0
    NumSample = 6 

    NumPts = Points3D.shape[0] # Number of sample points
    R_final, C_final = None, None

    while MaxIteration > iter_count:
        if NumPts != 0:
            idx = np.random.randint(0, NumPts, NumSample)
        else:
            print('\nPnP RANSAC no matching points!')
            return None, None

        Rand_3D = Points3D[idx, :]
        Rand_2D = Points2D[idx, :]

        R, C = LinearPnP(Rand_3D, Rand_2D, K)
        NumInliers = ProjectionError(R, C, Points2D, Points3D, K, threshold=5)
        
        if NumInliers > max_inlier_count:
            max_inlier_count = NumInliers
            R_final = R
            C_final = C

        iter_count += 1
    print(max_inlier_count)
    print(R_final, C_final)
    return R_final, C_final