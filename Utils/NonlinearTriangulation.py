import numpy as np
from scipy.optimize import least_squares


def NonlinearTriangulation(Pts1, Pts2, Pts3D, R2, C2, K):
    """
    Minimize the reprojected error
    """
    I = np.eye(3)
    R1 = np.eye(3)
    C1 = np.zeros((3,1))
    Pts3D = np.concatenate((Pts3D, np.ones(Pts3D.shape[0]).reshape(-1,1)), axis=1) # Add ones to the last column
    
    # Recontruct Camera poses
    P1 = K @ R1 @ np.concatenate((I, -C1.reshape(3,1)), axis=1)
    P2 = K @ R2 @ np.concatenate((I, -C2.reshape(3,1)), axis=1)

    Optimal_Point3D = []
    MSE = 0

    for i, X in enumerate(Pts3D):
        result = least_squares(fun=CostFunction, x0=X, method='trf',
                               args=[Pts1[i], Pts2[i], P1, P2])
        # CostFunction(X, Pts1[i], Pts2[i], P1, P2)
        Optimal_Point3D.append(result.x)

    return np.array(Optimal_Point3D)/ np.array(Optimal_Point3D)[:,3].reshape(-1,1)

def CostFunction(X, pts1, pts2, P1, P2):
    """
    The loss function that minimized by least square error
    """

    P1_row1, P1_row2, P1_row3 = P1[0,:], P1[1,:], P1[2,:]
    P2_row1, P2_row2, P2_row3 = P2[0,:], P2[1,:], P2[2,:]

    u_1, v_1 = pts1[0], pts1[1]
    u_2, v_2 = pts2[0], pts2[1]

    error1 = np.square(u_1 - ((P1_row1 @ X) / (P1_row3 @ X))) +\
          np.square(v_1 - ((P1_row2 @ X) / (P1_row3 @ X)))
    
    error2 = np.square(u_2 - ((P2_row1 @ X) / (P2_row3 @ X))) +\
          np.square(v_2 - ((P2_row2 @ X) / (P2_row3 @ X)))
    
    SumError = np.squeeze(error1 + error2)
    
    return SumError