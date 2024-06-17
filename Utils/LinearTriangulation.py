import numpy as np
    
def LinearTriangulation(Pts1, Pts2, C1, R1, C2, R2, K):
    Points3D = []
    I = np.eye(3)

    # Reconstruct camera matrices
    P1 = K @ R1 @ np.concatenate((I, -C1.reshape(3,1)), axis=1)
    P2 = K @ R2 @ np.concatenate((I, -C2.reshape(3,1)), axis=1)

    # Implement Direct Linear Transform (DLT)
    p1_row1 = P1[0,:]
    p1_row2 = P1[1,:]
    p1_row3 = P1[2,:]

    p2_row1 = P2[0,:]
    p2_row2 = P2[1,:]
    p2_row3 = P2[2,:]

    for i in range(Pts1.shape[0]):
        x1, y1 = Pts1[i][0], Pts1[i][1]
        x2, y2 = Pts2[i][0], Pts2[i][1]

        ''' 
        Compute the 3D coordinates X from 2 correspondences
        x1 and x2. We can solve it by direct linear transform,
        x cross product P X = 0. The 3D coordinate X can be solved
        AX by SVD.  
        '''
        A = [y1*p1_row3-p1_row2, p1_row1-x1*p1_row3, 
             y2*p2_row3-p2_row2, p2_row1-x2*p2_row3]
        A = np.array(A)
        
        # Solve AX = 0 by SVD
        _, _, Vt = np.linalg.svd(A)
        V = Vt.T
        X = V[:,-1]
        Points3D.append(X)

    return np.array(Points3D)