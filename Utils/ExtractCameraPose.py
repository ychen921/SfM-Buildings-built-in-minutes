import numpy as np

def ExtractCameraPose(E):
    U, _, Vt = np.linalg.svd(E)

    R = []
    C = []

    # Translation Matrix
    C.append(U[:, 2]) # C1
    C.append(-U[:, 2]) # C2
    C.append(U[:, 2]) # C3
    C.append(-U[:, 2]) # C4

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    # Rotation Matrix
    R.append(U @ (W @ Vt)) # R1
    R.append(U @ (W @ Vt)) # R2
    R.append(U @ (W.T @ Vt)) # R3
    R.append(U @ (W.T @ Vt)) # R4

     ### Important:  det(R) = 1. If det(R) = −1, # 
     ### the camera pose must be corrected ####### 
     ### i.e. C = −C and R = −R ##################
    for i in range(len(R)):
        if (np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            C[i] = -C[i]

    # All possible camera poses
    camera_poses = [[R[0], C[0]], [R[1], C[1]], [R[2], C[2]], [R[3], C[3]]]
    print(camera_poses)
    return camera_poses