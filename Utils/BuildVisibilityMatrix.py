import numpy as np

def BuildVisibilityMatrix(x_ids, X_all, cam_id):
    temp = np.zeros((x_ids.shape[0]), dtype=int)

    for i in range(cam_id+1):
        temp = temp | x_ids[:,i]
    inliers_3D = X_all.reshape(-1)

    X_ids = np.where(inliers_3D & temp)

    VMatrix = X_all[X_ids].reshape(-1,1)

    for i in range(cam_id+1):

        VMatrix = np.concatenate((VMatrix, x_ids[X_ids, i].reshape(-1,1)), axis=1)

    VMatrix = VMatrix[:, 1:VMatrix.shape[1]]

    return X_ids, VMatrix