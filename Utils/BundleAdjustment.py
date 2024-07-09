import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from Utils.BuildVisibilityMatrix import BuildVisibilityMatrix

def Get2DIndices(X_ids, x_feat, y_feat, VMatrix):
    pts_2d, pts_ids, cam_ids = [], [], []
    
    vis_x_feats = x_feat[X_ids]    
    vis_y_feats = y_feat[X_ids]

    for i in range(VMatrix.shape[0]):
        for j in range(VMatrix.shape[0]):
            if VMatrix[i,j] == 1:
                pts_2d.append(np.concatenate((vis_x_feats[i,j], vis_y_feats[i,j]), axis=1))
                pts_ids.append(i) # Row are equivalent to matches
                cam_ids.append(j) # Columns are equivelent to image id

    return np.array(pts_2d).reshape(-1,2), np.array(pts_ids).reshape(-1), np.array(cam_ids).reshape(-1)

def BundleAdjustmentSparsity(cam_id, Inliers3D_all_img_ids, inlier_ids, x_feat, y_feat):
    curr_cam = cam_id+1
    X_ids, VMatrix = BuildVisibilityMatrix(Inliers3D_all_img_ids, inlier_ids, cam_id)

    NumObserviations = np.sum(VMatrix)
    NumPoints = len(X_ids[0])
    m = NumObserviations*2
    n = curr_cam*6 + NumPoints*3

    A = lil_matrix((m,n), dtype=int)

    _, pts_ids, cam_ids = Get2DIndices(X_ids, x_feat, y_feat, VMatrix)
    s = np.arange(NumObserviations)

    for i in range(9):
        A[2*s, cam_ids*6 + i] = 1
        A[2*s+1, cam_ids*6 + i] = 1

    for i in range(3):
        A[2*s, cam_id*6 +pts_ids*3+i] = 1
        A[2*s+1, cam_id*6+pts_ids*3+i] = 1

    return A

def CostFunction():
    pass


def BundleAdjustment(x_feat, y_feat, inliers_ids, Inliers3D_all_img, 
                     Inliers3D_all_img_ids, Rotations, Translations,
                     K, cam_id):
    
    X_ids, VMatrix = BuildVisibilityMatrix(Inliers3D_all_img_ids, inliers_ids, cam_id)
    points_3d = Inliers3D_all_img[X_ids]
    points_2d, pts_ids, cam_ids = Get2DIndices(X_ids, x_feat, y_feat, VMatrix)

    Poses = []

    for i in range(cam_id+1):
        R, C = Rotations[i], Translations[i]
        Q = Rotation.from_matrix(R).as_rotvec
        RC = [Q[0], Q[1], Q[2], C[0], C[1], C[2]]
        Poses.append(RC)

    Poses = np.array(Poses).reshape(-1, 6)

    x0 = np.concatenate((Poses.ravel(), points_3d.ravel()), axis=1)
    NumPoint = points_3d.shape[0]

    A = BundleAdjustmentSparsity(cam_id, Inliers3D_all_img_ids, inliers_ids, x_feat, y_feat)

    result = least_squares()