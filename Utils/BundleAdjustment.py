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
        for j in range(VMatrix.shape[1]):
            if VMatrix[i,j] == 1:
                pts_2d.append(np.hstack((vis_x_feats[i,j], vis_y_feats[i,j])))
                pts_ids.append(i) # Row are equivalent to matches
                cam_ids.append(j) # Columns are equivelent to image id

    return np.array(pts_2d).reshape(-1,2), np.array(pts_ids).reshape(-1), np.array(cam_ids).reshape(-1)

def BundleAdjustmentSparsity(cam_id, Inliers3D_all_img_ids, inlier_ids, x_feat, y_feat):
    '''
    Refer to scipy sparse bundle adjustment
    (https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html)
    '''
    
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

def CostFunction(x0, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    
    '''
    Refer to scipy sparse bundle adjustment
    (https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html)
    '''
    total_cam = n_cameras+1
    camera_params = x0[:total_cam * 6].reshape((total_cam, 6))

    points_3d = x0[total_cam * 6:].reshape((n_points, 3))

    x_proj_list = []
    points_3d = points_3d[point_indices]
    camera_params = camera_params[camera_indices]

    for i in range(len(camera_params)):
        R_ = Rotation.from_rotvec(camera_params[i, :3])
        R = R_.as_matrix()
        C = camera_params[i, 3:].reshape(3,1)
        point_3d = points_3d[i]
        P = np.dot(K, np.dot(R, np.concatenate((np.identity(3), -C), axis=1)))
        x_4 = np.hstack((point_3d, 1))
        x_projected = np.dot(P, x_4.T)
        x_projected = x_projected/x_projected[-1]
        pt_proj = x_projected[:2]
        x_proj_list.append(pt_proj)
    x_projections = np.array(x_proj_list)
    error_vec = (x_projections - points_2d).ravel()

    return error_vec        

def BundleAdjustment(x_feat, y_feat, inliers_ids, Inliers3D_all_img, Inliers3D_all_img_ids, 
                     Rotations, Translations, K, cam_id):
    
    X_ids, VMatrix = BuildVisibilityMatrix(Inliers3D_all_img_ids, inliers_ids, cam_id)
    points_3d = Inliers3D_all_img[X_ids]
    points_2d, pts_ids, cam_ids = Get2DIndices(X_ids, x_feat, y_feat, VMatrix)

    Poses = []
    for i in range(cam_id+1):
        R, C = Rotations[i], Translations[i]
        if R is 0 and C is 0:
            continue
        Q_ = Rotation.from_matrix(R)
        Q = Q_.as_rotvec()
        RC = [Q[0], Q[1], Q[2], C[0][0], C[1][0], C[2][0]]
        Poses.append(RC)
    Poses = np.array(Poses).reshape(-1, 6)

    x0 = np.concatenate((Poses.ravel(), points_3d.ravel()), axis=0)
    NumPoint = points_3d.shape[0]

    A = BundleAdjustmentSparsity(cam_id, Inliers3D_all_img_ids, inliers_ids, x_feat, y_feat)

    result = least_squares(CostFunction, x0, jac_sparsity=A, verbose=2, x_scale='jac',
                           ftol=1e-4, method='trf',
                           args=(cam_id, NumPoint, cam_ids, pts_ids, points_2d, K))
    
    x_1 = result.x
    total_cameras = cam_id+1

    optimal_camera_params = x_1[:total_cameras*6].reshape((total_cameras, 6))
    optimal_points_3d = x_1[total_cameras*6:].reshape((NumPoint, 3))

    optim_points_3d_all = np.zeros_like(Inliers3D_all_img)   
    optim_points_3d_all[X_ids] = optimal_points_3d

    optim_R_set, optim_C_set = [], []
    for i in range(len(optimal_camera_params)):
        R_ = Rotation.from_rotvec(optimal_camera_params[i, :3])
        R = R_.as_matrix()
        C = optimal_camera_params[i, 3:].reshape(3,1)
        optim_R_set.append(R)
        optim_C_set.append(C)

    return optim_R_set, optim_C_set, optim_points_3d_all