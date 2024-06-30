import argparse
import cv2
import numpy as np
from tqdm import tqdm

from Utils.ParseData import ReadCalbMatrix, ParseMatches, LoadImages, LoadData, FindCommonPoints
from Utils.GetInliersRANSAC import GetInliersRANSAC
from Utils.EstimateFundamentalMatrix import ComputeFundamentalMatrix
from Utils.EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from Utils.ExtractCameraPose import ExtractCameraPose
from Utils.DisambiguateCameraPose import DisambiguateCamPoseAndTriangulate
from Utils.NonlinearTriangulation import NonlinearTriangulation
from Utils.Utils import PlotInliers, PlotNonTriangulation
from Utils.PnPRANSAC import PnPRANSAC

def main():
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataPath',  
                        default='/home/ychen921/733/project3/Data', help='Default:/home/ychen921/733/project3/Data')
    Parser.add_argument('--CalibPath',  
                        default='/home/ychen921/733/project3/Data/calibration.txt', help='Default:/home/ychen921/733/project3/Data/calibration.txt')
    
    Args = Parser.parse_args()
    DataPath = Args.DataPath
    CalibPath = Args.CalibPath
    NumImages = 6

    # Read Calibration Matrix K, correspondences and images
    K = ReadCalbMatrix(CalibPath)
    Matches = ParseMatches(DataPath, NumImages=NumImages)
    x_coords, y_coords, feature_indices = LoadData(DataPath, NumImages=NumImages)
    Images = LoadImages(DataPath, NumImages=NumImages)

    ####### Rejecting outliers for ######
    # ##### all possible image pairs ####
    # ###################################
    print('Computing Fundamental Matrix...')
    inlier_ids = np.zeros_like(feature_indices)
    f_matrix, final_inlier_ids = None, None
    for i in tqdm(range(NumImages-1)):
        for j in range(i+1, NumImages):
            common_ids = np.where(feature_indices[:,i] & feature_indices[:,j])
            src_coords = np.concatenate((x_coords[common_ids, i].reshape((-1,1)), 
                                        y_coords[common_ids, i].reshape((-1,1))), axis=1)
            targ_coords = np.concatenate((x_coords[common_ids, j].reshape((-1,1)), 
                                        y_coords[common_ids, j].reshape((-1,1))), axis=1)
            all_matches = common_ids
            common_ids = np.array(common_ids).reshape(-1)
            
            if len(common_ids) > 8:
                Matches = np.concatenate((src_coords, targ_coords), axis=1)
                f_matrix, inliers = GetInliersRANSAC(Matches)
                final_inlier_ids = all_matches[0][inliers]
                inlier_ids[final_inlier_ids, i] = 1
                inlier_ids[final_inlier_ids, j] = 1
                print(f'{len(all_matches[0])} matches and {len(final_inlier_ids)} inliers between image {i} and image {j}')

    # #################################
    # ##### Only for image 1 and 2 ####
    # #################################

    img1, img2 = Images[0:2]
    
    # Compute Fundamental matrix by RANSAC
    common_ids = np.where(inlier_ids[:,0] & inlier_ids[:,1])
    src_coords = np.concatenate((x_coords[common_ids, 0].reshape((-1,1)), 
                                y_coords[common_ids, 0].reshape((-1,1))), axis=1)
    targ_coords = np.concatenate((x_coords[common_ids, 1].reshape((-1,1)), 
                                y_coords[common_ids, 1].reshape((-1,1))), axis=1)
    Matches = np.concatenate((src_coords, targ_coords), axis=1)
    F = ComputeFundamentalMatrix(Matches)
    PlotInliers(img1, img2, Matches)

    # Compute Essentail matrix from Fundamental matrix
    print('\nComputing Essentail Matrix...')
    E = EssentialMatrixFromFundamentalMatrix(K, F)

    # Extract camera poses(C_set & R_set) from essential matrix
    print('\nExtract camera poses from essentail matrix...')
    CameraPoses = ExtractCameraPose(E)

    # Find unique camera pose using linear triangulation
    print('\nFinding best camera pose...')
    Points3D, R_set, C_set = DisambiguateCamPoseAndTriangulate(Pts1=src_coords, 
                                      Pts2=targ_coords,
                                      CameraPoses=CameraPoses, K=K)
    
    # # Minimize reprojected error by nonlinear triangulation
    print('\nMinimize the reprojection error...')
    Optim_Point3D = NonlinearTriangulation(Pts1=src_coords, 
                                      Pts2=targ_coords,
                                      Pts3D=Points3D, R2=R_set, 
                                      C2=C_set, K=K)

    PlotNonTriangulation(linear_pts=Points3D, non_linear_pts=Optim_Point3D, C=C_set)

    # # Inlier feature points in image 1
    # src_pts = matches[0]['1_2'][IdxInliers, 3:5]
    # for i in range(3, 7):
    #     # Get Inliers in image i and indices
    #     key = '1_' + str(i)
    #     if key not in matches[0]:
    #         continue
    #     print('key:', key)
    #     # Find common 2D and 3D points in the target image
    #     _, src_3D, target = FindCommonPoints(src_pts, target_pts=matches[0][key], Optim_Point3D=Optim_Point3D)
    #     R_new, C_new = PnPRANSAC(Points3D=src_3D, Points2D=target, K=K)

    # print("\n#---------------- Fundamental Matrix ----------------#")
    # print(F)
    # print("#----------------------------------------------------#")
    
    # print("\n#----------------- Number of Inliers ----------------#")
    # print(len(IdxInliers))
    # print("#----------------------------------------------------#")

    # print("\n#----------------- Essentail Matrix ----------------#")
    # print(E)
    # print("#----------------------------------------------------#")
    

if __name__ == '__main__':
    main()
