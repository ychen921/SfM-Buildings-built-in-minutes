import argparse
import cv2
import numpy as np

from Utils.ParseData import ReadCalbMatrix, ParseMatches, LoadImages, FindCommonPoints
from Utils.GetInliersRANSAC import GetInliersRANSAC
from Utils.EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from Utils.ExtractCameraPose import ExtractCameraPose
from Utils.DisambiguateCameraPose import DisambiguateCamPoseAndTriangulate
from Utils.NonlinearTriangulation import NonlinearTriangulation
from Utils.Utils import PlotInliers, PlotNonTriangulation

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

    # Read Calibration Matrix K, correspondences and images
    K = ReadCalbMatrix(CalibPath)
    matches = ParseMatches(DataPath, NumImages=6)
    Images = LoadImages(DataPath, NumImages=6)

    #################################
    ##### Only for image 1 and 2 ####
    #################################

    img1, img2 = Images[0:2]
    
    # Compute Fundamental matrix by RANSAC
    print('Computing Fundamental Matrix...')
    F, IdxInliers = GetInliersRANSAC(matches[0]['1_2'], K)

    PlotInliers(img1, img2, matches[0]['1_2'], IdxInliers)

    # Compute Essentail matrix from Fundamental matrix
    print('\nComputing Essentail Matrix...')
    E = EssentialMatrixFromFundamentalMatrix(K, F)

    # Extract camera poses(C_set & R_set) from essential matrix
    print('\nExtract camera poses from essentail matrix...')
    CameraPoses = ExtractCameraPose(E)

    # Find unique camera pose using linear triangulation
    print('\nFinding best camera pose...')
    Points3D, R_set, C_set = DisambiguateCamPoseAndTriangulate(Pts1=matches[0]['1_2'][IdxInliers,3:5], 
                                      Pts2=matches[0]['1_2'][IdxInliers,5:7],
                                      CameraPoses=CameraPoses, K=K)
    
    # Minimize reprojected error by nonlinear triangulation
    print('\nMinimize the reprojection error...')
    Optim_Point3D = NonlinearTriangulation(Pts1=matches[0]['1_2'][IdxInliers,3:5], 
                                      Pts2=matches[0]['1_2'][IdxInliers,5:7],
                                      Pts3D=Points3D, R2=R_set, 
                                      C2=C_set, K=K)

    PlotNonTriangulation(linear_pts=Points3D, non_linear_pts=Optim_Point3D, C=C_set)

    # Inlier feature points in image 1
    src_pts = matches[0]['1_2'][IdxInliers, 3:5]
    for i in range(3, 7):
        # Get Inliers in image i and indices
        key = '1_' + str(i)
        if key not in matches[0]:
            continue
        
        # Find common 2D and 3D points in the target image
        src, src_3D, target = FindCommonPoints(src_pts, target_pts=matches[0][key], Optim_Point3D=Optim_Point3D)
        

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
