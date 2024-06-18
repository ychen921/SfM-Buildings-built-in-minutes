import argparse
import cv2
import numpy as np

from Utils.ParseData import ReadCalbMatrix, ParseMatches
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

    # Read Calibration Matrix K and correspondences
    K = ReadCalbMatrix(CalibPath)
    matches = ParseMatches(DataPath, FileNum=1)

    img1 = cv2.imread(DataPath+'/1.jpg')
    img2 = cv2.imread(DataPath+'/2.jpg')

    # Compute Fundamental matrix by RANSAC
    print('Computing Fundamental Matrix...')
    F, IdxInliers = GetInliersRANSAC(matches['1_2'], K)

    PlotInliers(img1, img2, matches['1_2'], IdxInliers)

    # Compute Essentail matrix from Fundamental matrix
    print('\nComputing Essentail Matrix...')
    E = EssentialMatrixFromFundamentalMatrix(K, F)

    CameraPoses = ExtractCameraPose(E)

    print('\nFinding best camera pose...')
    Points3D, linear_R, linear_C = DisambiguateCamPoseAndTriangulate(Pts1=matches['1_2'][IdxInliers,3:5], 
                                      Pts2=matches['1_2'][IdxInliers,5:7],
                                      CameraPoses=CameraPoses, K=K)
    
    print('\nMinimize the reprojection error...')
    Optim_Point3D = NonlinearTriangulation(Pts1=matches['1_2'][IdxInliers,3:5], 
                                      Pts2=matches['1_2'][IdxInliers,5:7],
                                      Pts3D=Points3D, R2=linear_R, 
                                      C2=linear_C, K=K)


    PlotNonTriangulation(linear_pts=Points3D, non_linear_pts=Optim_Point3D, C=linear_C)

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
