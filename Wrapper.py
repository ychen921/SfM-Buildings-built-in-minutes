import argparse
import cv2
import numpy as np

from Utils.ParseData import ReadCalbMatrix, ParseMatches
from Utils.GetInliersRANSAC import GetInliersRANSAC

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

    image_1 = cv2.imread(DataPath+'/1.jpg')
    image_2 = cv2.imread(DataPath+'/2.jpg')

    # Compute Fundamental matrix by RANSAC
    GetInliersRANSAC(matches['1_2'], K)


if __name__ == '__main__':
    main()
