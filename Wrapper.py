import argparse
import numpy as np
import cv2
import os

from Utils.ParseData import ReadCalbMatrix

def main():
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataPath', dest='DataPath', 
                        default='/home/ychen921/733/project3/Data', help='Default:/home/ychen921/733/project3/Data')
    Parser.add_argument('--CalibPath', dest='CalibPath', 
                        default='/home/ychen921/733/project3/Data/calibration.txt', help='Default:/home/ychen921/733/project3/Data/calibration.txt')
    
    Args = Parser.parse_args()
    DataPath = Args.DataPath
    CalibPath = Args.CalibPath

    # Read Calibration Matrix K
    K = ReadCalbMatrix(CalibPath)


if __name__ == '__main__':
    main()
