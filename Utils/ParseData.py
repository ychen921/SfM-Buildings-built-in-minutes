import os
import cv2
import numpy as np

def ReadCalbMatrix(file):

    with open(file, 'r') as f:
        lines = f.readlines()

    matrix_lines = [line.strip().replace('K = [', '').replace(';', '').replace(']', '').split() for line in lines]
    matrix_values = [float(value) for sublist in matrix_lines for value in sublist]
    
    K = np.array(matrix_values).reshape(3, 3)
    return K