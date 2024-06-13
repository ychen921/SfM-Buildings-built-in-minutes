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

def ParseMatches(path, FileNum):
    data_name = 'matching'+str(FileNum)+'.txt'
    file_path = os.path.join(path, data_name)

    with open(file_path, 'r') as matches_file:
        correspondences = matches_file.readlines()
    
    correspondences = [x.split() for x in correspondences[1:]]
    
    matches = {}
    for row in correspondences:
        correspondence = row[1:6] # RGB values and u, v of the current feature point
        NumMatch = int(row[0]) # Number of mathcing of the current feature point

        # Classify the matching feature points to respective dictionary
        for i in range(1, NumMatch):
            current_correspondence = correspondence.copy()
            current_correspondence.extend([row[-(i*3)+1], row[-(i*3)+2]])
            key = str(FileNum)+'_'+row[-(i*3)]
            
            if key in matches:
                matches[key].append(current_correspondence)
            else:
                matches[key] = [current_correspondence]

    for key in matches:
        matches[key] = np.array(matches[key], dtype=np.float32)
    
    return matches