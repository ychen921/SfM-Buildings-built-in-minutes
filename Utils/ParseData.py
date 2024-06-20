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

def LoadImages(path, NumImages):
    Images = []
    for i in range(NumImages):
        file_path = os.path.join(path, str(i+1)+'.jpg')
        image = cv2.imread(file_path)
        Images.append(image)
    return Images

def ParseMatches(path, NumImages=6):
    ListMatches = []
    for FileNum in range(1, NumImages):
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
            _, indices = np.unique(matches[key][:,3:5], axis=0, return_index=True)
            matches[key] = matches[key][indices]
            
        ListMatches.append(matches)
    
    return ListMatches

def FindCommonPoints(src_pts, target_pts, Optim_Point3D):
    target_xy = target_pts[:, 3:5]
    target_bool = np.any(np.all(target_xy[:, None] == src_pts, axis=2), axis=1)
    target_indices = np.where(target_bool)[0]
    target = target_pts[target_indices, 5:7]

    src_bool = np.any(np.all(src_pts[:, None] == target_pts[target_indices, 3:5], axis=2), axis=1)
    src_indices = np.where(src_bool)[0]
    src = src_pts[src_indices]
    src_3D = Optim_Point3D[src_indices]
    
    return src, src_3D, target