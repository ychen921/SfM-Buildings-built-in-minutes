import cv2
import numpy as np
from Utils.EstimateFundamentalMatrix import ComputeFundamentalMatrix


def points_err(matches, F, threshold):
    NumInliers = 0
    IndexInliers = []

    for i, pt_pair in enumerate(matches):
        x1,y1,x2,y2 = pt_pair[3:]
        p1 = np.array([x1,y1,1])
        p2 = np.array([x2,y2,1])

        error = abs(p2.T @ F @ p1)

        if error < threshold:
            NumInliers += 1
            IndexInliers.append(i)

    return NumInliers, IndexInliers

def GetInliersRANSAC(Matches, K, threshold=0.02):
    max_inlier_count = 0
    MaxIteration = 5000
    iter_count = 0
    NumSample = 8 # Pick 8 correspondances for computing F matrix
    NumPts = Matches.shape[0]
    BestInliersIndex = []
    BestFundamentalMatrix = []

    while MaxIteration > iter_count:
        idx = np.random.randint(0, NumPts, NumSample)
        RandCorrespondences = Matches[idx, :]
        # RGB = RandCorrespondences[:, 0:3]
        F = ComputeFundamentalMatrix(RandCorrespondences)
        NumInliers, IndexInliers = points_err(RandCorrespondences, F, threshold)

        if NumInliers > max_inlier_count:
            max_inlier_count = NumInliers
            BestFundamentalMatrix = F
        iter_count += 1