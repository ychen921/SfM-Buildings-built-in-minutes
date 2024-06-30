import numpy as np
from Utils.Utils import PlotInitialTriangulation, PlotTriangulation
from Utils.LinearTriangulation import LinearTriangulation

def DisambiguateCamPoseAndTriangulate(Pts1, Pts2, CameraPoses, K):
    MaxInlierCounts = 0

    C1 = np.zeros(3)
    R1 = np.eye(3)

    AllPoints = []
    for i, pose in enumerate(CameraPoses):
        R2, C2 = pose
        Points3D = LinearTriangulation(Pts1, Pts2, C1, R1, C2, R2, K)
        Points3D = Points3D / Points3D[:,3].reshape(-1,1) # Devided by Z
        
        r3 = R2[-1, :].reshape(1,-1)
        Points3D = Points3D[:,:3]

        AllPoints.append(Points3D)

        # Chirality condition check
        ChieralityCondition = r3 @ (Points3D.T - C2.reshape((3,1)))
        Positive_Z = Points3D[:,2] > 0
        inlier_count = (ChieralityCondition > 0 * Positive_Z).sum()
        check = inlier_count+Positive_Z.sum()
        if (check > MaxInlierCounts):
            MaxInlierCounts = check
            Unique_R = R2
            Unique_T = C2
            Unique_3DPoints = Points3D
            BestPoseNum = i+1

    # Plot all 3D points and camera poses
    PlotInitialTriangulation(CameraPoses=CameraPoses, AllPoints=AllPoints)
    
    # Plot Unique Camera Pose
    PlotTriangulation(Unique_3DPoints, Unique_T, BestPoseNum)

    return Unique_3DPoints, Unique_R, Unique_T