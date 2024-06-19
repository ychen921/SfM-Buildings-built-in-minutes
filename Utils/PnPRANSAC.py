import numpy as np

def PnPRANSAC(Points3D, Points2D, K):
    max_inlier_count = 0
    MaxIteration = 5000
    iter_count = 0
    NumSample = 6 #

    while MaxIteration > iter_count:
        pass