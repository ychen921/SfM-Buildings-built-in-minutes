import numpy as np
import cv2

def PlotInliers(img1, img2, matches, idx):
    MatchImage = np.concatenate((img1, img2), axis=1)

    for i, pts in enumerate(matches):
        # print(pts.shape)
        if i in idx:
            x1, y1 = int(pts[3]), int(pts[4])
            x2, y2 = int(int(pts[5])+img1.shape[1]), int(pts[6]) 
            cv2.line(MatchImage,(x1,y1),(x2,y2),(0,255,0),1)
        else:
            x1, y1 = int(pts[3]), int(pts[4])
            x2, y2 = int(int(pts[5])+img1.shape[1]), int(pts[6]) 
            cv2.line(MatchImage,(x1,y1),(x2,y2),(0,0,255),1)

    MatchImage = cv2.resize(MatchImage, (0, 0), fx = 0.5, fy = 0.5)
    cv2.imwrite('../InliersMatches.jpg', MatchImage)
    