import numpy as np
import cv2
import matplotlib.pyplot as plt

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
    cv2.imwrite('./SaveFig/InliersMatches.jpg', MatchImage)

def PlotInitialTriangulation(CameraPoses, AllPoints):
    color_map = ['b', 'r', 'g', 'k']
    markers = ['^', '^', '<', '>']
    
    plt.figure(figsize=(6, 8))

    for i, pose in enumerate(CameraPoses):
        C2 = pose[1]
        camera_x, camera_z = C2[0], C2[2]
        
        points = AllPoints[i]
        points_x, points_z = points[:,0], points[:,2]

        plt.scatter(points_x, points_z, color=color_map[i], s=0.09)
        plt.scatter(camera_x, camera_z, color=color_map[i], alpha=0.5, 
                    s=150, marker=markers[i], label='Pose '+str(i+1))

    plt.xlabel('X')
    plt.ylabel('Z')
    plt.xlim(-15, 15) 
    plt.ylim(-20, 20)
    plt.grid()
    plt.legend()
    plt.title('Initial Triangulation')
    plt.savefig("./SaveFig/Initial_Triangulation.png")
    plt.show()
    
def PlotTriangulation(Pts, Translation, PoseNum):
    camera_x, camera_z = Translation[0], Translation[2]
    points_x, points_z = Pts[:,0], Pts[:,2]

    plt.figure(figsize=(6, 8))

    plt.scatter(points_x, points_z, color='b', s=0.09)
    plt.scatter(camera_x, camera_z, color='b', alpha=0.5, 
                s=150, marker='^', label='Best Camera Pose '+str(PoseNum))
    
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.xlim(-15, 15) 
    plt.ylim(-10, 30)
    plt.grid()
    plt.legend()
    plt.title('Linear Triangulation')
    plt.savefig("./SaveFig/Linear_Triangulation.png")
    plt.show()
