import numpy as np
import random
import cv2
import os
from numpy.linalg import inv
from numpy.linalg import norm
from func import *

def RANSAC(keypoint, points_3d, key_points_index, inlinear, is_3d, index):
    max_iteration = 10000
    max_inlier = 0
    max_pose = np.zeros((3, 4))
    threshold = 5e-4
    best_inlier = 0
    count = 0
    
    datapath = os.getcwd() + '/Data/'
    camera_matrix = np.loadtxt(datapath + 'intrinsic.txt')
    inv_camera_matrix = inv(camera_matrix)
    
    for iter in range(max_iteration):
        l = [i for i in range(len(key_points_index[1]))]
        #l = [i for i in range(len(points_3d))]
        idx = random.sample(l, 3)
        obj_point = []
        for i in range(3):
            obj_point.append(points_3d[is_3d[key_points_index[1][idx[i]]]])
            #obj_point.append(points_3d[idx[i]])
        obj_point = np.array(obj_point)
        img_point = []
        for i in range(3): 
            img_point.append(keypoint[key_points_index[0][idx[i]]])
            #img_point.append(keypoint[inlinear[idx[i]][index]])
        img_point = np.array(img_point)
        success, rvec, tvec = cv2.solveP3P(obj_point, img_point,camera_matrix,None,cv2.SOLVEPNP_P3P)
        pose = []
        for idx in range(len(rvec)):
            rotation = cv2.Rodrigues(rvec[idx])[0] #왜 rotation의 return값으로 3*3과 9*3행렬이 같이 나오지?
            translation = tvec[idx]
            #if np.any(points_3d.dot(rotation)+translation.T <0):
            #    continue
            pose.append(np.concatenate((rotation, translation),axis=1))
        count += len(pose) 
        for i in range(len(pose)):  
            inlier = 0
            for idx in range(len(inlinear)):
                if(index in inlinear[idx]):
                    proj = pose[i] @ (np.append(points_3d[idx],1))
                    proj /= proj[-1]
                    original = inv_camera_matrix @ np.append(keypoint[inlinear[idx][index]],1)
                    if(norm(proj - original) < threshold):
                        inlier += 1
            if(inlier > best_inlier):
                best_inlier = inlier
                best_pose = np.array(pose[i])
        printProgress(iter, max_iteration, 'Progress:', 'Complete', 1, 50)
    #if(best_inlier>0):
    print("\nbest inlier / count",best_inlier, count)
    return best_pose
