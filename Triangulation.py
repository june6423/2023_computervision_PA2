import numpy as np
import cv2
import os
from numpy.linalg import svd
from numpy.linalg import inv
from numpy.linalg import norm

def Triangulation(key_points, camera_pose, closest, cur_index, key_points_index, is_3d,points_3d,inlinear):
    
    datapath = os.getcwd() + '/Data/'
    camera_matrix = np.loadtxt(datapath + 'intrinsic.txt')
    #camera_matrix = np.load(datapath + 'intrinsic.npy')
    inv_camera_matrix = inv(camera_matrix)
    
    pose1 = camera_pose[cur_index]
    pose2 = camera_pose[closest]
    
    for idx, item in enumerate(key_points_index[0]):
        p1 = inv_camera_matrix @ np.append(key_points[cur_index][item],1)
        p1 = p1[0:2]
        p2 = inv_camera_matrix @ np.append(key_points[closest][key_points_index[1][idx]],1)
        p2 = p2[0:2]
        A = [p1[1]*pose1[2,:] - pose1[1,:],
                -p1[0]*pose1[2,:] + pose1[0,:],
                p2[1]*pose2[2,:] - pose2[1,:],
                -p2[0]*pose2[2,:] + pose2[0,:]]
        A = np.array(A).reshape(4,4)
        AA = A.T @ A
        _, _, v = svd(AA)
        
        new_3d = v[-1,:3]/v[-1,-1]
        new_3d.reshape(1,3)

        if item in is_3d[closest]:
            if(norm(points_3d[is_3d[closest][item]]-new_3d) < 5e-4):
                inlinear.append({cur_index:key_points_index[0][idx]})
                is_3d[cur_index][key_points_index[0][idx]] = is_3d[closest][item]
        else:
            proj = camera_pose[closest] @ (np.append(new_3d,1))
            proj /= proj[-1]
            if(norm(proj -(np.append(p2,1))) < 5e-4):
                new_3d = new_3d.reshape(1,3)
                points_3d = np.concatenate((points_3d,new_3d),axis=0)
                inlinear.append({cur_index:key_points_index[0][idx], closest:key_points_index[1][idx]})
                is_3d[cur_index][key_points_index[0][idx]] = len(points_3d)-1
                is_3d[closest][key_points_index[1][idx]] = len(points_3d)-1
    print("3D points",len(points_3d))
    return points_3d, inlinear, is_3d