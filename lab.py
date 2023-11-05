#import matlab.engine
import cv2
import os
import numpy as np
import copy

from functions import *
from RANSAC import *
from Triangulation import *
from Bundle import *
from func import *
from numpy.linalg import inv

#import matplotlib.pyplot as plt

#eng = matlab.engine.start_matlab()

initial_index = [3,4]
ratio_test = 0.95
datapath = os.getcwd() + '/PA2/Data/'
#datapath = os.getcwd() + '/Crop/'
infopath = os.getcwd() + '/PA2/two_view_recon_info/'

imglist = [file for file in os.listdir(datapath) if file.endswith('.jpg')]
imglist.sort()

remaining = [i for i in range(len(imglist))]
matched = []

matched.append(initial_index[0])
del remaining[initial_index[0]]
matched.append(initial_index[1]-1)
del remaining[initial_index[1]-1]


points_3d = np.load(infopath + '3D_points.npy')

camera_pose = [[] for i in range(len(imglist))]
camera_pose[3] = np.load(infopath + 'sfm03_camera_pose.npy')
camera_pose[4] = np.load(infopath + 'sfm04_camera_pose.npy')

key_points = [[] for i in range(len(imglist))]
descriptor = [[] for i in range(len(imglist))]
is_3d = [{} for i in range(len(imglist))]
is_matched = [{} for i in range(2)]
key_points_index = [[] for i in range(2)]

key_points[3] = np.load(infopath + 'sfm03_keypoints.npy')
key_points[4] = np.load(infopath + 'sfm04_keypoints.npy')
descriptor[3] = np.load(infopath + 'sfm03_descriptors.npy')
descriptor[4] = np.load(infopath + 'sfm04_descriptors.npy')
key_points_index[0] = np.load(infopath + 'sfm03_matched_idx.npy')
key_points_index[1] = np.load(infopath + 'sfm04_matched_idx.npy')
pseudo_inlinear = np.load(infopath + 'inlinear.npy')
#answer_inlinear = []
inlinear = []
#points_3d = []

is_3d = [{} for i in range(len(imglist))]
camera_matrix = np.loadtxt(datapath + 'intrinsic.txt')
inv_camera_matrix = inv(camera_matrix)

for i in range(len(pseudo_inlinear)):
    inlinear.append({3:key_points_index[0][pseudo_inlinear[i]], 4:key_points_index[1][pseudo_inlinear[i]]})
    is_3d[3][key_points_index[0][pseudo_inlinear[i]]] = i
    is_3d[4][key_points_index[1][pseudo_inlinear[i]]] = i
    
nn = RANSAC(key_points[4], points_3d, key_points_index ,inlinear, is_3d[4], 4)
           
#points_3d, inlinear,is_3d = Triangulation(key_points, camera_pose, 4, 3, key_points_index, is_3d,points_3d,inlinear)
