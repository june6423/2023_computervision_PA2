import matlab.engine
import cv2
import os
import numpy as np
import copy

from functions import *
from RANSAC import *
from Triangulation import *
from Bundle import *
from func import *
import json

from numpy.linalg import inv
#import matplotlib.pyplot as plt

num_of_image = 15

datapath = os.getcwd() + '/Data/'
infopath = os.getcwd() + '/two_view_recon_info/'
funcpath = os.getcwd() + '/functions/'
resultpath = os.getcwd() + '/result_100/'

points_3d = np.load(resultpath+ '100_result.npy')
camera_pose = np.load(resultpath + '100_result_pose.npy')
camera_matrix = np.loadtxt(datapath + 'intrinsic.txt')
inv_camera_matrix = inv(camera_matrix)

keypoint = [[] for i in range(num_of_image)]
for i in range(num_of_image):
    keypoint[i] = np.load(resultpath + 'keypoints'+str(i)+'.npy')

inlinear = []
with open(resultpath+'inlinear_result.json') as json_file:
    inlinear = json.load(json_file)

param = dict()
param['K'] = camera_matrix
uv = [[] for i in range(num_of_image)]

for i in range(len(inlinear)):
    for k,v in inlinear[i].items():
        norm = inv_camera_matrix @ np.append(keypoint[int(k)][v],1)
        uv[int(k)].append(np.array([matlab.double(norm[0]),matlab.double(norm[1]),matlab.double(1),matlab.double(i+1)]))
        #uv[int(k)].append(np.array([keypoint[int(k)][v][0],keypoint[int(k)][v][1],1,i+1]))
for i in range(len(uv)):
    uv[i] = np.array(uv[i]).T
    uv[i] = uv[i].squeeze()

param['uv'] = uv
param['nX'] = len(inlinear)
param['key1'] = 4
param['key2']  = 5
param['optimization'] = 1
param['dof_remove'] = 0

X = []
for i in range(camera_pose.shape[0]):
    rotation = camera_pose[i][:3,:3]
    translation = (camera_pose[i][:3,3]).reshape(3,1)
    rvec = (cv2.Rodrigues(rotation)[0]).reshape(1,3)
    pose = np.concatenate((rvec.T,translation), axis = 0).flatten()
    X.append(pose)

X = np.array(X)
X = X.flatten()
points_3d = points_3d.flatten()
#points_3d = points_3d + np.random.normal(0,1,points_3d.shape)
X = np.append(X, points_3d)
X = matlab.double(X)

eng = matlab.engine.start_matlab()
eng.addpath(funcpath, nargout=0)
x_BA = eng.LM2_iter_dof(X, param)

eng.quit()

x_BA = np.array(x_BA[6*camera_pose.shape[0]:])
x_BA = x_BA.reshape(-1,3)
points_3d = points_3d.reshape(-1,3)

np.save(resultpath + '100_BA_result.npy', x_BA)
np.save(resultpath + 'noisy_points_3d.npy', points_3d)
