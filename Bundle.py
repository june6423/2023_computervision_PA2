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

#import matplotlib.pyplot as plt

num_of_image = 15

datapath = os.getcwd() + '/Data/'
infopath = os.getcwd() + '/two_view_recon_info/'
funcpath = os.getcwd() + '/functions/'
resultpath = os.getcwd() + '/result/'

points_3d = np.load(resultpath+ '100_result.npy')
camera_pose = np.load(resultpath + '100_result_pose.npy')
camera_matrix = np.loadtxt(datapath + 'intrinsic.txt')

keypoint = [i for i in range(num_of_image)]
for i in range(num_of_image):
    keypoint[i] = np.load(resultpath + 'keypoints'+str(i)+'.npy')

inlinear = []
with open(resultpath+'inlinear_result.json') as json_file:
    inlinear = json.load(json_file)

param = dict()
param['K'] = camera_matrix
uv = []

for i in range(len(inlinear)):
    for k,v in inlinear[i].items():
        uv.append(np.array([keypoint[int(k)][v][0],keypoint[int(k)][v][1],1,v]))
        
param['uv'] = uv
param['nX'] = len(inlinear)
param['key1'] = 4
param['key2']  = 5
param['optimization'] = 1
param ['dof_remove'] = 0

X = []
for i in range(camera_pose.shape[0]):
    rotation = camera_pose[i][:3,:3]
    translation = (camera_pose[i][:3,3]).reshape(3,1)
    rvec = (cv2.Rodrigues(rotation)[0]).reshape(1,3)
    pose = np.concatenate((rvec.T,translation), axis = 0).flatten()
    X.append(pose)

X = np.array(X)
X = X.flatten()
np.append(X, points_3d.flatten())

eng = matlab.engine.start_matlab()
eng.cd(funcpath, nargout=0)
lm = eng.LM2_iter_dof(X, param)

