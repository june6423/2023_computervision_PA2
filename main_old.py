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

#import matplotlib.pyplot as plt

#eng = matlab.engine.start_matlab()

initial_index = [3,4]
ratio_test = 0.95
datapath = os.getcwd() + '/Data/'
#datapath = os.getcwd() + '/Crop/'
infopath = os.getcwd() + '/two_view_recon_info/'

imglist = [file for file in os.listdir(datapath) if file.endswith('.jpg')]
imglist.sort()
image = []

for idx in range(len(imglist)):
    img_array = np.fromfile(datapath + imglist[idx], dtype=np.uint8) #경로가 영어면 이럴 필요가 없습니다. 로컬 경로가 한글이더라구요...
    image.append(cv2.imdecode(img_array, cv2.IMREAD_COLOR))

print(len(imglist),"Image loaded")
print("Image size: ", image[0].shape)

remaining = [i for i in range(len(imglist))]
matched = []

matched.append(initial_index[0])
del remaining[initial_index[0]]
matched.append(initial_index[1]-1)
del remaining[initial_index[1]-1]

sift = cv2.SIFT_create()

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
inlinear = []

for i in range(len(pseudo_inlinear)):
    inlinear.append({3:key_points_index[0][pseudo_inlinear[i]], 4:key_points_index[1][pseudo_inlinear[i]]})
    is_3d[3][key_points_index[0][pseudo_inlinear[i]]] = i
    is_3d[4][key_points_index[1][pseudo_inlinear[i]]] = i
for idx, item in enumerate(remaining):
    key, descriptor[item] = sift.detectAndCompute(image[item],None)
    for i in range(len(key)):
        key_points[item].append([j for j in key[i].pt])
    key_points[item] = np.array(key_points[item])
    
while(len(remaining)>0):
    matched_points = np.zeros(len(imglist))
    good = [[] for i in range(len(remaining))]
    for idx in range(len(remaining)):
        matched_des = []
        closest = min(matched, key=lambda x: abs(x-remaining[idx]))
        for i in range(len(inlinear)):
            if closest in inlinear[i]:
                matched_des.append(descriptor[closest][inlinear[i][closest]]) 
        matched_des = np.array(matched_des)
        matcher = cv2.BFMatcher().knnMatch(matched_des, descriptor[remaining[idx]],k=2)
        for m,n in matcher:
            if m.distance < ratio_test*n.distance:
                good[idx].append([m])
        matched_points[idx] = len(good[idx])
    best_index = matched_points.argmax()
    print("Matched points",matched_points)
    key_points_index = [[] for i in range(2)]
    is_matched = [{} for i in range(2)]
    
    for idx,item in enumerate(good[remaining[best_index]]):
        key_points_index[0].append(item[0].trainIdx)
        key_points_index[1].append(inlinear[item[0].queryIdx][closest])
        is_matched[0][item[0].trainIdx] = idx
        is_matched[1][inlinear[item[0].queryIdx][closest]] = idx
        inlinear[item[0].queryIdx][remaining[best_index]] = item[0].trainIdx
    camera_pose[remaining[best_index]] = RANSAC(key_points[remaining[best_index]], points_3d, key_points_index ,inlinear,is_3d[closest], remaining[best_index])
    matcher = cv2.BFMatcher().knnMatch(descriptor[closest], descriptor[remaining[best_index]],k=2)
    good = []
    for m,n in matcher:
        if m.distance < ratio_test*n.distance:
            good.append([m])
    
    key_points_index = [[] for i in range(2)]
    is_matched = [{} for i in range(2)]
    for idx,item in enumerate(good):
        key_points_index[0].append(item[0].trainIdx)
        key_points_index[1].append(item[0].queryIdx)
        is_matched[0][item[0].trainIdx] = idx
        is_matched[1][item[0].queryIdx] = idx
        
    points_3d, inlinear,is_3d = Triangulation(key_points, camera_pose, closest, remaining[best_index], key_points_index, is_3d,points_3d,inlinear)
    matched.append(remaining[best_index])
    np.save(infopath + 'New_3D_points.npy',points_3d)
    del remaining[best_index]

#Bundle Adjustment