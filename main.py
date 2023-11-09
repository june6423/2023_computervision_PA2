import cv2
import os
import numpy as np

from functions import *
from RANSAC import *
from Triangulation import *
from func import *

import json
#import matplotlib.pyplot as plt

initial_index = [3,4]
ratio_test = 0.95
datapath = os.getcwd() + '/Data/'
infopath = os.getcwd() + '/two_view_recon_info/'
resultpath = os.getcwd() + '/result_10000/'

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
matched.append(initial_index[1])
del remaining[initial_index[1]-1]

sift = cv2.SIFT_create()

points_3d = np.load(infopath + '3D_points.npy')

camera_pose = [[] for i in range(len(imglist))]
camera_pose[3] = np.load(infopath + 'sfm03_camera_pose.npy')
camera_pose[4] = np.load(infopath + 'sfm04_camera_pose.npy')
#camera_pose[11] = np.load(infopath + 'sfm11_camera_pose.npy')
#camera_pose[12] = np.load(infopath + 'sfm12_camera_pose.npy')

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

""" key_points[11] = np.load(infopath + 'sfm11_keypoints.npy')
key_points[12] = np.load(infopath + 'sfm12_keypoints.npy')
descriptor[11] = np.load(infopath + 'sfm11_descriptors.npy')
descriptor[12] = np.load(infopath + 'sfm12_descriptors.npy')
key_points_index[0] = np.load(infopath + 'sfm11_matched_idx.npy')
key_points_index[1] = np.load(infopath + 'sfm12_matched_idx.npy') """
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
    good = [[] for i in range(len(imglist))]
    for idx in range(len(remaining)):
        matched_des = []
        closest = min(matched, key=lambda x: abs(x-remaining[idx]))
        for i in range(len(inlinear)):
            if(closest in inlinear[i]):
                matched_des.append(descriptor[closest][inlinear[i][closest]]) 
        matched_des = np.array(matched_des)
        matcher = cv2.BFMatcher().knnMatch(matched_des, descriptor[remaining[idx]],k=2)
        for m,n in matcher:
            if m.distance < ratio_test*n.distance:
                good[remaining[idx]].append([m])
        matched_points[remaining[idx]] = len(good[remaining[idx]])
    best_index = matched_points.argmax()
    if best_index not in remaining:
        print("NOT in remaining")
        break
    print("Matched points / best index ",best_index, matched_points)
    key_points_index = [[] for i in range(2)]
    is_matched = [{} for i in range(2)]
    memory = []
    
    for i in range(len(inlinear)):
        if (closest in inlinear[i]):
            memory.append(i)
    
    for idx,item in enumerate(good[best_index]):
        try:
            key_points_index[0].append(item[0].trainIdx) #best_index의 key point index
            key_points_index[1].append(inlinear[memory[item[0].queryIdx]][closest]) #closest의 key point index
            is_matched[0][item[0].trainIdx] = idx
            is_matched[1][inlinear[memory[item[0].queryIdx]][closest]] = idx
            inlinear[memory[item[0].queryIdx]][best_index] = item[0].trainIdx
        except:
            #print("error occured on query index",item[0].queryIdx,len(good[best_index]),len(memory))
            continue
    camera_pose[best_index] = RANSAC(key_points[best_index], points_3d, key_points_index ,inlinear,is_3d[closest], best_index)
    if type(camera_pose[best_index]) == int and camera_pose[best_index] == 0:
        print("RANSAC failed")
        break
    matcher = cv2.BFMatcher().knnMatch(descriptor[closest], descriptor[best_index],k=2)
    good = []
    for m,n in matcher:
        if m.distance < ratio_test*n.distance:
            good.append([m])
    
    print("good matching",len(good))
    key_points_index = [[] for i in range(2)]
    is_matched = [{} for i in range(2)]
    
    for i in range(len(inlinear)):
        if best_index in inlinear[i]:
            del inlinear[i][best_index]
    
    for idx,item in enumerate(good):
        key_points_index[0].append(item[0].trainIdx)
        key_points_index[1].append(item[0].queryIdx)
        is_matched[0][item[0].trainIdx] = idx
        is_matched[1][item[0].queryIdx] = idx
        
    points_3d, inlinear, is_3d = Triangulation(key_points, camera_pose, closest, best_index, key_points_index, is_3d,points_3d,inlinear)
    matched.append(best_index)
    #np.save(infopath + 'New_3D_points.npy',points_3d)
    remaining.remove(best_index)

np.save(resultpath + 'custom_result.npy',points_3d)
#np.save(resultpath + 'custom_result_pose.npy',camera_pose)
for i in range(len(key_points)):
    np.save(resultpath + 'keypoints'+str(i)+'.npy',key_points[i])

json_file_name = resultpath+'inlinear_result.json'

inlinear_result = []
for d in inlinear:
    inlinear_result.append({int(k):int(v) for k, v in d.items()})
            
with open(json_file_name, 'w') as outfile:
    json.dump(inlinear_result, outfile)