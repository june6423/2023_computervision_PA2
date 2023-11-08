import cv2
import os
import numpy as np

from functions import *
from RANSAC import *
from Triangulation import *
from func import *

import matplotlib.pyplot as plt
    
    
initial_index = [3,4]
ratio_test = 0.95
datapath = os.getcwd() + '/Data/'
infopath = os.getcwd() + '/two_view_recon_info/'

image2 = cv2.imread(datapath + 'sfm02.jpg')
image3 = cv2.imread(datapath + 'sfm03.jpg')


sift = cv2.SIFT_create()

key_points = [[] for i in range(15)]
descriptor = [[] for i in range(15)]

key_points[3] = np.load(infopath + 'sfm03_keypoints.npy')
key_points[4] = np.load(infopath + 'sfm04_keypoints.npy')
descriptor[3] = np.load(infopath + 'sfm03_descriptors.npy')
descriptor[4] = np.load(infopath + 'sfm04_descriptors.npy')

item = 2
key, descriptor[item] = sift.detectAndCompute(image2,None)

for i in range(len(key)):
    key_points[item].append([j for j in key[i].pt])
key_points[item] = np.array(key_points[item])
    
closest = 3
best_index = 2

matcher = cv2.BFMatcher().knnMatch(descriptor[closest], descriptor[best_index],k=2)
good = []
for m,n in matcher:
    if m.distance < ratio_test*n.distance:
        good.append([m])
img3 = cv2.drawMatchesKnn(image2,key_points[2],image3,key_points[3],good,None)
cv2.imwrite(datapath+'img3',img3)