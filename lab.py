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
resultpath = os.getcwd() + '/result/'

sift = cv2.SIFT_create()

points_3d = np.load(infopath + '3D_points.npy')


pseudo_inlinear = np.load(infopath + 'inlinear.npy')
inlinear = []

for i in range(len(pseudo_inlinear)):
    inlinear.append({3:1, 4:2})

json_file_name = resultpath+'inlinear.json'

with open(json_file_name, 'w') as outfile:
    json.dump(inlinear, outfile)