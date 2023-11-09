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
resultpath = os.getcwd() + '/result_10000/'