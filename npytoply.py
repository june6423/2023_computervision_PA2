import numpy as np
from plyfile import PlyData, PlyElement

import os
infopath = os.getcwd() + '/two_view_recon_info/'
#infopath = os.getcwd() + '/optionsl/'
resultpath = os.getcwd() + '/result_10000/'
#resultpath = os.getcwd() + '/result_custom/'

points_3d = np.load(resultpath + '100_BA_result.npy')
#new_points_3d = np.load(resultpath + 'custom_result.npy')

data = points_3d


vertices = np.array([(float(data[i, 0]), float(data[i, 1]), float(data[i, 2])) for i in range(data.shape[0])], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

# Create PlyElement with vertex information
vertex_element = PlyElement.describe(vertices, 'vertex')

# Create PlyData object
plydata = PlyData([vertex_element])

# Save the PLY file
ply_file = 'BA_res.ply'
plydata.write(ply_file)

print(f'PLY file saved as {ply_file}')