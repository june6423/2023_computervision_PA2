Run(RANSAC + Triangulation) : python main.py
Run(Bundle Adjustment) : python Bundle.py
Run(create ply file) : python npytoply.py

File explanation
Data : Original data (provided by TA)
functions : MATLAB function (provided by TA)
optional : custom dataset initial data (provided by TA)
two_view_recon_info : initial data (provided by TA)
custom dataset : image of custom dataset and camera intrinsic 
result_10000 : our result containing camera pose, 3d points, keypoints (npy file)

main.py : RANSAC + Triangulation run code
func.py : progress bar visualization code
npytoply.py : convert npy file to ply file (for mashlab visualization)
RANSAC.py : RANSAC code
Triangulation.py : Triangulation code
Bundle.py : Bundle Adjustment code

original_3d.ply : ply file of initial 3d points (provided by TA. sfm03 and sfm04 matched)
10000_output.ply : ply file of output 3d points
