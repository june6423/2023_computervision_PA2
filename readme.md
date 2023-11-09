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
custom_output.ply : ply file of custom dataset 3d points
10000_BA_output.ply : ply file of optimized 3d points(bundle adjustment)
noisy_output.ply : ply file of adding noise (10000_output + gaussian noise)

데이터 구조 설명(result_10000 폴더 내부 파일 설명)
keypoints0.npy : 0번 이미지의 SIFT결과 keypoint를 저장. [x,y]형태의 N*2 배열
10000_result.npy : Triangulation을 통해 복원한 3D점의 좌표. [x,y,z]형태의 N*3 배열
10000_result_pose.npy : RANSAC을 통해 얻은 각 camera pose. 15*4*3배열
inlinear_result.json : 3D점이 어떤 keypoint와 매칭되어있는지 알려주는 배열. list형태로 각 원소마다 dict가 저장되어 있다.
하나의 keypoint는 최대 1개의 3D점과 매칭됨이 보장되어있다.
ex)0번째 3D점이 3번 이미지의 1000번째 keypoint와 4번 이미지의 2000번째 keypoint와 매칭된 경우, inlinear_result[0]={3:1000,2:1000}