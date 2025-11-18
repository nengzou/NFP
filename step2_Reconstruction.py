import numpy as np
import cv2
import open3d as o3d
from scipy.io import loadmat
import cal_absolute_phase
import cal_xyz

# 数据文件夹路径
data_folder = "E:\Project/firstArtical\data/ball/45"
output_filename = "/ball.ply"#再data_folder下新建一个结果
N = 12
n = 4
num = n + 2 + 1
B_min = 10  # 低于这个调制度的我们认为它的相位信息不可靠
IT = 0.5  # 格雷码阈值
win_size = 7  # 中值滤波窗口大小
# 输入参数
width = 2448  /2# 相机宽度
height = 2048 /2 # 相机高度
prj_width = 1920  # 投影仪宽度
cam_calib_result='data\calib_ball/CamCalibResult45.npz'
prj_calib_result='data\calib_ball/PrjCalibResult45.npz'
# 加载表面法线
# nx = np.loadtxt('nx.txt')
# ny = np.loadtxt('ny.txt')
# nz = np.loadtxt('nz.txt')

# 加载相机和投影仪的标定结果
cam_calib_result = np.load(cam_calib_result)

prj_calib_result = np.load(prj_calib_result)

# 读取测试图片并计算三维重建
idx = 1
files_phaseShiftX = [f"{data_folder}/{idx + i}.bmp" for i in range(N)]
files_grayCodeX = [f"{data_folder}/{idx + N + i}.bmp" for i in range(num)]

# 计算绝对相位
phaX = cal_absolute_phase.calc_absolute_phase(files_phaseShiftX, files_grayCodeX, IT, B_min, win_size)
up_test_obj = phaX * 2 * np.pi
x_p = phaX * prj_width
cal_absolute_phase.plot_a_row(phaX,"jueduixiangwei 921hang",921)

xyz=cal_xyz.construction(x_p,cam_calib_result,prj_calib_result)

# 定义保存路径和文件名
output_file = data_folder + output_filename
cal_xyz.save_open3D_ply(output_file,xyz)