"""
该程序针对不同的系统结构，通过图片计算出物体的点云

1、采集系统的参数初始化
    系统设计，包括使用什么硬件，投影什么图，离线还是在线等
    搭建由普通相机和投影仪组成的系统离线采集图片
    设计投影光的结构：12步相移和4位互补格雷码
    生成图片和采集图片。


2、进入标定子程序
    标定子程序通过读入多张含标定板图片来计算相机和投影仪的内外参数
    这些参数是重构所必须的

    读入相应的图片

3、进入重构子程序
    通过所拍摄的图片和相机内外参数、系统参数来计算物体的三维点云
    这个方法是核心。由


4、结果存储与展示
    将各中间参数以及最终结果根据需求进行保存与展示


"""
import os
import cv2
import numpy as np
from scipy.io import savemat
from scipy import interpolate
from tqdm import tqdm
import time
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
import cal_absolute_phase
import json

calib_folder = "E:\Project/firstArtical\data\calib_ball"#文件是用fenge.py文件运行后获取的。
calib_sets = 6
Bright_img = "19.bmp"   #构建的文件路径是E:\Project/firstArtical\data\calib\1\45\19.bmp
Board_Size = (9, 11)  # 角点数为 11列 × 9行（对应 99 个点）
prj_Size = (1920,1200)#投影仪的分辨率
interval = 10

N = 12
n = 4
num = n + 2+1
B_min = 5
IT = 0.5
win_size = 3
Prj_Calib_Resultname = calib_folder + "/PrjCalibResult45.npz"
Cam_Calib_Resultname = calib_folder + "/CamCalibResult45.npz"
obj_Points = []
rec_boardPoints = []
prj_Points = []
# 生成对象点 (X, Y, 0)
objp = np.zeros((Board_Size[0] * Board_Size[1], 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:Board_Size[0], 0:Board_Size[1]].T.reshape(-1, 2) * interval

print("对象点示例（前5个）:\n", objp[:5])

params = cv2.SimpleBlobDetector_Params()        #创建检测器对象
params.maxArea = 3000
params.minArea = 30
params.minDistBetweenBlobs = 30
blobDetector = cv2.SimpleBlobDetector_create(params) #创建检测器
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0000000001)

for idx in tqdm(range(1, calib_sets + 1), "开始检测圆心"):
    # 因为是白点，需要颜色反过来
    file = os.path.join(calib_folder, str(idx),str(45) , Bright_img)
    img = 255 - cv2.imread(file, 0)
    ret, corners = cv2.findCirclesGrid(
            img, Board_Size, cv2.CALIB_CB_SYMMETRIC_GRID, blobDetector,None)

    img_c = np.stack((255 - img,) * 3, axis=-1)#生成一张彩色图
    cv2.drawChessboardCorners(img_c, Board_Size, corners, ret)
    cv2.imshow("corner", img_c)
    print(corners.shape)

    obj_Points.append(objp)  # 形状 (99, 3)
    rec_boardPoints.append(corners)#list
    cv2.waitKey(0)
cv2.destroyAllWindows()
# 将三维列表转换为 JSON 格式并写入文件
# 将 ndarray 转换为列表
# data_list_converted = [layer.tolist() for layer in rec_boardPoints]
# with open("rec_boardPoints.txt", "w", encoding="utf-8") as file:
#     json.dump(data_list_converted, file, indent=4)  # 使用缩进美化 JSON 格式
# cal_absolute_phase.plot_a_np(rec_boardPoints[3][:, 0 , 0],rec_boardPoints[3][:, 0 , 1],f"corners{3}")
boarpoints=rec_boardPoints
image_size = img.shape[:2]
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_Points, boarpoints, image_size, None, None)
print(ret,"\ncameracanshu\n",camera_matrix)
np.savez(Cam_Calib_Resultname,Kc=camera_matrix,Rc_1=rvecs,Tc_1=tvecs)
#思路是识别标定板的点，在投影仪上的像素位置
#根据图片上的标定板解相位，匹配到识别圆心位置的相位，这些相位*投影仪分辨率=投影仪像素位置。
# 投影仪标定点识别
for calib_idx in range(calib_sets):
    print(f"标定第{calib_idx+1}组")
    data_folder = os.path.join(calib_folder, str(calib_idx+1),str(45))

    # 读取相移和格雷码图像文件
    files_phaseShiftX = [os.path.join(data_folder, f"{idx}.bmp") for idx in range(1, N + 1)]
    files_grayCodeX = [os.path.join(data_folder, f"{idx}.bmp") for idx in range(N + 1, N + num + 1)]
    files_phaseShiftY = [os.path.join(data_folder, f"{idx}.bmp") for idx in range(N + num + 1, 2 * N + num + 1)]
    files_grayCodeY = [os.path.join(data_folder, f"{idx}.bmp") for idx in range(2 * N + num + 1, 2 * N + 2 * num + 1)]

    # 计算绝对相位
    phaX = cal_absolute_phase.calc_absolute_phase(files_phaseShiftX, files_grayCodeX, IT, B_min, win_size)
    phaY = cal_absolute_phase.calc_absolute_phase(files_phaseShiftY, files_grayCodeY, IT, B_min, win_size)

    # if( calib_idx == 3 ) :
    #     cal_absolute_phase.plot_a_row(phaX,"no.4 absolutephax",769)

    phaX = phaX * prj_Size[0]
    phaY = phaY * prj_Size[1]

    points = np.zeros_like(corners)

    for i in range(Board_Size[0] * Board_Size[1]):#找99个点的投影仪像素

        x, y = rec_boardPoints[calib_idx][i, 0 , :]
        x_round, y_round = round(x), round(y)

        pha_x = phaX[y_round,x_round]
        pha_y = phaY[y_round,x_round]
        # pha_x = phaX[x_round, y_round]
        # pha_y = phaY[x_round, y_round]

        points[i,0,0] = pha_x
        points[i,0,1] = pha_y

    # cal_absolute_phase.plot_a_np(rec_boardPoints[calib_idx][:, 0, 0], rec_boardPoints[calib_idx][:, 0, 1], f"corners_for{calib_idx}")
    prj_Points.append(points)

prj_ret, prj_matrix, prj_dist_coeffs, prj_rvecs, prj_tvecs = cv2.calibrateCamera(
    obj_Points, prj_Points, prj_Size, None, None)
print("\n重投影误差\n",prj_ret,"\nprjcanshu\n",prj_matrix)
np.savez(Prj_Calib_Resultname,Kp=prj_matrix,Rc_1=prj_rvecs,Tc_1=prj_tvecs)

#重构
# 加载相机和投影仪的标定结果
