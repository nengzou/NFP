#混合方法是：在解相位计算点xyz时，同步计算该点偏振法线，由多点的法线和同步计算的偏振法线做对比确定欧米噶的值
#由此来确定修正后的法线。
#由法线反过来计算该点的位置，对位置进行修正。由此达到提高精度。
#

import numpy as np
import cv2
import open3d as o3d
from scipy.io import loadmat
import cal_absolute_phase
import cal_xyz
import cal_normal


# 数据文件夹路径
data_folder = "H:\Project/firstArtical\data/shouYinJi"
output_filename = "fusiontest_allxyz_medea.ply"#在data_folder下新建一个结果
N = 12
n = 4
num = n + 2 + 1
B_min = 10  # 低于这个调制度的我们认为它的相位信息不可靠
IT = 0.5  # 格雷码阈值
win_size = 7  # 中值滤波窗口大小
combi_size=3  #融合数据的窗口大小
# 输入参数
width = 2448  /2# 相机宽度
height = 2048 /2 # 相机高度
prj_width = 1920  # 投影仪宽度
cam_calib_result='data\calib/CamCalibResult45.npz'
prj_calib_result='data\calib/PrjCalibResult45.npz'
files_component = [f"{data_folder}/{45 * i}/19.bmp" for i in range(4)]
# 加载表面法线
# nx = np.loadtxt('nx.txt')
# ny = np.loadtxt('ny.txt')
# nz = np.loadtxt('nz.txt')

# 加载相机和投影仪的标定结果
cam_calib_result = np.load(cam_calib_result)

prj_calib_result = np.load(prj_calib_result)

# 读取测试图片并计算三维重建
idx = 1
files_phaseShiftX = [f"{data_folder}/45/{idx + i}.bmp" for i in range(N)]
files_grayCodeX = [f"{data_folder}/45/{idx + N + i}.bmp" for i in range(num)]

# 计算绝对相位
phaX = cal_absolute_phase.calc_absolute_phase(files_phaseShiftX, files_grayCodeX, IT, B_min, win_size)
up_test_obj = phaX * 2 * np.pi
x_p = phaX * prj_width
cal_absolute_phase.plot_a_row(phaX,"jueduixiangwei 921hang",921)

#计算（H，W）的点坐标
all_xyz=cal_xyz.cal_all_xyz(x_p,cam_calib_result,prj_calib_result)

#融合计算
#计算法线
normal_vertor_sfp = cal_normal.calc_n(files_component)
print(len(normal_vertor_sfp))
points_list = []
# 从上到下，从左到右，进行搜索
for Y_pha in range(combi_size, x_p.shape[0] - combi_size):  # 从第2行开始，到600-1行

    for X_pha in range(combi_size, x_p.shape[1] - combi_size):  # 从第2列开始到1400-1列
        # 构造融合框
        roi_L = (X_pha - combi_size, Y_pha - combi_size, combi_size, combi_size)  # 范围的左上点坐标，窗口大小
        # BOX_L = x_p[roi_L[1]:roi_L[1] + roi_L[3], roi_L[0]:roi_L[0] + roi_L[2]]  # 绝对相位的范围内矩阵
        BOX_xyz=all_xyz[roi_L[1]:roi_L[1] + roi_L[3], roi_L[0]:roi_L[0] + roi_L[2]]  # 点云的范围内矩阵
        # if 0 in BOX_L:
        # if np.count_nonzero(BOX_L) < 3:
        if np.all(np.isnan(BOX_xyz)):#去除全为nan的点
        #     # print(f"数组包含 0，跳过处理")
            continue
        #得到了方框中9个点的坐标
        # 检查  检查每个点是否包含 NaN (返回 True 如果该点至少有一个 NaN)
        has_nan = np.isnan(BOX_xyz).any(axis=2)  # shape (8, 9)
        # 统计非 NaN 点的数量
        non_nan_count = np.sum(~has_nan)  # 取反后求和
        print(BOX_xyz.shape[0] * BOX_xyz.shape[1], "and", BOX_xyz[1, 1])

        if non_nan_count > 8:#符合要求9个点都有效，进行融合修正
            """先要将点展平，然后计算法线，然后融合法线，由法线修正点"""
            xyz_reshaped = BOX_xyz.reshape(-1, 3)#展平（9*3）

            normal_vector = cal_normal.fit_plane_normal(xyz_reshaped)  # 计算9个点的法线
            #jisuan角度差
            n1 = normal_vector
            n2 = normal_vertor_sfp[Y_pha*X_pha,:]
            theta = np.arccos(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2)))
            theta = np.degrees(theta)
            if theta>90:#如果夹角大于90，将偏振法线换向
                n2=-n2
            theta = np.arccos(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2)))
            #修正法线
            if theta < 20 :
                omiga=0.5
            else:
                omiga = 0.5 + ((theta-20) / 70) * 0.5 #theta越大，omiga越大，n1影响越大
            Nb = omiga*n1 +(1-omiga)*n2
            Nf = cal_normal.normalize_vector(Nb)#归一化
            #修正点
            #融合后的法线，求点在一定范围内使拟合法线最接近融合后法线的值
            f_point = cal_normal.point_optimize(xyz_reshaped, Nf)

            points_list.append(f_point)#存储融合点
            print("fusion complete!")

        else:#如果不符合要求，直接存入点
            points_list.append(BOX_xyz[1,1])#存储点


        # print(xyz)
        # print(len(xyz))
        # print(normal_vector)
        # print(normal_vertor_sfp[Y_pha*X_pha,:])#该点处的法线

        # print(f_point)
        #存储点  融合后的点和原本的点都要保留



print(len(points_list))
# 定义保存路径和文件名
output_file = data_folder + "/" + output_filename
cal_xyz.save_open3D_ply(output_file,points_list)