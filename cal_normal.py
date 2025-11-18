import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize_scalar

# 该函数将指定文件进行分割出四个不同偏振角度的图片，放在一个数组中（4，size0，size1）
def fenge2four(img_acq):
    image_origin = cv2.imread(img_acq,0)
    #获取图像大小
    imgsize = image_origin.shape
    img_acq2four = np.zeros((4, int(imgsize[0]/2), int(imgsize[1]/2)),dtype=np.uint8)#构建存储4张图片的数组
    a1 = np.arange(0, imgsize[0], 2)#行的偶数
    a2 = np.arange(0, imgsize[1], 2)#列的偶数
    a3 = np.arange(1, imgsize[0], 2)#行的奇数
    a4 = np.arange(1, imgsize[1], 2)#列的奇数
    # 将图片的偶数行，偶数列元素取出，并放在返回变量内
    img1 = image_origin[a1, :]
    img_acq2four[0,:,:] = img1[:, a2]
    img2 = image_origin[a1, :]
    img_acq2four[1,:,:] = img2[:, a4]
    img3 = image_origin[a3, :]
    img_acq2four[2,:,:] = img3[:, a2]
    img4 = image_origin[a3, :]
    img_acq2four[3,:,:] = img4[:, a4]
    return img_acq2four

#该函数将数据转化为unit8结构
def convert2u8(np_float):
    np_u8=np.array(np_float,dtype='uint8')
    return np_u8
#该函数将数据转化为float结构
def convert2float(np_u8):
    np_float=np.array(np_u8,dtype='float')
    return np_float

def calulate_from_stokes(I90,I45,I135,I0):
# 该函数用斯托克斯计算dolp和aop,dolp取值范围0-1；
# S_0=[█((I(0^° )+I(〖45〗^° )+I(〖90〗^° )+I(〖135〗^° ))/2@I(0^° )-I(〖90〗^° )@I(〖45〗^° )-I(〖135〗^° ) )]
    i0=np.array(I0,dtype=float)
    i45=np.array(I45,dtype=float)
    i90=np.array(I90,dtype=float)
    i135=np.array(I135,dtype=float)
    s_0_0=(i90+i45+i135+i0)/2
    s_0_1=i0-i90
    s_0_2=i45-i135
    dolp=np.sqrt(s_0_1*s_0_1+s_0_2*s_0_2)/s_0_0
    faiZero=0.5*np.arctan2(s_0_2 , s_0_1)
    return dolp,faiZero

def calculate_from_sin(I90, I45, I135, I0):
    #由sin的光强来计算dolp，假设光强呈sin变法。
    i0 = np.array(I0, dtype=float)
    i45 = np.array(I45, dtype=float)
    i90 = np.array(I90, dtype=float)
    i135 = np.array(I135, dtype=float)
    i_c1 = (i0 + i90) * 0.5
    i_c2 = (i45 + i135) * 0.5
    i_c = (i_c1+i_c2)/2
    fai = 0.5 * np.arctan( (i45 - i135) / (i0 + i90) )
    i_sv = (i45 - i135) / (2 * np.sin(2 * fai))
    dolp = i_sv / i_c1
    return dolp,fai

def show_3Dquiver(I0,n1,n2,n3):
    # 创建一个三维图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 定义起点、终点和箭头方向
    (length,width)=np.shape(I0)
    x = np.arange(width)
    y = np.arange(length)
    start_pointx, start_pointy = np.meshgrid(x, y)
    start_pointz = np.zeros((length, width))

    # 绘制箭头
    ax.quiver3D(start_pointx,start_pointy,start_pointz,n1,n2,n3,  # 定义了箭头的位置，方向
                pivot='tail', color='r')

    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim(0, 1.5)
    # ax.set_xlim(0, 1224)
    # ax.set_ylim(0, 1024)
    ax.view_init(elev=90, azim=0);#调整视图角度
    # 显示图形
    plt.show()


#功能：图片拼接，将4张图片拼接到一张大图上，按顺时针左上为第一张。
#目的：方便将四幅图放在一起对比
def pinjie_img(img1,img2,img3,img4):
    img_hang1 = np.concatenate((img1, img2), axis=1)
    img_hang2 = np.concatenate((img3, img4), axis=1)
    img = np.concatenate((img_hang1, img_hang2), axis=0)
    return img


def calc_n(files_component):
    # 1读取图片位置/文件名,初始化参数
    I0 = cv2.imread(files_component[0], cv2.IMREAD_GRAYSCALE)  # 以灰度图模式读取图像
    I45 = cv2.imread(files_component[1], cv2.IMREAD_GRAYSCALE)  # 以灰度图模式读取图像
    I90 = cv2.imread(files_component[2], cv2.IMREAD_GRAYSCALE)  # 以灰度图模式读取图像
    I135 = cv2.imread(files_component[3], cv2.IMREAD_GRAYSCALE)  # 以灰度图模式读取图像

    caizhi_zheshelv=1.8

    # 2滤波，高斯噪声和椒盐噪声较多。如何建立一个环境来调试滤波的效果

    I0 = cv2.GaussianBlur(I0, (7, 7), 0)
    I45 = cv2.GaussianBlur(I45, (7, 7), 0)
    I90 = cv2.GaussianBlur(I90, (7, 7), 0)
    I135 = cv2.GaussianBlur(I135, (7, 7), 0)
    I0 = cv2.medianBlur(I0, 7)  # 核大小需为奇数
    I45 = cv2.medianBlur(I45, 7)
    I90 = cv2.medianBlur(I90, 7)
    I135 = cv2.medianBlur(I135, 7)

# 3计算dolp,aop和法线
# S0=I0+I90;S1=I0-I90;S2=I45-I135
#dolp=sqrt(s1**2+s2**2)/s0;aop=arctan(s2/s1)/2
#I_C=1/2 (I_0+I_90 )=1/2 (I_45+I_135 )
    np.seterr(divide="ignore",invalid="ignore")#消除警告
    S0=np.array(I0,dtype=float)+np.array(I90,dtype=float)
    uint_S0=np.array(S0,dtype='uint8')#新建数组，将S0的数据结构从float转换为uint8
    S1=np.array(I0,dtype=float)-np.array(I90,dtype=float)
    S2=np.array(I45,dtype=float)-np.array(I135,dtype=float)
    dolp_img=255*np.sqrt(S1**2+S2**2)/S0
    aop_img=0.5*np.arctan(S2/S1)/2


# 保存图像
# cv2.imwrite('dolp_img.jpg', dolp_img)

# sin_dolp,fai=calculate_from_sin(I90, I45, I135, I0)
    stokes_dolp,faiZero=calulate_from_stokes(I90, I45, I135, I0)#根据施柏鑫论文，求dolp和有歧义的fai0，用的是没加pi的。
# fai180=fai/np.pi*180
# uint_dolp_img=dolp_img.astype(np.uint8)
# aop_max=np.max(aop_img)
# print("aopmax=",aop_max)
##下面都是使用斯托克斯矢量计算的法向量
    yiTa=caizhi_zheshelv*np.full(np.shape(I0),1,float)
    a=yiTa**4
    fenzi=yiTa**4*(1-stokes_dolp**2)+2*(2*stokes_dolp**2+stokes_dolp-1)*yiTa**2+stokes_dolp**2+2*stokes_dolp-4*yiTa**3*stokes_dolp*(np.sqrt(1-stokes_dolp**2))+1
    fenmu=(stokes_dolp+1)**2*(yiTa**4+1)+2*yiTa**2*(3*stokes_dolp**2+2*stokes_dolp-1)
    cos_sita=np.sqrt(fenzi/fenmu)
    siTa=np.arccos(cos_sita)
    n1=np.sin(siTa)*np.cos(faiZero)#n1,n2,n3为单位向量了
    n2=np.sin(siTa)*np.sin(faiZero)#n的维度为图片的宽*高
    n3=np.cos(siTa)
    # 创建法线云数据
    # 将三个方向的分量，按照展平堆垛成向量的形式。
    normal_vertor_sfp = np.vstack((n1.flatten(), n2.flatten(), n3.flatten())).T
    return normal_vertor_sfp
#图片拼接
#四张图要有同样的格式和数据类型
# img_four=pinjie_img(convert2u8(stokes_dolp*255),convert2u8(sin_dolp*255),I0,I0)

#保存图片
# cv2.imwrite("acq_img/zifenge.bmp",img_four)
# print("保存成功！")

#显示图片
# cv2.namedWindow("image_origin",cv2.WINDOW_NORMAL)
# cv2.resizeWindow("image_origin",np.shape(I0)[1]*2,np.shape(I0)[0]*2)
# cv2.imshow("image_origin",convert2u8(stokes_dolp*255))
# key=cv2.waitKey(0)

#保存法线文件
#因数组精度太高没太多必要# 使用np.around()函数将数组四舍五入到小数点后两位,没什么用
# nx = np.around(n1, decimals=4)
# ny = np.around(n2, decimals=4)
# nz = np.around(n3, decimals=4)
#构建（n1，n2，n3）数组，然后存储成txt，或者分开保存成nx，ny，nz三个文件。

# np.savetxt('nx.txt', n1, delimiter=',')
# np.savetxt('ny.txt', n2, delimiter=',')
# np.savetxt('nz.txt', n3, delimiter=',')
#显示法线图
#调用函数绘制法线，用I0获取图像大小，mesh一下带序号的数组，作为X，Y，创建一个0数组，作为Z值。
#绘制箭头需要起始点的坐标，上面就是构建的坐标。
# show_3Dquiver(I0,n1,n2,n3)

def fit_plane_normal(points):
    """
    通过最小二乘法拟合平面，返回法线向量
    代码解释
    质心计算        质心是所有点的平均坐标，用于平移坐标系到平面中心。
    去中心化        将每个点减去质心，使数据围绕原点分布，便于计算平面方向。
    SVD分解       对去中心化的数据矩阵进行奇异值分解（SVD），协方差矩阵的特征向量对应数据的主成分方向。
    最后一个右奇异向量（vh[2, :]）对应数据变化最小的方向，即平面的法线方向。
    方向调整        法线方向可能指向平面两侧，通过判断 z 分量的符号确保方向一致（例如朝上）。
    验证          检查法线与每个点到质心的向量的点积是否接近0（理论上严格垂直时点积为0）。
    # 可选：验证法线与各点是否垂直（点积接近0）
    centroid = np.mean(points, axis=0)
    for point in points:
        vector_to_point = point - centroid
        dot_product = np.dot(normal_vector, vector_to_point)
        print(f"点 {point} 与法线的点积: {dot_product:.6f}")  # 应接近0
    """
    # 1. 计算质心（所有点的均值）
    centroid = np.mean(points, axis=0)

    # 2. 去中心化：将点平移到质心坐标系
    centered = points - centroid

    # 3. 计算协方差矩阵的SVD分解
    _, _, vh = np.linalg.svd(centered)

    # 4. 法线向量是最后一个右奇异向量（对应最小奇异值）
    normal = vh[2, :]

    # 确保法线方向一致（可选：根据需求调整方向）
    # 例如，使法线朝上（z分量为正）
    if normal[-1] < 0:
        normal *= -1

    return normal

def normalize_vector(v):
    """
    单位化v
    :param v: 向量
    :return: v的单位向量
    """
    norm = np.linalg.norm(v)  # 计算 L2 范数
    if norm == 0:
        return v  # 避免除以 0，如果范数是 0，直接返回原向量
    return v / norm

def point_optimize(xyz,target_n):
    #输入为待优化的9个点坐标和目标法向量，输出优化后的点坐标
    z_init = xyz[4,2]   # 第4行，第2列（z坐标）第五个点
    # 定义优化目标：最小化 |f(x) - target_n|
    def objective(x):
        xyz[4,2] = x
        n1 = fit_plane_normal(xyz)
        n2 = target_n
        theta = np.arccos(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2)))
        return theta

    # 在 x ∈ [z_range-1, z_range+1] 范围内寻找最优解
    result = minimize_scalar(objective, bounds=(z_init-0.5, z_init+0.5), method='bounded')
    xyz[4,2] = result.x

    return xyz[4]

if __name__ == "__main__":
    # 测试数据
    data_folder = "E:\Project/firstArtical\data\yinLiaoPingZi"
    files_component = [f"{data_folder}/{45 * i}/19.bmp" for i in range(4)]

    # 调用函数并打印结果
    normals = calc_n(files_component)


    print(np.shape(normals))
    # print(f"DoLP: {dolp:.4f}")  # 预期 1.0
    # print(f"AoP: {aop:.2f}°")   # 预期 26.57°
