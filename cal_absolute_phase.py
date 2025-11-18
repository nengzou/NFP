#解相位
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

def calc_absolute_phase(files_phaseShift, files_grayCode, IT, B_min, win_size):
    """
    计算绝对相位

    参数:
    files_phaseShift : 相移图像文件列表
    files_grayCode : 格雷码图像文件列表
    IT : 格雷码阈值
    B_min : 调制度最小值
    win_size : 中值滤波窗口大小

    返回:
    pha_absolute : 绝对相位图
    dif : 相位差异图
    """
    N = len(files_phaseShift)
    pha_wrapped, B = calc_warppred_phase(files_phaseShift, N)
    #解出的pha_wrapped范围是0-2pi
    # 归一化到 0-255
    # normalized_data = cv2.normalize(pha_wrapped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # show_images_sequentially(normalized_data,"pha_wrapped")

    n = len(files_grayCode) - 2  # 投影了一黑一白两幅图片
    Ks1,Ks2,_ = calc_gray_code(files_grayCode, IT, n,win_size)

    mask1 = (pha_wrapped <= np.pi / 2)  # 相位小于 -pi / 2 时置1，大于置0。
    mask2 = ((np.pi / 2 < pha_wrapped) & (pha_wrapped < 3*np.pi / 2))  # 包裹相位大于 -pi / 2 且小于 pi / 2 时置1，否则置0。
    mask3 = (pha_wrapped >= 3*np.pi / 2)  # 包裹相位大于 pi / 2 置1，否则置0。
    #互补格雷码解码
    pha = (pha_wrapped + 2 * np.pi * Ks2) * mask1 + (pha_wrapped + 2 * np.pi * Ks1) * mask2 + (pha_wrapped + 2 * np.pi * Ks2 - 2 * np.pi) * mask3
    #普通格雷码解码
    # pha = pha_wrapped + 2*Ks1*np.pi

    pha_absolute = pha / (2 * np.pi * 2 ** n)  # 绝对相位归一化
    # 调制度滤波
    B_mask = B > B_min
    # pha_absolute = pha_absolute * B_mask
    pha_absolute = pha_absolute * B_mask
    print("pha_shape",pha.shape)
    # 边缘跳变误差
    # pha_absolute, dif = m_filter2d(pha_absolute, win_size)

    # 绘制pha中某行的二维图
    # plot_a_row(Ks2,"Ks2",932)
    # plot_a_row(pha_wrapped,"pha_wrapped",932)
    # plot_a_row(pha,"absolutpha",932)


    return pha_absolute#, dif

def calc_warppred_phase(files_phaseShift, N):
    """
    计算包裹相位和调制度
    参数:
    files : 图像文件列表
    N : 图像数量
    返回:
    pha : 包裹相位图
    B : 调制度图
    """
    # 初始化
    img = cv2.imread(files_phaseShift[0], cv2.IMREAD_GRAYSCALE)  # 以灰度图模式读取图像
    if img is None:
        raise FileNotFoundError(f"图像文件未找到: {files_phaseShift[0]}")
    h, w = img.shape
    sin_sum = np.zeros((h, w), dtype=np.float64)
    cos_sum = np.zeros((h, w), dtype=np.float64)

    # 读取每一幅图像
    for k in range(N):
        file = files_phaseShift[k]
        Ik = cv2.imread(file, cv2.IMREAD_GRAYSCALE)  # 以灰度图模式读取图像

        Ik = gaussian_blur(Ik, kernel_size=(3, 3), sigma_x=1)
        if Ik is None:
            raise FileNotFoundError(f"图像文件未找到: {file}")
        pk = 2.0 * k / N * np.pi
        sin_sum += Ik * np.sin(pk)
        cos_sum += Ik * np.cos(pk)

    # 计算相位和调制度
    pha = np.arctan2(sin_sum, cos_sum)
    B = np.sqrt(sin_sum**2 + cos_sum**2) * 2.0 / N

    # 因周期内是左斜的Z，0到pi到-pi到0的，需调整整数到单周期内
    pha = -pha
    e = -1e-10
    pha_low_mask = pha < e
    pha = pha + pha_low_mask * 2.0 * np.pi

    return pha, B

# 格雷码生成函数
def gray_code(n):
    if n < 1:
        print("格雷码数量必须大于0")
        return []
    elif n == 1:
        return ["0", "1"]
    else:
        code_pre = gray_code(n - 1)
        num = len(code_pre)
        code = [""] * (num * 2)
        # 第一步：每个字符串前面都+0
        for i in range(num):
            code[i] = "0" + code_pre[i]
        # 第二步：翻转首个元素，其余取对称
        for i in range(num):
            code[num + i] = "1" + code_pre[-1 - i]
        return code

def calc_gray_code(files_grayCode, IT, n,win_size):
    """
    计算格雷码解码结果

    参数:
    files : 图像文件列表,一次一组图片如4+2+1
    IT : 格雷码阈值
    n : 格雷码位数

    返回:
    Ks1 : 格雷码解码结果1
    Ks2 : 格雷码解码结果2
    gcs : 格雷码解码掩码
    """
    # 01 读取每一张图片进Is
    num = len(files_grayCode)
    img = cv2.imread(files_grayCode[0], cv2.IMREAD_GRAYSCALE)  # 以灰度图模式读取图像
    if img is None:
        raise FileNotFoundError(f"图像文件未找到: {files_grayCode[0]}")
    h, w = img.shape
    Is = np.zeros((num, h, w), dtype=np.float64)

    for i in range(num):
        img = cv2.imread(files_grayCode[i],cv2.IMREAD_GRAYSCALE)
        img = median_blur(img, kernel_size=win_size)
        Is[i, :, :] = img.astype(np.float64)

    # 02 计算Is_Max、Is_Min，对每个点进行阈值判断,计算出编码值
    Is_max = np.max(Is, axis=0)
    Is_min = np.min(Is, axis=0)
    Is_std = Is / (Is_max + 1e-10)  # 避免除以0
    gcs = Is_std > IT  # 生成格雷码阈值掩码
    img_gcs = np.where(gcs, 1, 0)  # 布尔类型二值为01
    img_gcs = (img_gcs * 255).astype(np.uint8)
    # plot_3d_surface(gcs[0])
    # 4位格雷码是7张图片

    # 开始解码
    n=n-1   #因n是从0开始，所以要-1
    Ks1 = np.zeros((h, w), dtype=np.int32)
    Ks2 = np.zeros((h, w), dtype=np.int32)
    gcs = np.where(gcs, 1, 0).astype(np.int32)#布尔类型二值为01
    V_list = [0,1,3,2,7,6,4,5,15,14,12,13,8,9,11,10]
    V_list_C = [0, 1, 3, 2, 7, 6, 4, 5, 15, 14, 12, 13, 8, 9, 11, 10,31,30,28,29,24,25,27,26,16,17,19,18,23,22,20,21]
    for v in range(h):
        if v==0 or v==h-1:
            print(f"第 {v + 1} 行")
        for u in range(w):
            # 不需要最后黑、白两幅图片的编码
            gc1 = gcs[:n, v, u].flatten()#flatten()：将 gc1 展平为一维数组（如果它原本不是一维的）
            V1 = 0
            for i in range(n):
                V1 += gc1[i] * (2 ** (n - i - 1))
            Ks1[v, u] = V_list[V1]

            gc2 = gcs[n, v, u]
            V2 = 0
            for i in range(n):
                V2 += gc1[i] * (2 ** (n - i))
            V2 += gc2
            Ks2[v, u] = V_list_C[V2]
            # Ks1[v, u] = V1K.get(V1, -1)
            # Ks2[v, u] = V2K.get(V2, -1)
    #test code

    # print("maxvalue_ks1",np.max(Ks1),"min",np.min(Ks1))


    Ks2 = np.ceil(Ks2/2)
    return Ks1, Ks2, gcs

def gaussian_blur(image, kernel_size=(5, 5), sigma_x=0):
    """
    对图像进行高斯滤波。

    参数:
        image (numpy.ndarray): 输入图像。
        kernel_size (tuple): 高斯核的大小，格式为 (宽度, 高度)。
                             必须是正奇数，例如 (5, 5)。
        sigma_x (float): 高斯核的标准差（X 方向）。如果为 0，则根据核大小自动计算。

    返回:
        numpy.ndarray: 高斯滤波后的图像。
    """
    # 检查核大小是否为正奇数
    if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
        raise ValueError("核大小必须是正奇数！")

    # 使用 OpenCV 的 GaussianBlur 函数进行高斯滤波
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigmaX=sigma_x)
    return blurred_image

def median_blur(image, kernel_size):
    """
    对图像进行中值滤波。

    参数:
        image (numpy.ndarray): 输入图像。
        kernel_size (int): 中值滤波的邻域大小，必须是正奇数。

    返回:
        numpy.ndarray: 中值滤波后的图像。
    """
    if kernel_size % 2 == 0:
        raise ValueError("邻域大小必须是正奇数！")

    blurred_image = cv2.medianBlur(image, kernel_size)
    return blurred_image

def plot_a_row(pha,title,row_index=322):
    # 绘制pha中某行的二维图
    # 选择要绘制的行（例如，第row_index行）

    selected_row = pha[row_index]

    # 获取列号（横坐标）
    column_indices = np.arange(len(selected_row))  # 生成列号数组 [0, 1, 2, ..., n-1]

    # 绘制二维图
    plt.figure(figsize=(8, 4))  # 设置图形大小
    plt.plot(column_indices, selected_row, marker='o', linestyle='-', color='b')  # 绘制折线图
    plt.scatter(column_indices, selected_row, color='r')  # 添加散点图以突出显示每个点

    # 添加标题和标签
    plt.title(f"number {row_index + 1} row 2D figure")
    plt.xlabel("N_column")
    plt.ylabel(title)

    # 显示网格
    plt.grid(True)

    # 显示图形
    plt.show()

def plot_a_np(x,y,title="np name"):
    # 创建一个空白图像
    # 绘制点图
    # x = points[:, 0, 0]
    # y = points[:, 0, 1]
    plt.scatter(x, y)

    # 显示坐标轴的标尺（默认情况下，matplotlib 会显示标尺）
    plt.grid(True)  # 显示网格线，方便观察
    plt.xlabel('X-axis')  # 设置 x 轴标签
    plt.ylabel('Y-axis')  # 设置 y 轴标签
    plt.title(title)  # 设置标题

    # 显示图形
    plt.show()