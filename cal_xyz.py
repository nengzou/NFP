import numpy as np
import cv2
import open3d as o3d

def construction(pha_x,cam_calib_result,prj_calib_result):
    # =============================================================================
    # 1. 加载校准数据
    # =============================================================================
    Kc = cam_calib_result['Kc']  # 相机内参
    Rc_1 = cam_calib_result['Rc_1'][0, :, :]
    Tc_1 = cam_calib_result['Tc_1'][0, :, :]
    Rc_1, _ = cv2.Rodrigues(Rc_1)
    Rc_1[:, [0, 1]]=Rc_1[:, [1, 0]]#第一列和第二列调换
    Rc_1[:, 2] = -Rc_1[:, 2]#第三列取负数
    # print(Kc, "\n", Tc_1, "\n", Rc_1, "\n")
    Ac = Kc @ np.hstack((Rc_1, Tc_1))
    # print(Ac.shape)

    Kp = prj_calib_result['Kp']  # 投影仪内参
    Rc_1 = prj_calib_result['Rc_1'][0, :, :]
    Tc_1 = prj_calib_result['Tc_1'][0, :, :]
    Rc_1, _ = cv2.Rodrigues(Rc_1)
    Rc_1[:, [0, 1]] = Rc_1[:, [1, 0]]  # 第一列和第二列调换
    Rc_1[:, 2] = -Rc_1[:, 2]  # 第三列取负数
    Ap = Kp @ np.hstack((Rc_1, Tc_1))
    # print(Kp, "\n", Tc_1, "\n", Rc_1)
    # =============================================================================
    # 2. 查看相位
    # =============================================================================
    # show_2Dnp(pha_x)

    up_test_obj = pha_x #* 2 * np.pi
    # print("up_test_obj min:", up_test_obj.min().item(), "up_test_obj max:", up_test_obj.max().item())
    x_p = pha_x #* self.pro_width
    # print("x_p min:", x_p.min().item(), "x_p max:", x_p.max().item())

    # =============================================================================
    # 3. 三维重建（矢量化实现）
    # =============================================================================
    # print("正在计算点云坐标并绘制点云...")

    # 1. 提取所有有效的像素点（up_test_obj != 0）
    valid_mask = up_test_obj != 0
    y_idxs, x_idxs = np.where(valid_mask)
    uc = x_idxs.astype(np.float32)
    vc = y_idxs.astype(np.float32)
    up = x_p[valid_mask].astype(np.float32)

    uc_num = len(uc)
    print(f"有效像素点数量: {uc_num}")

    if uc_num == 0:
        print("没有有效的像素点用于三维重建。")
        return

    # 2. 构建批量矩阵 A 和向量 b
    # A 的形状为 (uc_num, 3, 3)
    A = np.empty((uc_num, 3, 3), dtype=Ac.dtype)

    # 构建 A 的第一行
    A[:, 0, 0] = Ac[0, 0] - Ac[2, 0] * uc
    A[:, 0, 1] = Ac[0, 1] - Ac[2, 1] * uc
    A[:, 0, 2] = Ac[0, 2] - Ac[2, 2] * uc

    # 构建 A 的第二行
    A[:, 1, 0] = Ac[1, 0] - Ac[2, 0] * vc
    A[:, 1, 1] = Ac[1, 1] - Ac[2, 1] * vc
    A[:, 1, 2] = Ac[1, 2] - Ac[2, 2] * vc

    # 构建 A 的第三行
    A[:, 2, 0] = Ap[0, 0] - Ap[2, 0] * up
    A[:, 2, 1] = Ap[0, 1] - Ap[2, 1] * up
    A[:, 2, 2] = Ap[0, 2] - Ap[2, 2] * up

    # 构建向量 b，形状为 (N, 3)
    b = np.empty((uc_num, 3), dtype=Ac.dtype)
    b[:, 0] = Ac[2, 3] * uc - Ac[0, 3]
    b[:, 1] = Ac[2, 3] * vc - Ac[1, 3]
    b[:, 2] = Ap[2, 3] * up - Ap[0, 3]

    # 3. 计算矩阵 A 的行列式，以检测是否可逆
    detA = (
            A[:, 0, 0] * (A[:, 1, 1] * A[:, 2, 2] - A[:, 1, 2] * A[:, 2, 1]) -
            A[:, 0, 1] * (A[:, 1, 0] * A[:, 2, 2] - A[:, 1, 2] * A[:, 2, 0]) +
            A[:, 0, 2] * (A[:, 1, 0] * A[:, 2, 1] - A[:, 1, 1] * A[:, 2, 0])
    )

    # 定义一个阈值，用于判断矩阵是否可逆
    epsilon = 1e-6
    invertible_mask = np.abs(detA) > epsilon
    invertible_indices = np.where(invertible_mask)[0]
    non_invertible_count = uc_num - len(invertible_indices)
    if non_invertible_count > 0:
        print(f"跳过 {non_invertible_count} 个不可逆的像素点。")

    # 仅保留可逆的矩阵
    A_invertible = A[invertible_mask]
    b_invertible = b[invertible_mask]

    # 4. 批量求解线性方程组 A * xyz = b
    # 使用 numpy.linalg.solve 进行批量求解
    try:
        # xyz_w = np.linalg.solve(A_invertible, b_invertible)  # 形状 (M, 3)
        xyz_w = np.einsum('...ij,...j->...i', np.linalg.inv(A_invertible), b_invertible)
    except np.linalg.LinAlgError as e:
        print(f"批量求解线性方程组时出错: {e}")
        return

    # 5. 初始化三维坐标矩阵，并填入计算结果
    (height,width)=pha_x.shape
    xws = np.full((height, width), np.nan, dtype=np.float32)
    yws = np.full((height, width), np.nan, dtype=np.float32)
    zws = np.full((height, width), np.nan, dtype=np.float32)

    # 将计算得到的坐标赋值回对应的位置
    xws[y_idxs[invertible_mask], x_idxs[invertible_mask]] = xyz_w[:, 0]
    yws[y_idxs[invertible_mask], x_idxs[invertible_mask]] = xyz_w[:, 1]
    zws[y_idxs[invertible_mask], x_idxs[invertible_mask]] = xyz_w[:, 2]

    # =============================================================================
    # 4. 点云显示与保存
    # =============================================================================
    # 创建点云数据
    xyz_points = np.vstack((xws.flatten(), yws.flatten(), zws.flatten())).T
    # 移除包含 NaN 的点
    valid_points_mask = ~np.isnan(xyz_points).any(axis=1)
    xyz_points = xyz_points[valid_points_mask]

    if xyz_points.size == 0:
        print("没有有效的三维点用于生成点云。")
        return

    # show_3D(xyz_points)

    return xyz_points

def show_2Dnp(pha):
    # 以pha的行列号为x，y轴坐标，值为z值画三维图
    # 生成（x，y，z）坐标，用显示x，y，z的函数显示出来。
    # 获取二维数组的形状
    rows, cols = pha.shape

    # 创建网格坐标
    x = np.arange(cols)  # 列号作为 x 轴
    y = np.arange(rows)  # 行号作为 y 轴
    x, y = np.meshgrid(x, y)  # 生成网格坐标

    # 获取 z 值
    z = pha

    # 创建点云数据
    xyz_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    show_3D(xyz_points)

def show_3D(xyz):
    # 创建 Open3D 点云对象
    pt_cloud = o3d.geometry.PointCloud()
    pt_cloud.points = o3d.utility.Vector3dVector(xyz)
    # 可视化点云
    # 设置可视化参数
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pt_cloud)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1.0

    # 添加坐标轴
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0, origin=[0, 0, 0])
    vis.add_geometry(axis)

    vis.run()
    vis.destroy_window()


def save_ply(filename,points):
    # 保存成全精度的数据
    #如果用open3D保存数据只有3位小数
    # PLY 文件的头部
    header = """ply
    format ascii 1.0
    element vertex {num_points}
    property float x
    property float y
    property float z
    end_header
    """.format(num_points=len(points))

    try:
        with open(filename, 'w') as f:
            # 写入头部
            f.write(header)
            # 写入点云数据
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
        print(f"点云数据已成功保存为 {filename}")
    except Exception as e:
        print(f"保存点云数据失败：{e}")

def save_open3D_ply(filename,points):
    # 创建 Open3D 点云对象
    pt_cloud = o3d.geometry.PointCloud()
    pt_cloud.points = o3d.utility.Vector3dVector(points)

    # 定义保存路径和文件名
    output_file = filename
    # output_path = os.path.join(self.cloud_data_dir, output_file)

    # 保存点云为 PLY 格式
    try:
        # os.makedirs(self.cloud_data_dir, exist_ok=True)  # 创建多级目录，如果已存在则不会报错
        success = o3d.io.write_point_cloud(output_file, pt_cloud, write_ascii=True)
        if success:
            print(f"点云已成功保存到 {output_file}")
        else:
            print(f"保存点云失败，无法写入文件到 {output_file}")
    except Exception as e:
        print(f"保存点云时出错: {e}")


def cal_all_xyz(pha_x,cam_calib_result,prj_calib_result):
    # =============================================================================
    # 1. 加载校准数据
    # =============================================================================
    Kc = cam_calib_result['Kc']  # 相机内参
    Rc_1 = cam_calib_result['Rc_1'][0, :, :]
    Tc_1 = cam_calib_result['Tc_1'][0, :, :]
    Rc_1, _ = cv2.Rodrigues(Rc_1)
    Rc_1[:, [0, 1]]=Rc_1[:, [1, 0]]#第一列和第二列调换
    Rc_1[:, 2] = -Rc_1[:, 2]#第三列取负数
    # print(Kc, "\n", Tc_1, "\n", Rc_1, "\n")
    Ac = Kc @ np.hstack((Rc_1, Tc_1))
    # print(Ac.shape)

    Kp = prj_calib_result['Kp']  # 投影仪内参
    Rc_1 = prj_calib_result['Rc_1'][0, :, :]
    Tc_1 = prj_calib_result['Tc_1'][0, :, :]
    Rc_1, _ = cv2.Rodrigues(Rc_1)
    Rc_1[:, [0, 1]] = Rc_1[:, [1, 0]]  # 第一列和第二列调换
    Rc_1[:, 2] = -Rc_1[:, 2]  # 第三列取负数
    Ap = Kp @ np.hstack((Rc_1, Tc_1))
    # print(Kp, "\n", Tc_1, "\n", Rc_1)
    # =============================================================================
    # 2. 查看相位
    # =============================================================================
    # show_2Dnp(pha_x)

    up_test_obj = pha_x #* 2 * np.pi
    # print("up_test_obj min:", up_test_obj.min().item(), "up_test_obj max:", up_test_obj.max().item())
    x_p = pha_x #* self.pro_width
    # print("x_p min:", x_p.min().item(), "x_p max:", x_p.max().item())

    # =============================================================================
    # 3. 三维重建（矢量化实现）
    # =============================================================================
    # print("正在计算点云坐标并绘制点云...")

    # 1. 提取所有有效的像素点（up_test_obj != 0）
    valid_mask = up_test_obj != 0
    y_idxs, x_idxs = np.where(valid_mask)
    uc = x_idxs.astype(np.float32)
    vc = y_idxs.astype(np.float32)
    up = x_p[valid_mask].astype(np.float32)

    uc_num = len(uc)
    print(f"有效像素点数量: {uc_num}")

    if uc_num == 0:
        print("没有有效的像素点用于三维重建。")
        return

    # 2. 构建批量矩阵 A 和向量 b
    # A 的形状为 (uc_num, 3, 3)
    A = np.empty((uc_num, 3, 3), dtype=Ac.dtype)

    # 构建 A 的第一行
    A[:, 0, 0] = Ac[0, 0] - Ac[2, 0] * uc
    A[:, 0, 1] = Ac[0, 1] - Ac[2, 1] * uc
    A[:, 0, 2] = Ac[0, 2] - Ac[2, 2] * uc

    # 构建 A 的第二行
    A[:, 1, 0] = Ac[1, 0] - Ac[2, 0] * vc
    A[:, 1, 1] = Ac[1, 1] - Ac[2, 1] * vc
    A[:, 1, 2] = Ac[1, 2] - Ac[2, 2] * vc

    # 构建 A 的第三行
    A[:, 2, 0] = Ap[0, 0] - Ap[2, 0] * up
    A[:, 2, 1] = Ap[0, 1] - Ap[2, 1] * up
    A[:, 2, 2] = Ap[0, 2] - Ap[2, 2] * up

    # 构建向量 b，形状为 (N, 3)
    b = np.empty((uc_num, 3), dtype=Ac.dtype)
    b[:, 0] = Ac[2, 3] * uc - Ac[0, 3]
    b[:, 1] = Ac[2, 3] * vc - Ac[1, 3]
    b[:, 2] = Ap[2, 3] * up - Ap[0, 3]

    # 3. 计算矩阵 A 的行列式，以检测是否可逆
    detA = (
            A[:, 0, 0] * (A[:, 1, 1] * A[:, 2, 2] - A[:, 1, 2] * A[:, 2, 1]) -
            A[:, 0, 1] * (A[:, 1, 0] * A[:, 2, 2] - A[:, 1, 2] * A[:, 2, 0]) +
            A[:, 0, 2] * (A[:, 1, 0] * A[:, 2, 1] - A[:, 1, 1] * A[:, 2, 0])
    )

    # 定义一个阈值，用于判断矩阵是否可逆
    epsilon = 1e-6
    invertible_mask = np.abs(detA) > epsilon
    invertible_indices = np.where(invertible_mask)[0]
    non_invertible_count = uc_num - len(invertible_indices)
    if non_invertible_count > 0:
        print(f"跳过 {non_invertible_count} 个不可逆的像素点。")

    # 仅保留可逆的矩阵
    A_invertible = A[invertible_mask]
    b_invertible = b[invertible_mask]

    # 4. 批量求解线性方程组 A * xyz = b
    # 使用 numpy.linalg.solve 进行批量求解
    try:
        # xyz_w = np.linalg.solve(A_invertible, b_invertible)  # 形状 (M, 3)
        xyz_w = np.einsum('...ij,...j->...i', np.linalg.inv(A_invertible), b_invertible)
    except np.linalg.LinAlgError as e:
        print(f"批量求解线性方程组时出错: {e}")
        return

    # 5. 初始化三维坐标矩阵，并填入计算结果
    (height,width)=pha_x.shape
    xws = np.full((height, width), np.nan, dtype=np.float32)
    yws = np.full((height, width), np.nan, dtype=np.float32)
    zws = np.full((height, width), np.nan, dtype=np.float32)

    # 将计算得到的坐标赋值回对应的位置
    xws[y_idxs[invertible_mask], x_idxs[invertible_mask]] = xyz_w[:, 0]
    yws[y_idxs[invertible_mask], x_idxs[invertible_mask]] = xyz_w[:, 1]
    zws[y_idxs[invertible_mask], x_idxs[invertible_mask]] = xyz_w[:, 2]

    # =============================================================================
    # 4. 点云显示与保存
    # =============================================================================
    # 创建点云数据
    xyz_points = np.stack((xws,yws,zws),axis=-1)
    # xyz_points = np.vstack((xws.flatten(), yws.flatten(), zws.flatten())).T
    # 移除包含 NaN 的点
    # valid_points_mask = ~np.isnan(xyz_points).any(axis=1)
    # xyz_points = xyz_points[valid_points_mask]

    if xyz_points.size == 0:
        print("没有有效的三维点用于生成点云。")
        return

    # show_3D(xyz_points)

    return xyz_points
