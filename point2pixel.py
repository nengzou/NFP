import torch
def re_projection(pha, fringe_b, config):
    """
    该函数演示了：
    1) 从相位图 pha_x 反算出 3D 点云 xyz_points（camera & projector 立体标定/三角测量）
    2) 在世界坐标系下做投影仪条纹采样 -> 得到 colors
    3) 做 Blinn-Phong 着色 -> 得到 shading_map
    4) 将最终“带条纹+明暗”的表面颜色，从世界坐标系再投影回相机视图
    """
    pha_x = torch.from_numpy(pha).to(device).float()
    fringe_b = torch.from_numpy(fringe_b).to(device).float()
    H, W = pha_x.shape
    up_test_obj = pha_x * 2 * torch.pi
    x_p = pha_x * 1920

    valid_mask = (up_test_obj != 0).float()

    # =============== camera & projector 立体标定：三角测量出世界坐标 ===============
    # 创建像素坐标网格
    vc_grid, uc_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    uc = uc_grid.float()  # (H, W)
    vc = vc_grid.float()  # (H, W)

    # 将无效像素的坐标置零，以避免影响计算
    uc = uc.float() * valid_mask
    vc = vc.float() * valid_mask
    up = x_p * valid_mask

    # 获取投影仪参数（如果存在扰动则使用，否则用默认）
    proj_R = config.get('proj_R', Rc_p)
    proj_T = config.get('proj_T', Tc_p)
    # 重新计算投影矩阵Ap
    Ap = Kp @ torch.hstack((proj_R, proj_T)).to(device)  # (3x4)

    # 构建批量矩阵 A 和向量 b
    A = torch.zeros((H, W, 3, 3), dtype=Ac.dtype, device=device)

    # 构建 A 的各个元素
    A[:, :, 0, 0] = Ac[0, 0] - Ac[2, 0] * uc
    A[:, :, 0, 1] = Ac[0, 1] - Ac[2, 1] * uc
    A[:, :, 0, 2] = Ac[0, 2] - Ac[2, 2] * uc
    A[:, :, 1, 0] = Ac[1, 0] - Ac[2, 0] * vc
    A[:, :, 1, 1] = Ac[1, 1] - Ac[2, 1] * vc
    A[:, :, 1, 2] = Ac[1, 2] - Ac[2, 2] * vc
    A[:, :, 2, 0] = Ap[0, 0] - Ap[2, 0] * up
    A[:, :, 2, 1] = Ap[0, 1] - Ap[2, 1] * up
    A[:, :, 2, 2] = Ap[0, 2] - Ap[2, 2] * up

    # 构建向量 b
    b_vec = torch.zeros((H, W, 3), dtype=Ac.dtype, device=device)
    b_vec[:, :, 0] = Ac[2, 3] * uc - Ac[0, 3]
    b_vec[:, :, 1] = Ac[2, 3] * vc - Ac[1, 3]
    b_vec[:, :, 2] = Ap[2, 3] * up - Ap[0, 3]

    # 添加一个小的正则项以确保矩阵可逆
    epsilon = 1e-6
    A_reg = A + epsilon * torch.eye(3, device=device).view(1, 1, 3, 3)

    # 使用可微的线性求解器
    A_flat = A_reg.view(-1, 3, 3)
    b_flat = b_vec.view(-1, 3, 1)
    xyz_w_flat = torch.linalg.solve(A_flat, b_flat).view(H, W, 3)

    # 提取三维坐标
    xws = xyz_w_flat[:, :, 0] * valid_mask
    yws = xyz_w_flat[:, :, 1] * valid_mask
    zws = xyz_w_flat[:, :, 2] * valid_mask

    # 将点云从世界坐标系转换到投影仪坐标系
    xyz_points = torch.stack((xws, yws, zws), dim=2)  # (H, W, 3)
    # cam_center 的形状为 (3,)，表示相机光心在三维空间中的位置
    cam_center = cam_center_world.squeeze()
    # 计算每个点到相机中心的向量
    to_cam_vector = xyz_points - cam_center
    # 沿视线方向缩放距离（distance_factor < 1表示靠近）
    adjusted_points = cam_center + to_cam_vector * config['distance_factor']
    xyz_points = adjusted_points

    # =============== 从世界坐标系 -> 投影仪坐标系，采样条纹颜色 ============
    xyz_points_flat = xyz_points.view(-1, 3)

    # 使用扰动后的外参进行坐标转换
    vert_proj = (proj_R @ xyz_points_flat.t() + proj_T).t()
    vert_proj = vert_proj.view(H, W, 3)
    x_proj = vert_proj[:, :, 0]
    y_proj = vert_proj[:, :, 1]
    z_proj = vert_proj[:, :, 2]

    # 归一化平面坐标
    xn_p = x_proj / z_proj
    yn_p = y_proj / z_proj
    # k1_p = Rd_p[0][0] # 径向畸变参数
    # k2_p = Rd_p[0][1]
    # p1_p = Td_p[0][0]
    # p2_p = Td_p[0][1]

    r2_p = xn_p * xn_p + yn_p * yn_p
    # 用 2 项径向畸变
    # radial_p = 1 + k1_p * r2_p + k2_p * (r2_p ** 2)
    # x_dist_p = xn_p * radial_p + 2 * p1_p * xn_p * yn_p + p2_p * (r2_p + 2 * xn_p * xn_p)
    # y_dist_p = yn_p * radial_p + p1_p * (r2_p + 2 * yn_p * yn_p) + 2 * p2_p * xn_p * yn_p

    # 最终投影仪像素坐标
    up = fx_p * xn_p + cx_p
    vp = fy_p * yn_p + cy_p

    # 使用双线性插值获取颜色  将坐标归一化到 [-1, 1]
    H_prj, W_prj = fringe_b.shape
    up_norm = (up / (W_prj - 1)) * 2 - 1
    vp_norm = (vp / (H_prj - 1)) * 2 - 1

    grid_proj = torch.stack((up_norm, vp_norm), dim=2)  # (H, W, 2)
    grid_proj = grid_proj.clamp(-1, 1)  # 防止采样越界

    fringe_b_tensor = fringe_b.to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, H_prj, W_prj)
    colors = F.grid_sample(
        fringe_b_tensor,
        grid_proj.unsqueeze(0),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    colors = colors.squeeze(0).permute(1, 2, 0)  # (H, W, 1)
    colors = colors * valid_mask.unsqueeze(2)  # 无效像素颜色设为零

    # =============== 计算法线 + Blinn-Phong 着色 ===============
    normals_map = estimate_normals(xyz_points)

    shading_map = blinn_phong_shading_with_brdf(
        xyz_points=xyz_points,
        normals_map=normals_map,
        valid_mask=valid_mask,
        cam_pos_world=cam_center_world.squeeze(),  # (3,)
        proj_stripe_map=colors.squeeze(),  # 投影仪条纹
        config=config
    )

    # 真实物理中，相机看到的纹理 = 投影仪条纹 * (物体 BRDF 效应)
    surface_color = colors.squeeze() * shading_map  # (H,W)

    # =============== 从世界坐标系 -> 相机视图，做“Z-buffer”渲染 ===============
    # 将点云从模型坐标系转换到相机坐标系
    vert_cam = (Rc_1 @ xyz_points_flat.t() + Tc_1).t()
    vert_cam = vert_cam.view(H, W, 3)
    x_cam = vert_cam[:, :, 0]
    y_cam = vert_cam[:, :, 1]
    z_cam = vert_cam[:, :, 2]

    # 归一化平面
    xn_c = x_cam / (z_cam + 1e-8)
    yn_c = y_cam / (z_cam + 1e-8)
    # 对相机应用径向 & 切向畸变
    # k1_c = Rd_c[0][0]
    # k2_c = Rd_c[0][1]
    # k3_c = Rd_c[0][2]
    # p1_c = Td_c[0][0]
    # p2_c = Td_c[0][1]
    # r2_c = xn_c * xn_c + yn_c * yn_c
    # radial_c = 1 + k1_c * r2_c + k2_c * (r2_c ** 2) + k3_c * (r2_c ** 3)
    # x_dist_c = xn_c * radial_c + 2 * p1_c * xn_c * yn_c + p2_c * (r2_c + 2 * xn_c * xn_c)
    # y_dist_c = yn_c * radial_c + p1_c * (r2_c + 2 * yn_c * yn_c) + 2 * p2_c * xn_c * yn_c

    u_c = fx_c * xn_c + cx_c
    v_c = fy_c * yn_c + cy_c

    # # 获取深度图：平移将最小深度值设置为 0
    # depth = z_cam * valid_mask  # 只考虑物体区域
    # # 计算物体区域的最小深度值，并平移到 0
    # min_depth = depth[depth > 0].min()  # 只考虑非零深度区域
    # depth = depth - min_depth  # 平移物体区域的深度到零
    # # 输出 u, v, depth 图
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(u_c.squeeze().detach().cpu().numpy(), v_c.squeeze().detach().cpu().numpy(),
    #            depth.squeeze().detach().cpu().numpy(), c=depth.squeeze().detach().cpu().numpy(), cmap='viridis',
    #            marker='o', s=5)
    # ax.set_xlabel('u (Pixel X)')
    # ax.set_ylabel('v (Pixel Y)')
    # ax.set_zlabel('Depth')
    # ax.set_title('Perspective Projection with Depth')
    # plt.show()

    # 简易 Z-buffer: 前向映射(点云 -> 像素) 这里做一个最近邻插值，把 (i,j) 的颜色放到 (round(u_c), round(v_c)).
    H_cam, W_cam = H, W  # 假设相机分辨率与原图一致，如有不同需改成真实分辨率
    final_image_cam = torch.zeros((H_cam, W_cam), device=device)
    final_depth_cam = torch.ones((H_cam, W_cam), device=device) * 1e8

    surface_color_flat = surface_color.view(-1)  # 注意这里，最终要绘制的是“带条纹的表面颜色”

    # 扁平化加速
    z_cam_flat = z_cam.view(-1)
    u_c_flat = u_c.view(-1)
    v_c_flat = v_c.view(-1)

    # 转为整型索引
    u_c_floor = torch.floor(u_c_flat + 0.5).long()
    v_c_floor = torch.floor(v_c_flat + 0.5).long()

    # clamp 防止越界
    u_c_floor = u_c_floor.clamp(0, W_cam - 1)
    v_c_floor = v_c_floor.clamp(0, H_cam - 1)

    # 做一个简单的 Z-Buffer
    idx = torch.arange(H * W, device=device)
    z_val = z_cam_flat
    old_depth = final_depth_cam[v_c_floor, u_c_floor]
    update_mask = z_val < old_depth

    # 更新深度
    final_depth_cam[v_c_floor[update_mask], u_c_floor[update_mask]] = z_val[update_mask]
    # 更新颜色
    final_image_cam_flat = final_image_cam.view(-1)
    final_image_cam_flat[update_mask] = surface_color_flat[update_mask]
    final_image_cam = final_image_cam_flat.view(H_cam, W_cam)

    final_image_cam = final_image_cam.squeeze().detach().cpu().numpy() * 255

    # plt.figure(figsize=(20, 20))
    # plt.imshow(final_image_cam, cmap='gray', vmin=0, vmax=255)
    # plt.title('blended')
    # plt.axis('off')
    # plt.show()

    return final_image_cam



def point_cloud_to_depth_map(xyz_points):
    """将三维点云转换为深度图"""

    points_list = xyz_points.reshape(-1, 3)
    valid_mask = ~np.isnan(points_list).any(axis=1)
    xyz_points = points_list[valid_mask]

    fx, fy = Kc[0, 0].cpu().numpy(), Kc[1, 1].cpu().numpy()
    cx, cy = Kc[0, 2].cpu().numpy(), Kc[1, 2].cpu().numpy()

    points_homogeneous = np.hstack((xyz_points, np.ones((xyz_points.shape[0], 1))))
    Rt = np.hstack((Rc_1.cpu().numpy(), Tc_1.cpu().numpy()))
    points_camera = (Rt @ points_homogeneous.T)
    z = points_camera[2, :]
    x = points_camera[0, :] / z
    y = points_camera[1, :] / z

    u = fx * x + cx
    v = fy * y + cy
    u_int = np.round(u).astype(int)
    v_int = np.round(v).astype(int)
    height, width = 1024, 1408
    # 筛选有效坐标
    valid = (u_int >= 0) & (u_int < width) & (v_int >= 0) & (v_int < height)
    valid_u = u_int[valid]
    valid_v = v_int[valid]
    valid_z = z[valid]

    # 用np.minimum.at直接更新深度图
    depth_map = np.full((height, width), np.inf, dtype=np.float32)
    np.minimum.at(depth_map, (valid_v, valid_u), valid_z)
    depth_map[depth_map == np.inf] = 0  # 将无效值设为0

    return depth_map