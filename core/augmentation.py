# 全局/局部 3D 增强及 2D/3D 联合增强
import numpy as np
import open3d as o3d
import copy


# ==========================================
# 核心增强函数 1：全局旋转 (绕 Z 轴)
# ==========================================
def rotate_scene(points, boxes, angle_deg):
    """
    points: (N, 3) 点云
    boxes: (M, 7) 格式为 [x, y, z, l, w, h, yaw]
    """
    points_aug = copy.deepcopy(points)
    boxes_aug = copy.deepcopy(boxes)

    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    # 旋转矩阵 R_z
    R_z = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

    # 1. 旋转点云
    points_aug[:, :3] = np.dot(points_aug[:, :3], R_z.T)

    if len(boxes_aug) > 0:
        # 2. 旋转框的中心点 (x, y, z)
        boxes_aug[:, :3] = np.dot(boxes_aug[:, :3], R_z.T)
        # 3. 修改朝向角 (Yaw) 并归一化到 [-pi, pi]
        boxes_aug[:, 6] = boxes_aug[:, 6] + theta
        boxes_aug[:, 6] = (boxes_aug[:, 6] + np.pi) % (2 * np.pi) - np.pi

    return points_aug, boxes_aug


# ==========================================
# 核心增强函数 2：全局缩放
# ==========================================
def scale_scene(points, boxes, scale_factor):
    points_aug = copy.deepcopy(points)
    boxes_aug = copy.deepcopy(boxes)

    # 1. 缩放点云
    points_aug[:, :3] = points_aug[:, :3] * scale_factor

    if len(boxes_aug) > 0:
        # 2. 缩放框中心点
        boxes_aug[:, :3] = boxes_aug[:, :3] * scale_factor
        # 3. 缩放长宽高 (l, w, h)
        boxes_aug[:, 3:6] = boxes_aug[:, 3:6] * scale_factor

    return points_aug, boxes_aug


# ==========================================
# 核心增强函数 3：沿 X 轴翻转 (左右镜像)
# ==========================================
def flip_scene_x(points, boxes):
    points_aug = copy.deepcopy(points)
    boxes_aug = copy.deepcopy(boxes)

    # 1. 翻转点云 X 轴
    points_aug[:, 0] = -points_aug[:, 0]

    if len(boxes_aug) > 0:
        # 2. 翻转框中心 X 轴
        boxes_aug[:, 0] = -boxes_aug[:, 0]
        # 3. 翻转朝向角 (Yaw) ： 镜像后车头朝向反了
        boxes_aug[:, 6] = np.pi - boxes_aug[:, 6]
        boxes_aug[:, 6] = (boxes_aug[:, 6] + np.pi) % (2 * np.pi) - np.pi

    return points_aug, boxes_aug


# ==========================================
# 辅助可视化函数：将 3D 框转换成 Open3D 对象
# ==========================================
def boxes_to_open3d(boxes, color=(1, 0, 0)):
    lines = []
    # 3D 框的 8 个顶点
    for box in boxes:
        x, y, z, l, w, h, yaw = box
        # 以中心点为基准，生成 8 个角点
        corners = np.array([
            [-l / 2, -w / 2, -h / 2], [l / 2, -w / 2, -h / 2], [l / 2, w / 2, -h / 2], [-l / 2, w / 2, -h / 2],
            [-l / 2, -w / 2, h / 2], [l / 2, -w / 2, h / 2], [l / 2, w / 2, h / 2], [-l / 2, w / 2, h / 2]
        ])
        # 绕 Z 轴旋转 Yaw
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        corners = np.dot(corners, R.T)
        # 平移到中心点
        corners += [x, y, z]

        # 定义 12 条边
        edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        for edge in edges:
            lines.append([corners[edge[0]], corners[edge[1]]])

    if not lines: return None
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.vstack([l[0] for l in lines] + [l[1] for l in lines]))
    # 为了避免重复点，简单构建索引
    indices = [[2 * i, 2 * i + 1] for i in range(len(lines))]
    line_set.lines = o3d.utility.Vector2iVector(indices)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(indices))])
    return line_set


# ==========================================
# 测试主函数
# ==========================================
if __name__ == "__main__":
    # 1. 伪造一帧数据 (一辆车朝向正前方 X 轴)
    # points: N x 3
    raw_points = np.random.uniform(-10, 10, (1000, 3))
    # box: [x, y, z, l, w, h, yaw] -> 中心在 (15, 0, -1)，朝向 0 度
    raw_boxes = np.array([[15.0, 0.0, -1.0, 4.5, 1.8, 1.5, 0.0]])

    # 2. 执行增强 (旋转 45 度)
    aug_points_rot, aug_boxes_rot = rotate_scene(raw_points, raw_boxes, angle_deg=45)

    # 3. 执行增强 (缩放 1.2 倍)
    aug_points_scale, aug_boxes_scale = scale_scene(raw_points, raw_boxes, scale_factor=1.2)

    # 4. 执行增强 (沿 X 翻转)
    aug_points_flip, aug_boxes_flip = flip_scene_x(raw_points, raw_boxes)

    # 5. 构建 Open3D 可视化对象
    pcd_raw = o3d.geometry.PointCloud()
    pcd_raw.points = o3d.utility.Vector3dVector(raw_points)
    pcd_raw.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色原始点

    pcd_rot = o3d.geometry.PointCloud()
    pcd_rot.points = o3d.utility.Vector3dVector(aug_points_rot)
    pcd_rot.paint_uniform_color([1, 0, 0])  # 红色旋转后点

    # 画出框线
    box_raw_o3d = boxes_to_open3d(raw_boxes, color=(0, 1, 0))  # 绿框（原始）
    box_rot_o3d = boxes_to_open3d(aug_boxes_rot, color=(0, 0, 1))  # 蓝框（旋转后）

    print(f"原始框 Yaw: {raw_boxes[0, 6]:.2f} rad")
    print(f"旋转后框 Yaw: {aug_boxes_rot[0, 6]:.2f} rad (增加了 0.785 rad = 45度)")

    # 可视化
    o3d.visualization.draw_geometries(
        [pcd_raw, box_raw_o3d, pcd_rot, box_rot_o3d],
        window_name="Global Augmentation (Gray/Green=Raw, Red/Blue=Rotated 45°)"
    )



import numpy as np
import open3d as o3d
import copy
import random
from shapely.geometry import Polygon


# ==========================================
# 1. 辅助函数：获取 3D 框的 BEV 俯视多边形
# ==========================================
def get_bev_polygon(box):
    x, y, z, l, w, h, yaw = box
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    corners = np.array([
        [x + l / 2 * cos_y - w / 2 * sin_y, y + l / 2 * sin_y + w / 2 * cos_y],
        [x - l / 2 * cos_y - w / 2 * sin_y, y - l / 2 * sin_y + w / 2 * cos_y],
        [x - l / 2 * cos_y + w / 2 * sin_y, y - l / 2 * sin_y - w / 2 * cos_y],
        [x + l / 2 * cos_y + w / 2 * sin_y, y + l / 2 * sin_y - w / 2 * cos_y]
    ])
    return Polygon(corners)


# ==========================================
# 2. 核心函数：从源点云中抠出目标
# ==========================================
def extract_object(points, box):
    """根据 3D 框，裁剪出框内的点云"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 构建 OBB 包围盒
    x, y, z, l, w, h, yaw = box
    R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))
    obb = o3d.geometry.OrientedBoundingBox([x, y, z], R, [l, w, h])

    # 裁剪
    cropped_pcd = pcd.crop(obb)
    return np.asarray(cropped_pcd.points)


# ==========================================
# 3. 核心函数：将目标贴入新场景
# ==========================================
def paste_object(target_points, target_boxes, obj_points, obj_box):
    """将抠出的点云粘贴到目标场景，处理防碰撞和贴地"""
    best_box = copy.deepcopy(obj_box)

    # 尝试随机寻找 10 个落点
    for _ in range(10):
        # a. 在目标场景前方随机生成一个 XY 坐标 (避开自车)
        rand_x = random.uniform(10, 40)
        rand_y = random.uniform(-10, 10)
        rand_yaw = random.uniform(-np.pi, np.pi)

        # b. 临时构建新框的中心位置
        best_box[0], best_box[1], best_box[6] = rand_x, rand_y, rand_yaw

        # c. 碰撞检测（防穿模）
        new_poly = get_bev_polygon(best_box)
        collision = False
        for existing_box in target_boxes:
            existing_poly = get_bev_polygon(existing_box)
            if new_poly.intersection(existing_poly).area > 0:
                collision = True
                break
        if collision:
            continue  # 发生碰撞，重试

        # d. 地面吸附（防悬空）
        # 寻找落点 XY 附近的目标场景点云（半径 2 米内）
        distances = np.linalg.norm(target_points[:, :2] - np.array([rand_x, rand_y]), axis=1)
        nearby_points = target_points[distances < 2.0]

        if len(nearby_points) < 10:
            continue  # 落点附近没点云（可能是树丛或盲区），换个地方

        # 计算地面高度 (取附近点云 Z 轴 5% 分位数，避免踩到噪点)
        ground_z = np.percentile(nearby_points[:, 2], 5)

        # e. 把框的 Z 坐标对齐到新地面
        # 源框的底部 Z 是 z - h/2，我们要让底部和新地面齐平
        bottom_z = ground_z
        best_box[2] = bottom_z + best_box[5] / 2.0

        # f. 平移被抠出来的点云
        # 目标坐标 - 源点云的中心坐标
        delta_x = best_box[0] - obj_box[0]
        delta_y = best_box[1] - obj_box[1]
        delta_z = best_box[2] - obj_box[2]

        # 注意：还要处理 Yaw 的旋转差异 (此处简化，直接应用新 Yaw)
        # 工业级会用一个完整的 4x4 矩阵进行变换
        pasted_points = obj_points + np.array([delta_x, delta_y, delta_z])

        # 融合成功！返回新的点云和框
        return np.vstack((target_points, pasted_points)), np.vstack((target_boxes, best_box))

    print("⚠️ 找不到合适的落点，放弃粘贴")
    return target_points, target_boxes


# ==========================================
# 4. 测试与可视化
# ==========================================
if __name__ == "__main__":
    # 1. 伪造数据：背景场景 (一片平地) 和 已有目标
    bg_points = np.random.uniform(-20, 50, (10000, 3))
    bg_points[:, 2] = np.random.normal(-1.7, 0.05, 10000)  # Z = -1.7 的地面
    bg_boxes = np.array([[20.0, 0.0, -0.95, 4.5, 1.8, 1.5, 0.0]])  # 场景里本来就有一辆车

    # 2. 伪造数据：要抠出来的目标 (一辆卡车)
    truck_box = np.array([10.0, 10.0, -1.0, 10.0, 2.5, 3.0, 0.5])
    truck_points = np.random.normal([10.0, 10.0, -1.0], [2.0, 0.5, 1.0], (500, 3))

    # 3. 执行 Copy-Paste
    print("开始执行 Copy-Paste...")
    merged_points, merged_boxes = paste_object(bg_points, bg_boxes, truck_points, truck_box)
    print(f"原场景点数: {len(bg_points)}, 粘贴后点数: {len(merged_points)}")

    # 4. 可视化结果
    pcd_bg = o3d.geometry.PointCloud()
    pcd_bg.points = o3d.utility.Vector3dVector(bg_points)
    pcd_bg.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色背景

    pcd_truck = o3d.geometry.PointCloud()
    # 找出新贴进来的点（即最后 500 个点）
    pcd_truck.points = o3d.utility.Vector3dVector(merged_points[-500:])
    pcd_truck.paint_uniform_color([1, 0, 0])  # 红色目标

    # ... 前面的代码保持不变 ...
    print("红点代表新贴进来的目标，正在使用代码强制拉取俯视图...")

    # 🌟 终极绝招：使用 Visualizer 底层接口，直接控制相机
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Copy-Paste (Forced Top View)", width=1600, height=900)

    vis.add_geometry(pcd_bg)
    vis.add_geometry(pcd_truck)

    # 获取相机控制器
    view_control = vis.get_view_control()

    # 1. 设置为正上方俯视 (Top View)
    # 视线方向：沿 Z 轴垂直向下看
    view_control.set_front([0, 0, -1])
    # 屏幕上方向：指向 Y 轴
    view_control.set_up([0, 1, 0])
    # 视线焦点：大概位于前景区域 (X=15, Y=0, Z=-1)
    view_control.set_lookat([15, 0, -1])
    # 缩放：拉到能看全场景的程度
    view_control.set_zoom(0.4)

    # 运行窗口
    vis.run()
    vis.destroy_window()


import cv2
import numpy as np
from pathlib import Path

# 1. 路径配置 (请换成你自己的 KITTI 路径)
BASE_DIR = Path(r"E:\AI_job\AI_Perception\KITTI\data\kitti\training")
IMG_PATH = BASE_DIR / "image_2" / "000001.png"
LIDAR_PATH = BASE_DIR / "velodyne" / "000001.bin"
CALIB_PATH = BASE_DIR / "calib" / "000001.txt"


# 2. 解析标定文件
def read_calib(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    calib = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':')
            calib[key] = np.array([float(x) for x in value.strip().split()])

    P2 = calib['P2'].reshape(3, 4)
    R0 = np.eye(4)
    R0[:3, :3] = calib['R0_rect'].reshape(3, 3)
    V2C = np.eye(4)
    V2C[:3, :4] = calib['Tr_velo_to_cam'].reshape(3, 4)
    return P2, R0, V2C


P2, R0, V2C = read_calib(CALIB_PATH)


# 3. 三维点投影到二维像素的函数
def project_3d_to_2d(points_3d, P2, R0, V2C):
    # 转为齐次坐标
    pts_hom = np.hstack((points_3d, np.ones((len(points_3d), 1))))
    # 雷达 -> 相机 -> 图像
    pts_cam = (V2C @ pts_hom.T).T
    pts_rect = (R0 @ pts_cam.T).T
    pts_img_hom = (P2 @ pts_rect.T).T
    # 归一化
    pts_img = pts_img_hom[:, :2] / pts_img_hom[:, 2:]
    return pts_img


# 4. 联合增强：水平翻转
def joint_horizontal_flip(img, img_width, boxes_3d):
    """
    将图像、3D框进行同步水平翻转
    """
    # a. 图像翻转
    img_flipped = cv2.flip(img, 1)

    # b. 3D框数学变换
    boxes_flipped = boxes_3d.copy()
    if len(boxes_flipped) > 0:
        # 点云在雷达系下的 Y 轴是向左的。水平翻转意味着 Y 取反
        boxes_flipped[:, 1] = -boxes_flipped[:, 1]
        # 偏航角 (Yaw) 镜像：yaw = -yaw
        boxes_flipped[:, 6] = -boxes_flipped[:, 6]

    return img_flipped, boxes_flipped


# 5. 画 3D 框的 8 个顶点
def get_box_corners_3d(box):
    x, y, z, l, w, h, yaw = box
    # 雷达系下底面四个角
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]

    corners = np.vstack([x_corners, y_corners, z_corners]).T
    # 旋转
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = np.dot(corners, R.T)
    # 平移
    corners += [x, y, z]
    return corners


# ==========================================
# 主测试
# ==========================================
if __name__ == "__main__":
    # 加载原始图像
    img = cv2.imread(str(IMG_PATH))
    img_height, img_width = img.shape[:2]

    # 伪造一个 3D 框 (假设前方 15 米处有一辆车，朝向正前方)
    # [x, y, z, l, w, h, yaw]
    fake_box = np.array([[15.0, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0]])

    # 1. 原本未翻转时，计算 2D 框
    corners_3d = get_box_corners_3d(fake_box[0])
    pts_2d = project_3d_to_2d(corners_3d, P2, R0, V2C)
    # 在图像上画出原始 2D 框 (绿色)
    img_draw = img.copy()
    x_min, y_min = np.min(pts_2d, axis=0).astype(int)
    x_max, y_max = np.max(pts_2d, axis=0).astype(int)
    cv2.rectangle(img_draw, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # 2. 执行联合增强 (翻转)
    img_flipped, boxes_flipped = joint_horizontal_flip(img, img_width, fake_box)

    # 3. 在翻转后的图像上，重新投影翻转后的 3D 框
    corners_3d_flipped = get_box_corners_3d(boxes_flipped[0])
    pts_2d_flipped = project_3d_to_2d(corners_3d_flipped, P2, R0, V2C)

    # 画出翻转后的 2D 框 (蓝色)
    img_flipped_draw = img_flipped.copy()
    x_min_f, y_min_f = np.min(pts_2d_flipped, axis=0).astype(int)
    x_max_f, y_max_f = np.max(pts_2d_flipped, axis=0).astype(int)
    cv2.rectangle(img_flipped_draw, (x_min_f, y_min_f), (x_max_f, y_max_f), (255, 0, 0), 2)

    # 4. 拼接显示结果
    # 将两张图缩小一半，左右拼接对比
    h, w = img_height // 2, img_width // 2
    img_draw_small = cv2.resize(img_draw, (w, h))
    img_flipped_small = cv2.resize(img_flipped_draw, (w, h))
    canvas = np.hstack((img_draw_small, img_flipped_small))

    import matplotlib.pyplot as plt

    print("正在使用 Matplotlib 弹出窗口...")
    # OpenCV 是 BGR 格式，Matplotlib 是 RGB 格式，需要转换一下颜色
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 6))
    plt.imshow(canvas_rgb)
    plt.title("Left: Original Projection | Right: Joint Augmentation (Flipped)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()