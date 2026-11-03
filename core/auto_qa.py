# 4维质检（尺寸、包含率、穿模、超速）拦截逻辑
'''
尺寸校验：查标注员有没有把一辆奔驰 Smart 画成 10 米长的重卡。

BEV 穿模检测：利用 Shapely 把 3D 框压扁成 2D 多边形计算交集。如果有两个框重叠，说明标注员画了“幽灵车”，或者把同一辆车标了两次。这会导致模型 NMS 崩溃。

点云包含率校验：计算该 3D 框裁剪出来的点云数量。如果 < 5，说明这是“空手画框”，彻底剔除。
'''
import numpy as np
import open3d as o3d
from shapely.geometry import Polygon


class AutoQA:
    def __init__(self):
        # 1. 物理尺寸限制阈值 (长 L, 宽 W, 高 H)
        self.size_limits = {
            "Car": {"L": (2.3, 6.0), "W": (1.2, 2.5), "H": (1.2, 2.0)},  # 下限放宽到 2.5
            "Truck": {"L": (4.0, 18.0), "W": (1.8, 3.2), "H": (2.5, 4.5)},
            "Pedestrian": {"L": (0.2, 1.2), "W": (0.3, 1.2), "H": (1.4, 2.2)}
        }

    def _get_bev_polygon(self, box):
        """将 3D 框投影到 BEV 平面，生成 Shapely 2D 多边形"""
        # 兼容可能带有 8 个元素（包含类型）的情况
        x, y, z, l, w, h, yaw = box[:7]
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        corners = np.array([
            [x + l / 2 * cos_y - w / 2 * sin_y, y + l / 2 * sin_y + w / 2 * cos_y],
            [x - l / 2 * cos_y - w / 2 * sin_y, y - l / 2 * sin_y + w / 2 * cos_y],
            [x - l / 2 * cos_y + w / 2 * sin_y, y - l / 2 * sin_y - w / 2 * cos_y],
            [x + l / 2 * cos_y + w / 2 * sin_y, y + l / 2 * sin_y - w / 2 * cos_y]
        ])
        return Polygon(corners)

    def check_dimensions(self, obj_type, l, w, h):
        """1. 物理尺寸校验"""
        if obj_type not in self.size_limits: return True, ""
        limits = self.size_limits[obj_type]
        if not (limits['L'][0] <= l <= limits['L'][1]): return False, f"{obj_type} 长度异常({l:.1f}m)"
        if not (limits['W'][0] <= w <= limits['W'][1]): return False, f"{obj_type} 宽度异常({w:.1f}m)"
        if not (limits['H'][0] <= h <= limits['H'][1]): return False, f"{obj_type} 高度异常({h:.1f}m)"
        return True, ""

    def check_bev_collision(self, boxes):
        """2. BEV 视角穿模检测 (加入容忍度)"""
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                poly1 = self._get_bev_polygon(boxes[i])
                poly2 = self._get_bev_polygon(boxes[j])
                intersection_area = poly1.intersection(poly2).area
                union_area = poly1.union(poly2).area

                if union_area > 0:
                    iou = intersection_area / union_area
                    # 只有 IoU 超过 20% 才算严重穿模，避免误杀拥堵路段
                    if iou > 0.2:
                        return False, f"目标 {i} 与 {j} 发生严重穿模 (IoU={iou:.2f})"
        return True, ""

    def check_point_containment(self, pcd, boxes, min_points=5):
        """3. 点云包含率校验 (防止空手画框)"""
        for i, box in enumerate(boxes):
            x, y, z, l, w, h, yaw = box[:7]
            R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))
            obb = o3d.geometry.OrientedBoundingBox([x, y, z], R, [l, w, h])
            cropped = pcd.crop(obb)
            if len(cropped.points) < min_points:
                return False, f"目标 {i} 框内仅 {len(cropped.points)} 个点 (疑似空手画框)"
        return True, ""

    def run_all(self, pcd, boxes):
        """执行全面质检，返回 (是否通过, 错误原因)"""
        # 1. 检查尺寸
        for box in boxes:
            is_valid, msg = self.check_dimensions(box[7], box[3], box[4], box[5])  # 假设 box[7] 是类别
            if not is_valid: return False, msg

        # 2. 检查穿模
        is_valid, msg = self.check_bev_collision(boxes)
        if not is_valid: return False, msg

        # 3. 检查包含率
        is_valid, msg = self.check_point_containment(pcd, boxes)
        if not is_valid: return False, msg

        return True, "PASSED"