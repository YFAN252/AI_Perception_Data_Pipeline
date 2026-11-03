import open3d as o3d
import numpy as np
import os

def process_pointcloud(bin_path, voxel_size=0.2):
    """读取点云 -> ROI裁剪 -> 降采样 -> 去地面。返回清洗后的点云和统计数据"""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    raw_count = len(points)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:, :3]))

    # 裁剪
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-20, -20, -3), max_bound=(50, 20, 3))
    pcd = pcd.crop(bbox)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # 去地面
    _, inliers = pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=100)
    non_ground_pcd = pcd.select_by_index(inliers, invert=True)

    return non_ground_pcd, raw_count, len(non_ground_pcd.points)