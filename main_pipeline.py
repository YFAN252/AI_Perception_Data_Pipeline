import os
import time
import sqlite3
import yaml
import logging
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import open3d as o3d

from core.preprocess import process_pointcloud
from core.auto_qa import AutoQA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_kitti_labels(label_path, calib_path):
    boxes = []
    print(f"🔍 尝试读取标签: {label_path}")

    if not os.path.exists(label_path) or not os.path.exists(calib_path):
        print(f"❌ 找不到文件！label存在: {os.path.exists(label_path)}, calib存在: {os.path.exists(calib_path)}")
        return np.zeros((0, 8))

    # 1. 解析标定文件，获取 相机 -> 雷达 的转换矩阵
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    calib_dict = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':')
            calib_dict[key] = np.array([float(x) for x in value.strip().split()])

    V2C = np.eye(4)
    V2C[:3, :4] = calib_dict['Tr_velo_to_cam'].reshape(3, 4)
    C2V = np.linalg.inv(V2C)  # 相机到雷达的逆矩阵

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            obj_type = parts[0]
            if obj_type not in ['Car', 'Truck', 'Pedestrian']: continue

            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            x_cam, y_cam, z_cam = float(parts[11]), float(parts[12]), float(parts[13])
            ry_cam = float(parts[14])

            # 2. KITTI 给的是底部中心，转换到几何中心
            center_cam = np.array([x_cam, y_cam - h / 2, z_cam, 1.0])
            # 3. 转到雷达系
            center_velo = C2V @ center_cam
            yaw_velo = -ry_cam - np.pi / 2

            # # ==========================================
            # # 🚨 真正生效的投毒测试代码
            # if len(boxes) == 0:
            #     # 投毒1：把第一辆车的长度改成 20 米
            #     l = 20.0
            # elif len(boxes) == 1:
            #     # 投毒2：让第二辆车的中心和第一辆车完全重合，制造 100% 穿模
            #     center_velo[0] = boxes[0][0]
            #     center_velo[1] = boxes[0][1]
            #     center_velo[2] = boxes[0][2]
            #     yaw_velo = boxes[0][6]
            # # ==========================================

            boxes.append([center_velo[0], center_velo[1], center_velo[2], l, w, h, yaw_velo, obj_type])

    print(f"【调试】读取标签: {os.path.basename(label_path)}，共提取到 {len(boxes)} 个目标框。")

    return np.array(boxes, dtype=object) if boxes else np.zeros((0, 8))


def process_single_frame(bin_path, label_dir, voxel_size):
    frame_id = os.path.basename(bin_path).replace(".bin", "")
    label_path = os.path.join(label_dir, f"{frame_id}.txt")
    calib_path = os.path.join(os.path.dirname(label_dir), "calib", f"{frame_id}.txt")
    start_time = time.time()

    try:
        # 1. 预处理：裁剪、降采样、去地面
        non_ground_pcd, raw_count, clean_count = process_pointcloud(bin_path, voxel_size)

        # 2. 读取 3D 框
        boxes = read_kitti_labels(label_path, calib_path)

        # 3. 执行 Auto-QA (尺寸、穿模、包含率)
        qa = AutoQA()
        is_pass, qa_msg = qa.run_all(non_ground_pcd, boxes)

        if not is_pass:
            qa_status = f"QA_FAILED: {qa_msg}"
        elif clean_count < 500:
            qa_status = f"QA_FAILED: 点云过少({clean_count})"
        else:
            qa_status = "PASSED"

    except Exception as e:
        raw_count, clean_count = 0, 0
        qa_status = f"ERROR: {str(e)}"

    elapsed = time.time() - start_time
    return (frame_id, raw_count, clean_count, qa_status, elapsed)


def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_dir = config["data_dir"]
    # 假设 label 在 data_dir 的上一级的 label_2 文件夹
    label_dir = os.path.join(os.path.dirname(data_dir), "label_2")
    db_path = config["db_path"]
    workers = config["workers"]
    voxel_size = config["voxel_size"]

    bin_files = [str(p) for p in Path(data_dir).glob("*.bin")]
    logger.info(f"🚀 Pipeline 启动，共发现 {len(bin_files)} 帧数据，启动 {workers} 个进程...")

    # 初始化数据库
    conn = sqlite3.connect(db_path)
    conn.execute(
        'CREATE TABLE IF NOT EXISTS frames (frame_id TEXT PRIMARY KEY, raw_points INT, clean_points INT, qa_status TEXT, process_time REAL)')
    conn.commit()
    conn.close()

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single_frame, f, label_dir, voxel_size): f for f in bin_files}
        for future in as_completed(futures):
            results.append(future.result())

    # 统一落库
    conn = sqlite3.connect(db_path)
    conn.executemany('INSERT OR REPLACE INTO frames VALUES (?, ?, ?, ?, ?)', results)
    conn.commit()
    conn.close()

    passed = sum(1 for r in results if r[3] == "PASSED")
    logger.info(f"✅ 处理完成！总计: {len(results)} 帧，通过: {passed} 帧，被拦截: {len(results) - passed} 帧。")
    logger.info(f"请使用 DBeaver 查看 {db_path} 中被拦截的脏数据原因！")


if __name__ == "__main__":
    main()