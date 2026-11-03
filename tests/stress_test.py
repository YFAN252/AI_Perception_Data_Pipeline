import os
import time
import glob
import numpy as np
import open3d as o3d
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# 数据源目录（请确保里面有几十个 bin 文件）
SRC_DIR = r"E:\AI_job\AI_Perception\KITTI\data\kitti\training\velodyne"
# 输出目录
DST_DIR = r".\output_processed"
os.makedirs(DST_DIR, exist_ok=True)


# ==========================================
# 核心：子进程执行的点云处理函数
# ⚠️ 注意：这个函数必须定义在顶层，才能在 Windows 下被多进程成功序列化
# ==========================================
def process_single_frame(bin_path):
    try:
        # 1. 解析文件名
        file_name = os.path.basename(bin_path)
        out_path = os.path.join(DST_DIR, file_name.replace('.bin', '.pcd'))

        # 如果文件已经处理过，直接跳过（断点续传逻辑）
        if os.path.exists(out_path):
            return {"file": file_name, "status": "skipped", "points": 0}

        # 2. 读取点云
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        # 3. 工业级预处理：裁剪 ROI -> 体素降采样
        # 裁剪前方 50 米，左右 20 米
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(-20, -20, -3), max_bound=(50, 20, 3)
        )
        pcd = pcd.crop(bbox)
        pcd = pcd.voxel_down_sample(voxel_size=0.2)

        # 4. 保存结果
        o3d.io.write_point_cloud(out_path, pcd)
        return {"file": file_name, "status": "success", "points": len(pcd.points)}

    except Exception as e:
        return {"file": bin_path, "status": "error", "message": str(e)}


# ==========================================
# 主函数
# ==========================================
if __name__ == "__main__":
    # 获取前 50 个文件作为测试集
    bin_files = glob.glob(os.path.join(SRC_DIR, "*.bin"))
    # 为了测试效果，我们选 50 个文件
    test_files = bin_files[:500]
    print(f"准备处理 {len(test_files)} 个点云文件...")

    # ==========================================
    # 模式一：单进程处理 (传统 For 循环)
    # ==========================================
    print("\n🐢 开始单进程处理...")
    start_time = time.time()

    for file in test_files:
        process_single_frame(file)

    single_time = time.time() - start_time
    print(f"🐢 单进程处理完成，耗时: {single_time:.2f} 秒")

    # 清理刚才生成的文件，为了多进程测试公平
    for f in os.listdir(DST_DIR):
        os.remove(os.path.join(DST_DIR, f))
    print("已清空输出目录，准备多进程测试...")

    # ==========================================
    # 模式二：多进程处理 (ProcessPoolExecutor)
    # ==========================================
    print("\n🚀 开始多进程处理...")

    # 获取 CPU 核心数 (通常是 8, 12, 16)
    max_workers = os.cpu_count()
    print(f"检测到当前 CPU 逻辑核心数: {max_workers}，已全量开启并发！")

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # executor.map 会自动分配任务并回收结果
        results = list(executor.map(process_single_frame, test_files))

    multi_time = time.time() - start_time

    # 统计结果
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")

    print(f"🚀 多进程处理完成，耗时: {multi_time:.2f} 秒")
    print(f"成功: {success_count}, 失败: {error_count}")

    # ==========================================
    # 性能对比总结
    # ==========================================
    print("\n" + "=" * 40)
    print(f"🔥 性能加速比: {single_time / multi_time:.2f} 倍！")
    print("=" * 40)