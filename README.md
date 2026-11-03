# 🚀 L4 Perception Data Pipeline & MLOps Framework

> 一套面向 L4 级自动驾驶（干线物流重卡场景）的感知数据全生命周期管理与闭环系统。涵盖多进程数据清洗、自动化质检 (Auto-QA)、复杂 3D 数据增强、Web 微服务 API 以及 MLOps 实验追踪体系。

![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.10-blue)
![Open3D](https://img.shields.io/badge/Open3D-0.19-green)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13-red)
![ClearML](https://img.shields.io/badge/ClearML-MLOps-orange)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

---

## 📖 项目简介 (Introduction)

在自动驾驶实际业务中，“垃圾进，垃圾出（Garbage in, Garbage out）”是铁律。本项目致力于解决海量点云数据处理慢、人工质检成本高、长尾场景稀缺、数据与模型版本脱节等痛点，实现从**“原始脏数据”到“模型高价值训练集”**的自动化流转。

## ✨ 核心特性 (Core Features)

- **⚡ 高性能多进程 Pipeline**：基于 `concurrent.futures` 实现点云多进程清洗，有效解决 CPU 密集型任务的 GIL 锁瓶颈，在 16 核环境下实测加速比达 **3.7x**。
- **🛡️ 四维自动化质检 (Auto-QA)**：独立开发质检脚本，实现 100% 全量拦截：包含*物理尺寸校验*、*OBB 点云包含率*、*BEV 视角穿模检测 (Shapely)* 以及*跨帧时序物理校验*。
- **🧊 物理约束的 3D 数据增强**：实现 Copy-Paste 局部目标增强（含碰撞检测与地面吸附）；实现基于内外参投影矩阵的 **2D/3D 联合增强**，保证多传感器绝对对齐。
- **🌐 数据中台微服务**：基于 Tornado 构建标注数据接收 API，使用多线程压测达到 **531 QPS**，实现平滑削峰填谷（集成 Redis 逻辑）。
- **🖥️ 数据资产与 MLOps 闭环**：深度集成 **FiftyOne**（可视化与难例挖掘）与 **ClearML / DVC**（版本控制与实验参数追踪）。

## 🚀 快速开始 (Quick Start)

### 1. 环境准备
建议使用 Conda 隔离环境（避免 OpenCV/Open3D 冲突），本项目已在 Python 3.8 - 3.10 环境下测试通过：

```bash
# 创建并激活虚拟环境
conda create -n perception_data python=3.8 -y
conda activate perception_data

# 安装核心依赖
pip install -r requirements.txt
```
### 2. 环境准备
请将 KITTI 数据集按如下标准结构放置于 data/kitti/ 目录下：
```text
data/kitti/
└── training/
    ├── calib/       # 相机内外参矩阵
    ├── image_2/     # 左侧彩色相机图像
    ├── label_2/     # 3D 边界框标签
    └── velodyne/    # 激光雷达点云 (.bin)
```
### 3. 运行终极清洗 Pipeline
执行多进程预处理与 Auto-QA 质检，结果会实时输出到终端，并写入本地 pipeline_meta.db SQLite 数据库：
```bash
python main_pipeline.py --data_dir data/kitti/training/velodyne --workers 8
```
运行后，你可以使用 DBeaver 连接 SQLite 数据库，输入 SELECT * FROM frames WHERE qa_status='FAILED'; 查看被拦截的异常数据。
### 4. 启动数据接收 Web API
如果你需要测试 Tornado 接口的高并发性能，请先启动服务端：
```bash
# 终端 1：启动后端服务
python web_api/server.py
```
另开一个终端运行多线程压测脚本（模拟 50 个并发，发送 500 个标注请求）：
```bash
# 终端 2：运行压测
python tests/stress_test.py --requests 500 --workers 50
```
### 5. 终极 Pipeline 全量测试
*经历多传感器坐标系对齐修复后，11 帧 KITTI 数据 100% 通过质检，多进程耗时仅 4 秒。*
![Pipeline db](assets/pipeline_db.png)
图：Auto-QA 系统成功捕获 KITTI 数据集中的极小截断目标、BEV 视角穿模以及裁剪越界异常，触发告警拦截。
---


## 📂 工程目录结构 (Directory Structure)

项目采用模块化设计，便于解耦与扩展：

```text
L4-Perception-Data-Pipeline/
│
├── configs/                 # 配置文件目录
│   └── config.yaml          # 数据路径、超参数(voxel_size等)、DB配置
│
├── core/                    # 🌟 核心算法与处理模块
│   ├── preprocess.py        # Open3D 点云读取、ROI裁剪、体素降采样、RANSAC去地面
│   ├── auto_qa.py           # 4维质检逻辑：尺寸、包含率、BEV IoU 穿模、时序速度拦截
│   ├── augmentation.py      # 全局增强(旋转/缩放/翻转) + 局部增强(Copy-Paste)
│   └── joint_projection.py  # 2D/3D 联合增强：3D框重投影至图像并镜像验证
│
├── web_api/                 # 🌐 Web 后端微服务
│   ├── server.py            # Tornado 服务：接收标注JSON、查询统计数据
│   └── db_manager.py        # SQLite/MySQL 数据库操作封装 (CTE与窗口函数实现)
│
├── mlo_ops/                 # ⚙️ MLOps 联动脚本
│   ├── clearml_track.py     # 实验追踪：自动记录参数与环境
│   └── dvc_config.sh        # DVC 初始化与远程存储推拉脚本
│
├── tests/                   # 单元测试 (如验证 IoU 计算、坐标变换逻辑)
├── assets/                  # 存放 README 引用的截图 (点云、ClearML、FiftyOne截图)
├── main_pipeline.py         # 🚀 终极主程序：整合多进程与 Auto-QA 的自动化流水线
├── requirements.txt         # 依赖清单
└── README.md                # 项目说明文档