# Tornado 接口（接收标注、查询统计）
'''
实现三个核心接口：

POST /api/upload：标注员上传一帧的标注结果（写入数据库）。

GET /api/frame/<frame_id>：查询某帧的详细标注（单表查询）。

GET /api/stats：查询各类障碍物的总数量（利用 SQL 聚合查询）。
'''
import tornado.ioloop
import tornado.web
import sqlite3
import json
import os

DB_PATH = "kitti_annotations.db"


# ==========================================
# 数据库初始化模块
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 建表：场景表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS frames (
            frame_id TEXT PRIMARY KEY,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # 建表：物体表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS objects (
            obj_id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_id TEXT,
            obj_type TEXT,
            x REAL, y REAL, z REAL,
            FOREIGN KEY(frame_id) REFERENCES frames(frame_id)
        )
    ''')
    conn.commit()
    conn.close()


# ==========================================
# 接口 1：上传标注数据 (POST)
# ==========================================
class UploadHandler(tornado.web.RequestHandler):
    def post(self):
        try:
            data = json.loads(self.request.body)
            frame_id = data.get("frame_id")
            objects = data.get("objects", [])

            if not frame_id:
                self.set_status(400)
                self.write({"status": "error", "message": "缺失 frame_id"})
                return

            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            # 插入场景记录 (忽略重复)
            cursor.execute("INSERT OR IGNORE INTO frames (frame_id) VALUES (?)", (frame_id,))

            # 插入物体记录
            for obj in objects:
                cursor.execute('''
                    INSERT INTO objects (frame_id, obj_type, x, y, z)
                    VALUES (?, ?, ?, ?, ?)
                ''', (frame_id, obj.get("type"), obj.get("x"), obj.get("y"), obj.get("z")))

            conn.commit()
            conn.close()

            self.write({"status": "success", "message": f"帧 {frame_id} 的 {len(objects)} 个目标入库成功！"})
        except Exception as e:
            self.set_status(500)
            self.write({"status": "error", "message": str(e)})


# ==========================================
# 接口 2：查询单帧详情 (GET)
# ==========================================
class FrameHandler(tornado.web.RequestHandler):
    def get(self, frame_id):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 查询该帧的所有物体
        cursor.execute("SELECT obj_type, x, y, z FROM objects WHERE frame_id = ?", (frame_id,))
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            self.set_status(404)
            self.write({"status": "error", "message": f"未找到帧 {frame_id} 的数据"})
            return

        result = []
        for row in rows:
            result.append({"type": row[0], "x": row[1], "y": row[2], "z": row[3]})

        self.write({
            "status": "success",
            "frame_id": frame_id,
            "total_objects": len(result),
            "data": result
        })


# ==========================================
# 接口 3：全局数据统计 (GET)
# ==========================================
class StatsHandler(tornado.web.RequestHandler):
    def get(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 利用 SQL 聚合查询 (GROUP BY)
        cursor.execute('''
            SELECT obj_type, COUNT(*) as count 
            FROM objects 
            GROUP BY obj_type
        ''')
        rows = cursor.fetchall()

        # 统计总帧数
        cursor.execute("SELECT COUNT(*) FROM frames")
        total_frames = cursor.fetchone()[0]

        conn.close()

        stats = {row[0]: row[1] for row in rows}
        self.write({
            "status": "success",
            "total_frames_uploaded": total_frames,
            "object_counts": stats
        })


# ==========================================
# 路由配置与启动
# ==========================================
def make_app():
    return tornado.web.Application([
        (r"/api/upload", UploadHandler),
        (r"/api/frame/([0-9]+)", FrameHandler),  # 正则匹配 frame_id
        (r"/api/stats", StatsHandler),
    ])


if __name__ == "__main__":
    init_db()
    app = make_app()
    app.listen(8889)  # 换个端口，避免和昨天的冲突
    print("🚀 数据集管理 API 已启动，监听端口: 8889")
    tornado.ioloop.IOLoop.current().start()