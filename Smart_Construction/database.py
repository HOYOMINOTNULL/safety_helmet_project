import cv2
import torch
import numpy as np
import face_recognition
import sqlite3
import os
import json
import random
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.utils import non_max_suppression, scale_coords
from config import  FACE_MODEL_PATH

class HelmetFaceDetector:
    def __init__(self, weights, db_path):
        self.weights = weights
        self.db_path = db_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_thres = 0.25
        self.iou_thres = 0.5
        self.model = self.load_model()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        print(f"模型支持的类别: {self.names}")

    def load_model(self):
        """加载模型并启用半精度推理"""
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).half().eval()  # 使用半精度（FP16）推理
        return model

    def detect_faces(self, image):
        """检测人脸位置"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(
            image_rgb,
            number_of_times_to_upsample=2,
            model="small"
        )
        print(f"检测到的人脸数量: {len(locations)}, 位置: {locations}")
        return locations

    def save_face_to_db(self, face_encoding):
        """把单个人脸编码存入 SQLite，并生成 6 位工号、随机职务和随机中文姓名"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 如果库里有旧触发器，先删掉
        cursor.execute("DROP TRIGGER IF EXISTS set_employee_id")

        # 创建表（如果不存在）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_data (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                employee_id TEXT,
                position TEXT,
                violation_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                name TEXT,
                face_encoding TEXT
            )
        """)

        # 随机生成工地职务
        positions = [
            "项目经理", "安全员", "施工员", "质检员", "技术员",
            "材料员", "测量员", "焊工", "木工", "电工",
            "机械操作工", "架子工", "混凝土工", "抹灰工", "油漆工"
        ]
        position = random.choice(positions)

        # 随机生成中文姓名
        surnames = [
            "赵", "钱", "孙", "李", "周", "吴", "郑", "王",
            "冯", "陈", "褚", "卫", "蒋", "沈", "韩", "杨",
            "朱", "秦", "尤", "许", "何", "吕", "施", "张"
        ]
        given_chars = [
            "伟", "芳", "娜", "敏", "静", "丽", "强", "磊",
            "军", "洋", "勇", "艳", "杰", "娟", "涛", "明",
            "超", "霞", "平", "刚"
        ]
        given_name = "".join(random.choices(given_chars, k=random.choice([1, 2])))
        name = random.choice(surnames) + given_name

        # 准备要插入的编码
        encoding_list = face_encoding.tolist()
        encoding_json = json.dumps(encoding_list)

        # 插入新记录（face_encoding, position, name）
        cursor.execute(
            "INSERT INTO face_data (face_encoding, position, name) VALUES (?, ?, ?)",
            (encoding_json, position, name)
        )
        last_id = cursor.lastrowid  # 拿到自增的主键

        # 用主键生成 6 位前导零工号，并更新回去
        employee_id = f"{last_id:06d}"
        cursor.execute(
            "UPDATE face_data SET employee_id = ? WHERE id = ?",
            (employee_id, last_id)
        )

        conn.commit()

        # 打印验证一下
        cursor.execute(
            "SELECT id, employee_id, position, name, violation_time "
            "FROM face_data WHERE id = ?",
            (last_id,)
        )
        latest = cursor.fetchone()
        print(
            f"新增记录: 工号={latest[1]}, 职务={latest[2]}, 姓名={latest[3]}, "
            f"违规时间={latest[4]}"
        )

        conn.close()

    def detect(self, source):
        """检测图像中的目标并处理未戴头盔的人员"""
        dataset = LoadImages(source, img_size=640)  # 减小输入分辨率

        os.makedirs("detection_results", exist_ok=True)
        os.makedirs("detected_faces", exist_ok=True)

        for path, img, im0s, vid_cap in dataset:
            output_path = os.path.join("detection_results", os.path.basename(path))
            print(f"正在处理图像: {path} => 输出到: {output_path}")

            # 转成 FP16 tensor
            img = torch.from_numpy(img).to(self.device).half()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            with torch.no_grad():
                pred = self.model(img)[0]
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

            if pred[0] is None or len(pred[0]) == 0:
                print("未检测到任何目标")
                continue

            det = pred[0]
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

            no_helmet_detected = False
            for *xyxy, conf, cls in det:
                label = self.names[int(cls)]
                print(f"检测到类别: {label}, 置信度: {conf:.2f}")
                x1, y1, x2, y2 = map(int, xyxy)

                if label == "head":
                    no_helmet_detected = True
                    # 扩大一点区域方便人脸检测
                    x1c = max(0, x1 - 100)
                    y1c = max(0, y1 - 150)
                    x2c = min(im0s.shape[1], x2 + 100)
                    y2c = min(im0s.shape[0], y2 + 100)
                    face_img = im0s[y1c:y2c, x1c:x2c]

                    print(f"裁剪区域大小: {face_img.shape}")
                    face_filename = (
                        f"detected_faces/"
                        f"{os.path.splitext(os.path.basename(path))[0]}_"
                        f"{x1c}_{y1c}.jpg"
                    )
                    cv2.imwrite(face_filename, face_img)
                    print(f"保存人脸图像到: {face_filename}")

                    # 再做一次人脸检测 + 编码
                    locs = self.detect_faces(face_img)
                    if locs:
                        encs = face_recognition.face_encodings(
                            cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB),
                            known_face_locations=locs,
                            num_jitters=1
                        )
                        print(f"提取到的人脸编码数量: {len(encs)}")
                        if encs:
                            self.save_face_to_db(encs[0])
                            print(f"检测到未佩戴安全帽的人员，已存入数据库: {path}")

                # 画框 + 标注
                cv2.rectangle(im0s, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    im0s,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

            cv2.imwrite(output_path, im0s)
            print(f"已保存结果图像到: {output_path}")
            if not no_helmet_detected:
                print("未检测到未戴头盔的人员")

            # 释放显存
            del img, pred, det
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # 调整 PyTorch CUDA 内存分配
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

    detector = HelmetFaceDetector(
        weights=FACE_MODEL_PATH,
        db_path="./faces.db"
    )
    detector.detect(
        "D:/PyCharm/Deep_learning/safetyhelmet/VOC2028/JPEGImages"
    )

    # 演示：打印所有数据库记录
    conn = sqlite3.connect("faces.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id,
               employee_id,
               position,
               name,
               strftime('%Y-%m-%d %H:%M:%S', violation_time),
               face_encoding
        FROM face_data
    """)
    for row in cursor.fetchall():
        id_, emp_id, pos, name, viol, enc = row
        print(f"工号: {emp_id}, 职务: {pos}, 姓名: {name}, 违规时间: {viol}")
    conn.close()
