import cv2
import torch
import numpy as np
import face_recognition
import sqlite3
import os
import json
from datetime import datetime
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.utils import non_max_suppression, scale_coords

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
        model.to(self.device).half().eval()
        return model

    def detect_faces(self, image):
        """检测人脸位置"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(image_rgb, number_of_times_to_upsample=2, model="small")
        print(f"检测到的人脸数量: {len(locations)}, 位置: {locations}")
        return locations

    def get_all_face_encodings_from_db(self):
        """从数据库中获取所有人脸编码及对应信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, employee_id, position, name, face_encoding FROM face_data")
        rows = cursor.fetchall()
        encodings = []
        for row in rows:
            id_, emp_id, pos, name, encoding_json = row
            try:
                encoding_list = json.loads(encoding_json)
                encoding = np.array(encoding_list)
                encodings.append((id_, emp_id, pos, name, encoding))
            except Exception as e:
                print(f"解析人脸编码失败：{e}")
        conn.close()
        return encodings

    def update_violation_time(self, matched_id, violation_time):
        """更新匹配人员的违规时间为传入的实际时间"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE face_data SET violation_time = ? WHERE id = ?",
            (violation_time, matched_id)
        )
        conn.commit()
        conn.close()

    def match_face(self, unknown_encoding, known_encodings, tolerance=0.6):
        """将未知人脸编码与已知编码匹配"""
        if not known_encodings:
            return None, None

        ids = [item[0] for item in known_encodings]
        emp_ids = [item[1] for item in known_encodings]
        positions = [item[2] for item in known_encodings]
        names = [item[3] for item in known_encodings]
        encodings = [item[4] for item in known_encodings]

        distances = face_recognition.face_distance(encodings, unknown_encoding)
        best_idx = np.argmin(distances)
        best_distance = float(distances[best_idx])

        if best_distance <= tolerance:
            matched_info = {
                'id': ids[best_idx],
                'employee_id': emp_ids[best_idx],
                'position': positions[best_idx],
                'name': names[best_idx]
            }
            return matched_info, best_distance
        else:
            return None, best_distance

    def detect(self, source, violation_time=None):
        """检测逻辑：使用传入的违规时间更新数据库"""
        if violation_time is None:
            violation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            try:
                violation_time = datetime.strptime(violation_time, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                print(f"无效的时间格式：{violation_time}，使用当前时间代替")
                violation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        dataset = LoadImages(source, img_size=320)

        os.makedirs("detection_results", exist_ok=True)
        os.makedirs("detected_faces", exist_ok=True)

        for path, img, im0s, vid_cap in dataset:
            output_path = os.path.join("detection_results", os.path.basename(path))
            print(f"正在处理图像: {path} => 输出到: {output_path}")

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
            matched_ids = set()  # 用于避免重复更新同一人

            for *xyxy, conf, cls in det:
                label = self.names[int(cls)]
                print(f"检测到类别: {label}, 置信度: {conf:.2f}")
                x1, y1, x2, y2 = map(int, xyxy)

                if label == "head":
                    no_helmet_detected = True
                    x1_crop = max(0, x1 - 100)
                    y1_crop = max(0, y1 - 150)
                    x2_crop = min(im0s.shape[1], x2 + 100)
                    y2_crop = min(im0s.shape[0], y2 + 100)
                    face_img = im0s[y1_crop:y2_crop, x1_crop:x2_crop]

                    face_filename = f"detected_faces/{os.path.basename(path).split('.')[0]}_face_{x1_crop}_{y1_crop}.jpg"
                    cv2.imwrite(face_filename, face_img)

                    face_locations = self.detect_faces(face_img)
                    if face_locations:
                        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        face_encodings = face_recognition.face_encodings(face_img_rgb,
                                                                         known_face_locations=face_locations)

                        for encoding in face_encodings:
                            print(f"未知人脸编码（前5个值）: {encoding[:5]}")
                            known_encodings = self.get_all_face_encodings_from_db()
                            matched_info, distance = self.match_face(encoding, known_encodings)

                            if matched_info:
                                matched_id = matched_info['id']
                                if matched_id not in matched_ids:
                                    self.update_violation_time(matched_id, violation_time)
                                    conn = sqlite3.connect(self.db_path)
                                    cursor = conn.cursor()
                                    cursor.execute(
                                        "SELECT id, employee_id, position, name, strftime('%Y-%m-%d %H:%M:%S', violation_time) "
                                        "FROM face_data WHERE id = ?",
                                        (matched_id,)
                                    )
                                    row = cursor.fetchone()
                                    print(f"匹配到人员: ID={row[0]}, 工号={row[1]}, 职务={row[2]}, 姓名={row[3]}, "
                                          f"违规时间={row[4]}, 匹配距离={distance:.4f}")
                                    conn.close()
                                    matched_ids.add(matched_id)
                                else:
                                    print(f"重复检测到同一人 ID={matched_id}，跳过记录。")
                            else:
                                print(f"未找到匹配人员，最小距离为: {distance:.4f}")

                cv2.rectangle(im0s, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(im0s, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imwrite(output_path, im0s)
            print(f"已保存结果图像到: {output_path}")
            if not no_helmet_detected:
                print("未检测到未戴头盔的人员")

            del img, pred, det
            torch.cuda.empty_cache()


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    detector = HelmetFaceDetector(weights="weights/helmet_head_person_l.pt", db_path="faces.db")
    detector.detect("D:/PyCharm/Deep_learning/safetyhelmet/VOC2028/JPEGImages/000009.jpg")
    conn = sqlite3.connect("faces.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, employee_id, position, name, strftime('%Y-%m-%d %H:%M:%S', violation_time) FROM face_data")
    conn.close()