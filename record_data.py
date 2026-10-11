# -*- coding: utf-8 -*-
"""
采集脚本：给多个测试者录数据，用于标定阈值、检验泛化

用法：
  python record_data.py 名字 [标签]
  例：python record_data.py 同学A 正常
      python record_data.py 同学B 疲劳模拟

每次运行生成：
  data/名字_标签_时间戳.csv   逐帧数据（EAR/MAR/pitch/状态/计数）
  （把下面 VIDEO_RECORD 改成 True 会同时存视频，文件较大，默认关）

按 q 停止，停止时控制台打印一份统计摘要。
"""
import sys
import time
import os
import csv
from collections import deque
import cv2
import mediapipe as mp
import numpy as np

# ---------- 关键点索引 & 阈值（和 stage4 一致，已按你标定） ----------
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
NOSE_TIP, CHIN = 1, 152
LEFT_CHEEK, RIGHT_CHEEK = 234, 454

EAR_THRESHOLD = 0.24
CLOSE_FRAMES = 2
MAR_THRESHOLD = 0.6
YAWN_FRAMES = 6
NOD_THRESHOLD = 0.50
NOD_FRAMES = 3
PERCLOS_WINDOW = 60

VIDEO_RECORD = False   # 改成 True 会同时录视频（MJPG .avi，1 分钟约几十 MB）

name = sys.argv[1] if len(sys.argv) > 1 else "sample"
label = sys.argv[2] if len(sys.argv) > 2 else "默认"


def _dist(lm, a, b):
    dx = lm[a].x - lm[b].x
    dy = lm[a].y - lm[b].y
    return (dx * dx + dy * dy) ** 0.5


def eye_aspect_ratio(landmarks, eye):
    p = np.array([[landmarks[i].x, landmarks[i].y] for i in eye])
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    return (v1 + v2) / (2.0 * h) if h else 0.0


def mouth_aspect_ratio(landmarks):
    v = _dist(landmarks, MOUTH_TOP, MOUTH_BOTTOM)
    h = _dist(landmarks, MOUTH_LEFT, MOUTH_RIGHT)
    return v / h if h else 0.0


def pitch_ratio(landmarks):
    v = _dist(landmarks, NOSE_TIP, CHIN)
    h = _dist(landmarks, LEFT_CHEEK, RIGHT_CHEEK)
    return v / h if h else 0.0


# ---------- 初始化 ----------
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("打不开摄像头，改脚本里的摄像头编号再试")
    exit(1)

# 输出目录与文件
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(out_dir, exist_ok=True)
stamp = time.strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(out_dir, f"{name}_{label}_{stamp}.csv")
csv_file = open(csv_path, "w", newline="", encoding="utf-8-sig")  # utf-8-sig 让 Excel 打开不乱码
writer = csv.writer(csv_file)
writer.writerow(["t_sec", "frame", "ear_left", "ear_right", "ear_avg", "mar",
                 "pitch", "perclos", "closed", "yawning", "head_down",
                 "blinks", "yawns"])

video_writer = None
if VIDEO_RECORD:
    ok, probe = cap.read()
    if ok:
        hgt, wid = probe.shape[:2]
        video_path = os.path.join(out_dir, f"{name}_{label}_{stamp}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (wid, hgt))
        print("视频将保存到:", video_path)

print(f"样本: {name}  标签: {label}")
print("3 秒后开始录制，按 q 结束")

# ---------- 3 秒倒计时 ----------
for i in (3, 2, 1):
    t0 = time.time()
    while time.time() - t0 < 1.0:
        ok, frame = cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"{i}", (frame.shape[1] // 2 - 30, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            cv2.imshow("record", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ---------- 状态变量 ----------
start_time = time.time()
frame_count = 0
close_counter = 0
yawn_counter = 0
nod_counter = 0
prev_closed = False
prev_yawn = False
blink_count = 0
yawn_count = 0
closed_history = deque(maxlen=PERCLOS_WINDOW)

# 摘要累加器
ear_sum = 0.0; ear_min = 9e9; ear_max = -9e9
ear_open_sum = 0.0; ear_open_n = 0
ear_closed_sum = 0.0; ear_closed_n = 0
mar_max = -9e9
pitch_sum = 0.0; pitch_min = 9e9
perclos_sum = 0.0; perclos_max = -9e9

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        ear_l = eye_aspect_ratio(landmarks, LEFT_EYE)
        ear_r = eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear = (ear_l + ear_r) / 2.0
        if ear < EAR_THRESHOLD:
            close_counter += 1
        else:
            close_counter = 0
        closed = close_counter >= CLOSE_FRAMES
        if closed and not prev_closed:
            blink_count += 1
        prev_closed = closed

        mar = mouth_aspect_ratio(landmarks)
        if mar > MAR_THRESHOLD:
            yawn_counter += 1
        else:
            yawn_counter = 0
        yawning = yawn_counter >= YAWN_FRAMES
        if yawning and not prev_yawn:
            yawn_count += 1
        prev_yawn = yawning

        pitch = pitch_ratio(landmarks)
        if pitch < NOD_THRESHOLD:
            nod_counter += 1
        else:
            nod_counter = 0
        head_down = nod_counter >= NOD_FRAMES

        closed_history.append(closed)
        perclos = sum(closed_history) / len(closed_history)

        # ---------- 写一行 CSV ----------
        t = time.time() - start_time
        writer.writerow([round(t, 3), frame_count,
                         round(ear_l, 4), round(ear_r, 4), round(ear, 4),
                         round(mar, 4), round(pitch, 4), round(perclos, 4),
                         int(closed), int(yawning), int(head_down),
                         blink_count, yawn_count])

        # ---------- 摘要累加 ----------
        ear_sum += ear; ear_min = min(ear_min, ear); ear_max = max(ear_max, ear)
        if closed:
            ear_closed_sum += ear; ear_closed_n += 1
        else:
            ear_open_sum += ear; ear_open_n += 1
        mar_max = max(mar_max, mar)
        pitch_sum += pitch; pitch_min = min(pitch_min, pitch)
        perclos_sum += perclos; perclos_max = max(perclos_max, perclos)

        # ---------- 画面提示 ----------
        h, w = frame.shape[:2]
        cv2.putText(frame, f"EAR {ear:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"MAR {mar:.3f}  pitch {pitch:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"REC {t:.0f}s  blinks={blink_count} yawns={yawn_count}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "no face", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if video_writer is not None:
        video_writer.write(frame)

    cv2.imshow("record", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:          # q 或 ESC 键
        break
    if cv2.getWindowProperty("record", cv2.WND_PROP_VISIBLE) < 1:  # 直接点窗口右上角 X
        break

    frame_count += 1
    if frame_count % 10 == 0:
        csv_file.flush()

# ---------- 收尾 & 摘要 ----------
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
csv_file.close()
if video_writer is not None:
    video_writer.release()

total_t = time.time() - start_time
print("\n==== 采集摘要 ====")
print(f"样本: {name}  标签: {label}")
print(f"时长: {total_t:.1f} 秒   帧数: {frame_count}")
print(f"眨眼: {blink_count} 次   哈欠: {yawn_count} 次")
if frame_count > 0:
    print(f"EAR 整体: min={ear_min:.3f}  mean={ear_sum/frame_count:.3f}  max={ear_max:.3f}")
    if ear_open_n:
        print(f"  睁眼时平均 EAR: {ear_open_sum/ear_open_n:.3f}  (n={ear_open_n})")
    if ear_closed_n:
        print(f"  闭眼时平均 EAR: {ear_closed_sum/ear_closed_n:.3f}  (n={ear_closed_n})")
    print(f"MAR 最大: {mar_max:.3f}")
    print(f"pitch: mean={pitch_sum/frame_count:.3f}  min={pitch_min:.3f}")
    print(f"PERCLOS: 平均={perclos_sum/frame_count:.1%}  峰值={perclos_max:.1%}")
print(f"CSV 已保存: {csv_path}")
