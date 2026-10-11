# -*- coding: utf-8 -*-
"""
阶段 4：疲劳判定 + 报警（成品版）

报警规则（任一触发即报警，并显示原因）：
1. 长时间闭眼：连续闭眼超过 1 秒（30 帧）
2. 打哈欠
3. 持续低头：连续低头超过 1 秒（30 帧）

另外实时显示 PERCLOS（最近 2 秒内闭眼时间占比），这是行业主流疲劳指标，
你可以把它理解成"闭眼时长占比"，越高越困。

报警动作：红屏闪烁 + Windows 提示音（报警期间每 2 秒响一次）
运行：python stage4_alarm.py    操作：按 q 退出
"""
import time
import threading
import winsound
from collections import deque
import cv2
import mediapipe as mp
import numpy as np

from cn_text import draw_texts

# ---------- 关键点索引 ----------
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
NOSE_TIP, CHIN = 1, 152
LEFT_CHEEK, RIGHT_CHEEK = 234, 454

# ---------- 阈值（已按你的实测标定） ----------
EAR_THRESHOLD = 0.24
CLOSE_FRAMES = 2
MAR_THRESHOLD = 0.6
YAWN_FRAMES = 6
NOD_THRESHOLD = 0.50
NOD_FRAMES = 3

# ---------- 报警参数 ----------
ALARM_CLOSE_FRAMES = 30   # 连续闭眼 30 帧(约1秒)就报警
ALARM_NOD_FRAMES = 30     # 连续低头 30 帧(约1秒)就报警
PERCLOS_WINDOW = 60       # PERCLOS 统计窗口：最近 60 帧(约2秒)


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
print("按 q 退出。试试：闭眼1秒 / 打哈欠 / 低头1秒，看会不会报警")

fps_start = time.time()
frame_count = 0
fps = 0.0
close_counter = 0
yawn_counter = 0
nod_counter = 0
closed_duration = 0       # 连续闭眼帧数
head_down_duration = 0    # 连续低头帧数
prev_closed = False
prev_yawn = False
blink_count = 0
yawn_count = 0
closed_history = deque(maxlen=PERCLOS_WINDOW)   # 最近 60 帧的闭眼状态
last_beep = 0.0

# 报警提示音：winsound.Beep 是 Windows 最底层的"发一个音"接口，最不容易被静音。
# 它是同步的，所以放到子线程里跑，不卡住视频主循环。
def _play_beep():
    try:
        winsound.Beep(880, 400)   # 880Hz，响 0.4 秒
    except Exception:
        pass

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    alarm = False
    reason = ""
    cn = []  # 中文叠加文字（cv2 画不了中文，统一交给 draw_texts 画）

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        # 1) 眼睛
        ear = (eye_aspect_ratio(landmarks, LEFT_EYE)
               + eye_aspect_ratio(landmarks, RIGHT_EYE)) / 2.0
        if ear < EAR_THRESHOLD:
            close_counter += 1
        else:
            close_counter = 0
        closed = close_counter >= CLOSE_FRAMES
        if closed and not prev_closed:
            blink_count += 1
        prev_closed = closed

        # 2) 嘴巴
        mar = mouth_aspect_ratio(landmarks)
        if mar > MAR_THRESHOLD:
            yawn_counter += 1
        else:
            yawn_counter = 0
        yawning = yawn_counter >= YAWN_FRAMES
        if yawning and not prev_yawn:
            yawn_count += 1
        prev_yawn = yawning

        # 3) 头部
        pitch = pitch_ratio(landmarks)
        if pitch < NOD_THRESHOLD:
            nod_counter += 1
        else:
            nod_counter = 0
        head_down = nod_counter >= NOD_FRAMES

        # ---------- 时长统计 ----------
        closed_duration = closed_duration + 1 if closed else 0
        head_down_duration = head_down_duration + 1 if head_down else 0

        # PERCLOS：最近窗口内闭眼占比
        closed_history.append(closed)
        perclos = sum(closed_history) / len(closed_history)

        # ---------- 报警判定 ----------
        if closed_duration >= ALARM_CLOSE_FRAMES:
            reason = "长时间闭眼"
        elif yawning:
            reason = "打哈欠"
        elif head_down_duration >= ALARM_NOD_FRAMES:
            reason = "持续低头"
        alarm = bool(reason)

        # ---------- 画测量线 ----------
        def px(i):
            return int(landmarks[i].x * w), int(landmarks[i].y * h)

        eye_color = (0, 0, 255) if closed else (0, 255, 0)
        for eye in (LEFT_EYE, RIGHT_EYE):
            pts = [px(i) for i in eye]
            for (x, y) in pts:
                cv2.circle(frame, (x, y), 2, eye_color, -1)
            cv2.line(frame, pts[0], pts[3], eye_color, 1)
            cv2.line(frame, pts[1], pts[5], eye_color, 1)
            cv2.line(frame, pts[2], pts[4], eye_color, 1)

        mouth_color = (255, 0, 255) if yawning else (0, 165, 255)
        m_pts = [px(i) for i in (MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT)]
        for (x, y) in m_pts:
            cv2.circle(frame, (x, y), 2, mouth_color, -1)
        cv2.line(frame, m_pts[0], m_pts[1], mouth_color, 1)
        cv2.line(frame, m_pts[2], m_pts[3], mouth_color, 1)

        head_color = (0, 0, 255) if head_down else (255, 255, 0)
        h_pts = [px(i) for i in (NOSE_TIP, CHIN, LEFT_CHEEK, RIGHT_CHEEK)]
        for (x, y) in h_pts:
            cv2.circle(frame, (x, y), 2, head_color, -1)
        cv2.line(frame, h_pts[0], h_pts[1], head_color, 1)
        cv2.line(frame, h_pts[2], h_pts[3], head_color, 1)

        # ---------- 仪表盘 ----------
        eye_state = "CLOSED" if closed else "OPEN"
        mouth_state = "YAWN" if yawning else "normal"
        head_state = "HEAD-DOWN" if head_down else "level"
        cv2.putText(frame, f"EAR:   {ear:.3f}  {eye_state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
        cv2.putText(frame, f"MAR:   {mar:.3f}  {mouth_state}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, mouth_color, 2)
        cv2.putText(frame, f"pitch: {pitch:.3f}  {head_state}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, head_color, 2)
        cn.append((f"PERCLOS: {perclos:.0%}  眨眼={blink_count} 哈欠={yawn_count}",
                   10, 118, 20, (200, 200, 200)))

    else:
        cv2.putText(frame, "no face", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # ---------- 报警 UI ----------
    if alarm:
        now = time.time()
        if now - last_beep > 2.0:   # 每 2 秒响一次
            threading.Thread(target=_play_beep, daemon=True).start()
            last_beep = now

        # 红屏闪烁
        if (frame_count // 10) % 2 == 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        cn.append(("!! 疲劳警告 !!", frame.shape[1] // 2 - 120, 40, 32, (0, 0, 255)))
        cn.append((f"原因: {reason}", frame.shape[1] // 2 - 90, 84, 24, (0, 0, 255)))
    else:
        cn.append(("状态: 正常", frame.shape[1] - 150, 28, 22, (0, 255, 0)))

    # 帧率
    frame_count += 1
    if frame_count % 10 == 0:
        now = time.time()
        fps = 10 / (now - fps_start)
        fps_start = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    draw_texts(frame, cn)

    cv2.imshow("DMS stage4 - fatigue alarm", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
print("程序已退出")
