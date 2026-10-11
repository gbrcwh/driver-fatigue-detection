# -*- coding: utf-8 -*-
"""
阶段 3：三个疲劳信号同屏 —— 闭眼(EAR) + 打哈欠(MAR) + 低头(pitch)

新增两个量（和 EAR 同一个套路：算比值 → 设阈值 → 连续几帧才判定）：
1. MAR（嘴巴长宽比）：
   MAR = 上下唇内侧中点(13,14)竖直距离 ÷ 两嘴角(61,291)水平距离
   闭嘴小，说话中等，打哈欠很大（且持续更久）。
2. pitch（低头比）：
   pitch = 鼻尖(1)到下巴(152)竖直距离 ÷ 脸宽(234,454)
   抬头直视大，低头打瞌睡时变小。

运行：python stage3_fatigue_signals.py
操作：按 q 退出
重点：盯着 MAR 和 pitch 两个数值，把自己"闭嘴/说话/打哈欠"和"抬头/低头"
      的值记下来，回来改下面带"占位"二字的阈值。
"""
import time
import cv2
import mediapipe as mp
import numpy as np

# ---------- 关键点索引 ----------
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

MOUTH_TOP, MOUTH_BOTTOM = 13, 14      # 上下嘴唇内侧中点
MOUTH_LEFT, MOUTH_RIGHT = 61, 291     # 左右内嘴角

NOSE_TIP, CHIN = 1, 152               # 鼻尖、下巴
LEFT_CHEEK, RIGHT_CHEEK = 234, 454    # 左右脸颊最宽点

# ---------- 阈值（标"占位"的等你实测后回来改） ----------
EAR_THRESHOLD = 0.24    # 已标定好，不动
CLOSE_FRAMES = 2        # 已标定好，不动

MAR_THRESHOLD = 0.6     # 已标定：闭嘴0.05/说话0.15/哈欠0.8+，取0.6卡在中间
YAWN_FRAMES = 6         # 连续 6 帧(~0.2s)都超阈值才算打哈欠，过滤掉"说话"的短暂张嘴

NOD_THRESHOLD = 0.50    # 已标定：平视0.72/低头0.30，取中间0.50
NOD_FRAMES = 3          # 连续 3 帧低于阈值才算低头


def _dist(lm, a, b):
    """两个关键点的欧氏距离（用归一化坐标，比值为纯比值）"""
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
print("按 q 退出。记下 MAR 和 pitch 的数值用来标定阈值")

fps_start = time.time()
frame_count = 0
fps = 0.0
close_counter = 0
yawn_counter = 0
nod_counter = 0
prev_closed = False
prev_yawn = False
blink_count = 0
yawn_count = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        # 1) 眼睛 EAR
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

        # 2) 嘴巴 MAR
        mar = mouth_aspect_ratio(landmarks)
        if mar > MAR_THRESHOLD:
            yawn_counter += 1
        else:
            yawn_counter = 0
        yawning = yawn_counter >= YAWN_FRAMES
        if yawning and not prev_yawn:
            yawn_count += 1
        prev_yawn = yawning

        # 3) 头部 pitch
        pitch = pitch_ratio(landmarks)
        if pitch < NOD_THRESHOLD:
            nod_counter += 1
        else:
            nod_counter = 0
        head_down = nod_counter >= NOD_FRAMES

        # 控制台每 15 帧打印一次（视频窗口左下角也显示，这里方便你复制数值）
        if frame_count % 15 == 0:
            print(f"EAR={ear:.3f}({'闭' if closed else '睁'})  "
                  f"MAR={mar:.3f}({'哈欠' if yawning else '正常'})  "
                  f"pitch={pitch:.3f}({'低头' if head_down else '正常'})  "
                  f"眨眼={blink_count} 哈欠={yawn_count}")

        # ---------- 画测量线 ----------
        def px(i):
            return int(landmarks[i].x * w), int(landmarks[i].y * h)

        # 眼睛：12 个点 + 线（闭眼红 / 睁眼绿）
        eye_color = (0, 0, 255) if closed else (0, 255, 0)
        for eye in (LEFT_EYE, RIGHT_EYE):
            pts = [px(i) for i in eye]
            for (x, y) in pts:
                cv2.circle(frame, (x, y), 2, eye_color, -1)
            cv2.line(frame, pts[0], pts[3], eye_color, 1)
            cv2.line(frame, pts[1], pts[5], eye_color, 1)
            cv2.line(frame, pts[2], pts[4], eye_color, 1)

        # 嘴巴：4 个点 + 竖横线（打哈欠紫红 / 正常橙）
        mouth_color = (255, 0, 255) if yawning else (0, 165, 255)
        m_pts = [px(i) for i in (MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT)]
        for (x, y) in m_pts:
            cv2.circle(frame, (x, y), 2, mouth_color, -1)
        cv2.line(frame, m_pts[0], m_pts[1], mouth_color, 1)   # 竖直
        cv2.line(frame, m_pts[2], m_pts[3], mouth_color, 1)   # 水平

        # 头部：鼻尖/下巴/两颊 4 点 + 竖横线（低头红 / 正常蓝）
        head_color = (0, 0, 255) if head_down else (255, 255, 0)
        h_pts = [px(i) for i in (NOSE_TIP, CHIN, LEFT_CHEEK, RIGHT_CHEEK)]
        for (x, y) in h_pts:
            cv2.circle(frame, (x, y), 2, head_color, -1)
        cv2.line(frame, h_pts[0], h_pts[1], head_color, 1)
        cv2.line(frame, h_pts[2], h_pts[3], head_color, 1)

        # ---------- 仪表盘文字 ----------
        eye_state = "CLOSED" if closed else "OPEN"
        mouth_state = "YAWN" if yawning else "normal"
        head_state = "HEAD-DOWN" if head_down else "level"
        cv2.putText(frame, f"EAR:   {ear:.3f}  {eye_state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
        cv2.putText(frame, f"MAR:   {mar:.3f}  {mouth_state}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, mouth_color, 2)
        cv2.putText(frame, f"pitch: {pitch:.3f}  {head_state}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, head_color, 2)
        cv2.putText(frame, f"blinks: {blink_count}   yawns: {yawn_count}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    else:
        cv2.putText(frame, "no face", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 帧率
    frame_count += 1
    if frame_count % 10 == 0:
        now = time.time()
        fps = 10 / (now - fps_start)
        fps_start = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("DMS stage3 - eye + yawn + head", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
print("程序已退出")
