# -*- coding: utf-8 -*-
"""
阶段 2：闭眼检测（EAR 眼睛长宽比）

原理：
  睁眼时上下眼皮离得远，眼睛"高"；闭眼时上下眼皮贴一起，眼睛扁成一条缝。
  每只眼睛用 6 个点描述：[外眼角, 上眼皮1, 上眼皮2, 内眼角, 下眼皮2, 下眼皮1]
  EAR = (两条竖线长度之和) / (2 × 横线长度)
  睁眼 EAR 大（约 0.25~0.35），闭眼 EAR 小（约 0.1 以下）。

运行：python stage2_ear.py
操作：按 q 退出
重点：盯着窗口里显示的 EAR 数值，自己睁眼/闭眼各观察几次，
      把下面 EAR_THRESHOLD 改成你"睁眼值"和"闭眼值"的中间数。
"""
import time
import cv2
import mediapipe as mp
import numpy as np

# ---------- 眼睛关键点索引（mediapipe 468 点里编号固定的 12 个） ----------
LEFT_EYE = [362, 385, 387, 263, 373, 380]   # 画面左侧那只眼
RIGHT_EYE = [33, 160, 158, 133, 153, 144]   # 画面右侧那只眼

EAR_THRESHOLD = 0.24   # 闭眼阈值：根据你的实测（睁眼0.33 / 闭眼0.18）取中间偏闭一侧
CLOSE_FRAMES = 2       # 连续 2 帧低于阈值就算闭眼（快速眨眼约 2~4 帧，太大抓不到）

def eye_aspect_ratio(landmarks, eye):
    """根据一只眼睛的 6 个关键点坐标，算 EAR 值"""
    p = np.array([[landmarks[i].x, landmarks[i].y] for i in eye])
    # 两条竖线：上眼皮到下眼皮
    v1 = np.linalg.norm(p[1] - p[5])   # P2 - P6
    v2 = np.linalg.norm(p[2] - p[4])   # P3 - P5
    # 横线：两个眼角之间
    h = np.linalg.norm(p[0] - p[3])    # P1 - P4
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)


# ---------- 初始化 ----------
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("打不开摄像头，把 CAMERA_INDEX 改成 1 或 2 再试")
    exit(1)
print("按 q 退出。观察 EAR 数值：睁眼和闭眼各记一次，用来调 EAR_THRESHOLD")

fps_start = time.time()
frame_count = 0
fps = 0.0
close_counter = 0   # 连续低于阈值的帧数（防抖用）
prev_closed = False  # 上一帧是否闭眼（用于统计眨眼次数）
blink_count = 0      # 累计眨眼次数

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

        # 两只眼各算一个 EAR，再取平均
        ear_l = eye_aspect_ratio(landmarks, LEFT_EYE)
        ear_r = eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear = (ear_l + ear_r) / 2.0

        # 防抖：连续 CLOSE_FRAMES 帧都低于阈值才算闭眼；一旦睁眼立即清零
        if ear < EAR_THRESHOLD:
            close_counter += 1
        else:
            close_counter = 0
        closed = close_counter >= CLOSE_FRAMES

        # 从"睁眼"跳变到"闭眼"的瞬间，算一次眨眼
        if closed and not prev_closed:
            blink_count += 1
        prev_closed = closed

        color = (0, 0, 255) if closed else (0, 255, 0)   # 闭眼红 / 睁眼绿

        # 画出 12 个眼睛点 + 测量线，让你看清"测的是什么"
        for eye in (LEFT_EYE, RIGHT_EYE):
            pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye]
            for (x, y) in pts:
                cv2.circle(frame, (x, y), 2, color, -1)
            cv2.line(frame, pts[0], pts[3], color, 1)   # 横线：两眼角
            cv2.line(frame, pts[1], pts[5], color, 1)   # 竖线1
            cv2.line(frame, pts[2], pts[4], color, 1)   # 竖线2

        # 屏幕显示 EAR 值和状态
        state = "CLOSED (闭眼)" if closed else "OPEN (睁眼)"
        cv2.putText(frame, f"EAR: {ear:.3f}  {state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 每 15 帧打印一次到命令行，方便你记数值
        if frame_count % 15 == 0:
            print(f"EAR 左={ear_l:.3f} 右={ear_r:.3f} 平均={ear:.3f} -> {state}")
    else:
        cv2.putText(frame, "no face", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 帧率
    frame_count += 1
    if frame_count % 10 == 0:
        now = time.time()
        fps = 10 / (now - fps_start)
        fps_start = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("DMS stage2 - EAR eye closure", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
print("程序已退出")
