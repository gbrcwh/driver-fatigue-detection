# -*- coding: utf-8 -*-
"""
阶段 1：摄像头取流 + FaceMesh 检测 + 画出 468 个人脸关键点

运行：python stage1_facemesh.py
操作：按键盘 q 退出

说明：
- 默认用摄像头编号 0（笔记本自带 / 第一个 USB 摄像头）
- 每张人脸会得到 468 个关键点，覆盖眼、嘴、眉、脸轮廓等
- 后面阶段算「闭眼 / 打哈欠 / 点头」全靠这些点的坐标
"""
import time
import cv2
import mediapipe as mp

# ---------- 1. 初始化 ----------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils  # 画连接线会用（方式 B 可选）

# FaceMesh 参数（先照用，不懂没关系）：
#   max_num_faces=1          只检测一张脸（驾驶员场景够用，也更快）
#   min_detection_confidence 首次找到脸的置信度门槛，越低越容易找到、也越容易误判
#   min_tracking_confidence  后续跟踪的置信度门槛
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# 摄像头编号：0 通常是笔记本自带 / 第一个 USB 摄像头；打不开就改成 1 或 2 再试
CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("打不开摄像头，请检查：")
    print("   1. 摄像头是否被其他程序占用")
    print("   2. 把脚本里的 CAMERA_INDEX 改成 1 或 2 再试")
    exit(1)

print("已打开摄像头，按 q 退出程序")

# ---------- 2. 主循环 ----------
fps_start = time.time()   # 帧率计时起点
frame_count = 0
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        print("读取画面失败")
        break

    # 水平翻转，让画面像照镜子（不然你动右手，画面里动的是左手，很别扭）
    frame = cv2.flip(frame, 1)

    # OpenCV 读进来是 BGR，mediapipe 要 RGB，先转一下
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 这一行是核心：把整帧图丢进去，检测人脸关键点
    result = face_mesh.process(rgb)

    # ---------- 3. 画出关键点 ----------
    if result.multi_face_landmarks:
        h, w = frame.shape[:2]

        for face_landmarks in result.multi_face_landmarks:
            # 方式 A：画点 —— 把 468 个点每个画一个小圆点
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # 方式 B（可选）：画连接线，脸部轮廓更直观。
            # 想用就把上面的 for 循环删掉，取消下面这行的注释：
            # mp_drawing.draw_landmarks(
            #     frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

        # 显示检测到多少个关键点（正常是 468）
        n_points = len(result.multi_face_landmarks[0].landmark)
        cv2.putText(frame, f"points: {n_points}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "no face", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # ---------- 4. 帧率 ----------
    frame_count += 1
    if frame_count % 10 == 0:
        now = time.time()
        fps = 10 / (now - fps_start)
        fps_start = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # ---------- 5. 显示 ----------
    cv2.imshow("DMS stage1 - FaceMesh 468 points", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------- 6. 收尾 ----------
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
print("程序已退出")
