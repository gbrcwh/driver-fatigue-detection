# -*- coding: utf-8 -*-
"""环境体检：验证 DMS 疲劳检测项目所需的全部库能正常工作"""
import numpy as np
import cv2
import mediapipe as mp
import pygame

print("cv2      :", cv2.__version__)
print("mediapipe:", mp.__version__)
print("numpy    :", np.__version__)
print("pygame   :", pygame.ver)

# 1. cv2 图像操作测试（画一张纯绿图，转 RGB）
img = np.zeros((100, 100, 3), np.uint8)
img[:, :, 1] = 255
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print("cv2 图像操作: 正常")

# 2. FaceMesh 模型实际跑一张图（噪音图检测不到脸，但不该报错）
fm = mp.solutions.face_mesh.FaceMesh()
result = fm.process(rgb)
fm.close()
print("FaceMesh 跑通, 检测到人脸数:", len(result.multi_face_landmarks or []))

print("\n全部通过 ✅ 可以开始阶段 1")
