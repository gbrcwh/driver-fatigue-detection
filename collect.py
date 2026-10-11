# -*- coding: utf-8 -*-
"""
结构化数据采集脚本（多步骤、多测试者）—— 正式采集用这个

流程：
1. 启动时在命令行录入测试者信息（编号/性别/身高/年龄/是否戴眼镜）
2. 摄像头窗口里按数字键 1~5 开始对应步骤，按 s/空格 停止该步骤，按 q/ESC 结束
3. 只有某步骤"进行中"时，数据才写入 CSV（每步数据纯净、便于后期验证真实性）
4. 每个测试者一个文件夹（文件夹名含基本信息），每步一个 CSV，外加一份 _info.txt 溯源记录

步骤按键（用键盘上方的数字键，不是小键盘）：
  1 = 睁眼基线   2 = 闭眼   3 = 哈欠   4 = 低头   5 = 自然眨眼
  s / 空格 = 停止当前步骤
  q / ESC   = 结束本次采集

输出结构：
  data/测试者1号_男_175cm_22岁_戴眼镜/
      _info.txt                       （测试者信息 + 各步骤时长，用于溯源）
      睁眼基线_20261011_120000.csv
      闭眼_20261011_120030.csv
      ...
"""
import time
import os
import csv
from collections import deque
import cv2
import mediapipe as mp
import numpy as np

# ---------- 关键点 & 阈值（已按你标定） ----------
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

STEPS = {"1": "睁眼基线", "2": "闭眼", "3": "哈欠", "4": "低头", "5": "自然眨眼"}


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


# ---------- 录入测试者信息 ----------
def ask(prompt, default):
    v = input(prompt).strip()
    return v if v else default


print("=== 测试者信息录入（直接回车用默认值）===")
num = ask("测试者编号（如 1）: ", "1")
gender = ask("性别（男/女）: ", "未知")
height = ask("身高 cm（如 175）: ", "0")
age = ask("年龄（如 22）: ", "0")
glasses_raw = ask("是否佩戴眼镜（是/否）: ", "否")
glasses = "戴眼镜" if glasses_raw in ("是", "y", "Y", "yes", "Yes") else "不戴眼镜"

tester_dir_name = f"测试者{num}号_{gender}_{height}cm_{age}岁_{glasses}"
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", tester_dir_name)
os.makedirs(out_dir, exist_ok=True)

info_lines = [
    f"测试者编号: {num}",
    f"性别: {gender}",
    f"身高: {height} cm",
    f"年龄: {age} 岁",
    f"佩戴眼镜: {glasses}",
    f"采集时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "各步骤记录:",
]

# ---------- 摄像头 ----------
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("打不开摄像头，改脚本里的摄像头编号再试")
    exit(1)

session_start_time = time.time()
print("\n摄像头就绪。操作：")
print("  1~5 开始步骤    s/空格 停止当前步骤    q/ESC 结束")
print("  步骤表: " + "  |  ".join(f"{k}={v}" for k, v in STEPS.items()))

# ---------- 状态变量 ----------
active_step = None
step_file = None
step_writer = None
step_start_t = 0.0
step_frame = 0
step_blink_start = 0
step_yawn_start = 0
stats = None

close_counter = 0
yawn_counter = 0
nod_counter = 0
prev_closed = False
prev_yawn = False
blink_count = 0
yawn_count = 0
closed_history = deque(maxlen=PERCLOS_WINDOW)
frame_count = 0


def new_stats():
    return dict(n=0, ear_sum=0.0, ear_min=9e9, ear_max=-9e9,
                ear_open_sum=0.0, ear_open_n=0, ear_closed_sum=0.0, ear_closed_n=0,
                mar_max=-9e9, pitch_sum=0.0, pitch_min=9e9,
                perclos_sum=0.0, perclos_max=-9e9)


def start_step(name):
    global active_step, step_file, step_writer, step_start_t, step_frame
    global step_blink_start, step_yawn_start, stats
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"{name}_{stamp}.csv")
    step_file = open(csv_path, "w", newline="", encoding="utf-8-sig")
    step_writer = csv.writer(step_file)
    step_writer.writerow(["t_step", "t_global", "frame", "ear_left", "ear_right", "ear_avg",
                          "mar", "pitch", "perclos", "closed", "yawning", "head_down",
                          "blinks", "yawns"])
    active_step = name
    step_start_t = time.time()
    step_frame = 0
    step_blink_start = blink_count
    step_yawn_start = yawn_count
    stats = new_stats()
    print(f"[开始] 步骤「{name}」")


def end_step():
    global active_step, step_file, step_writer
    if active_step is None:
        return
    step_file.close()
    dur = time.time() - step_start_t
    s = stats
    n = s["n"]
    step_blinks = blink_count - step_blink_start
    step_yawns = yawn_count - step_yawn_start
    print(f"\n==== 步骤「{active_step}」摘要 ====")
    print(f"时长 {dur:.1f}s  帧 {n}   眨眼 {step_blinks} 次  哈欠 {step_yawns} 次")
    if n > 0:
        print(f"EAR: min={s['ear_min']:.3f} mean={s['ear_sum']/n:.3f} max={s['ear_max']:.3f}")
        if s["ear_open_n"]:
            print(f"  睁眼平均 EAR {s['ear_open_sum']/s['ear_open_n']:.3f} (n={s['ear_open_n']})")
        if s["ear_closed_n"]:
            print(f"  闭眼平均 EAR {s['ear_closed_sum']/s['ear_closed_n']:.3f} (n={s['ear_closed_n']})")
        print(f"MAR 最大 {s['mar_max']:.3f}   pitch mean={s['pitch_sum']/n:.3f} min={s['pitch_min']:.3f}")
        print(f"PERCLOS 平均 {s['perclos_sum']/n:.1%} 峰值 {s['perclos_max']:.1%}")
    info_lines.append(f"  {active_step}: {dur:.1f}s 帧={n} 眨眼={step_blinks} 哈欠={step_yawns}")
    active_step = None
    step_writer = None
    step_file = None


# ---------- 主循环 ----------
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    face_ok = False
    if result.multi_face_landmarks:
        face_ok = True
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

        # 画面：实时数值 + 当前步骤状态
        cv2.putText(frame, f"EAR {ear:.3f}  MAR {mar:.3f}  pitch {pitch:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if active_step is not None:
            cv2.putText(frame, f"步骤: {active_step}  已录 {time.time()-step_start_t:.0f}s",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "no face", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 步骤菜单
    y = 90
    for k, v in STEPS.items():
        mark = ">" if active_step == v else " "
        cv2.putText(frame, f"{mark}{k} {v}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 22
    cv2.putText(frame, "s=停止  q=结束", (10, y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("collect", frame)

    # 按键处理
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    if 0 <= key < 256 and chr(key) in STEPS:
        if active_step is not None:
            end_step()
        start_step(STEPS[chr(key)])
    elif key == ord('s') or key == 32:
        end_step()
    if cv2.getWindowProperty("collect", cv2.WND_PROP_VISIBLE) < 1:
        break

    # 只有步骤进行中才写 CSV（本帧数据）
    if active_step is not None and face_ok:
        step_frame += 1
        t_step = time.time() - step_start_t
        t_global = time.time() - session_start_time
        step_writer.writerow([round(t_step, 3), round(t_global, 3), step_frame,
                              round(ear_l, 4), round(ear_r, 4), round(ear, 4),
                              round(mar, 4), round(pitch, 4), round(perclos, 4),
                              int(closed), int(yawning), int(head_down),
                              blink_count, yawn_count])
        s = stats
        s["n"] += 1
        s["ear_sum"] += ear; s["ear_min"] = min(s["ear_min"], ear); s["ear_max"] = max(s["ear_max"], ear)
        if closed:
            s["ear_closed_sum"] += ear; s["ear_closed_n"] += 1
        else:
            s["ear_open_sum"] += ear; s["ear_open_n"] += 1
        s["mar_max"] = max(s["mar_max"], mar)
        s["pitch_sum"] += pitch; s["pitch_min"] = min(s["pitch_min"], pitch)
        s["perclos_sum"] += perclos; s["perclos_max"] = max(s["perclos_max"], perclos)

    frame_count += 1
    if frame_count % 10 == 0 and step_file is not None:
        step_file.flush()

# ---------- 收尾 ----------
end_step()
cap.release()
cv2.destroyAllWindows()
face_mesh.close()

info_lines.append("")
info_lines.append(f"总时长 {time.time()-session_start_time:.1f}s，共 {frame_count} 帧")
with open(os.path.join(out_dir, "_info.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(info_lines) + "\n")
print("\n本次采集结束，溯源信息已保存到:")
print(" ", os.path.join(out_dir, "_info.txt"))
