# 疲劳驾驶检测系统 Driver Fatigue Detection System

> 基于 MediaPipe 的实时驾驶员疲劳监测 Demo：通过人脸关键点检测**闭眼、打哈欠、低头**三种疲劳特征，实时计算 PERCLOS 指标，检测到疲劳时红屏 + 声音报警。

A real-time driver fatigue monitoring demo built with MediaPipe Face Mesh. It detects eye closure, yawning and head-down (nodding), computes the industry-standard PERCLOS metric, and raises a visual + audio alarm on drowsiness.

## 项目背景

疲劳驾驶是道路交通事故的重要诱因。本项目是「AI + 智能驾驶」方向的入门级落地 Demo，聚焦驾驶员监控系统（DMS, Driver Monitoring System）中的**疲劳检测**核心功能。

## 功能特性

- 人脸 468 关键点实时检测（MediaPipe Face Mesh）
- **闭眼检测**：眼睛长宽比 EAR + 连续帧防抖
- **打哈欠检测**：嘴巴长宽比 MAR + 持续时间过滤
- **低头（点头）检测**：鼻尖-下巴 / 脸宽 比值
- **PERCLOS** 疲劳指标（行业主流标准）
- 综合疲劳判定 + 红屏闪烁 + 声音报警
- 多测试者结构化数据采集工具

## 原理

| 信号 | 指标 | 疲劳时的变化 |
|---|---|---|
| 眼睛 | EAR（眼睛长宽比） | 闭眼 → EAR 下降 |
| 嘴巴 | MAR（嘴巴长宽比） | 打哈欠 → MAR 升高 |
| 头部 | pitch（鼻-下巴 / 脸宽） | 低头 → pitch 下降 |

## 环境要求

- Python 3.12
- 一个 USB 摄像头

## 安装

```bash
pip install -r requirements.txt
```

> 注意：本项目只用一个 OpenCV（contrib 版）。如果你机器上之前装过 `opencv-python`，请先卸载，避免两个 OpenCV 冲突：`pip uninstall -y opencv-python`。

## 快速开始

```bash
# 一键体验完整功能（疲劳检测 + 报警）
python stage4_alarm.py
```

按 `q` 退出（需先点击视频窗口使其获得焦点）。

## 脚本说明

| 脚本 | 用途 |
|---|---|
| `stage1_facemesh.py` | 入门：摄像头 + 468 关键点 |
| `stage2_ear.py` | 闭眼检测（EAR） |
| `stage3_fatigue_signals.py` | 三信号同屏（眼 / 嘴 / 头） |
| `stage4_alarm.py` | 成品：疲劳判定 + 报警 |
| `collect.py` | 多测试者结构化数据采集 |
| `env_check.py` | 环境体检脚本 |

## 数据采集

`collect.py` 支持分步骤、多测试者的结构化采集，输出到 `data/测试者X号_性别_身高_年龄_眼镜/`，每步一个 CSV，并生成 `_info.txt` 溯源记录。

> ⚠️ 采集数据含个人隐私（性别 / 年龄 / 身高 / 是否戴眼镜），已通过 `.gitignore` 排除，请勿上传到公开仓库。

## 演示

（此处放一张运行截图或 GIF）

## 许可

[MIT License](LICENSE)
