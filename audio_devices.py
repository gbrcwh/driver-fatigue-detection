# -*- coding: utf-8 -*-
"""列出 pygame(SDL) 能看到的输出音频设备，帮判断声音被路由到了哪里"""
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame

pygame.mixer.init()   # 先初始化音频系统，才能枚举设备
print("mixer 初始化:", pygame.mixer.get_init())

import pygame._sdl2.audio as sdl_audio
devices = sdl_audio.get_audio_device_names(iscapture=False)
print("共", len(devices), "个输出设备:")
for i, name in enumerate(devices):
    print(f"  [{i}] {repr(name)}")
