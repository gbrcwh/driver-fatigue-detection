# -*- coding: utf-8 -*-
"""OpenCV 画面上显示中文的小工具。

cv2.putText 只支持 ASCII，遇到中文会显示成问号「?」。
本模块用 Pillow 把中文画到画面上去，用法：
    from cn_text import draw_texts
    draw_texts(frame, [("闭眼", 10, 30, 20, (0, 0, 255)), ...])
每项是 (文字, x, y, 字号, BGR颜色)，x/y 是文字左上角坐标。
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CACHE = {}


def _font(size):
    """按字号取一个中文字体（缓存起来，避免每帧重复读字体文件）"""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    for path in (
        r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑
        r"C:\Windows\Fonts\simhei.ttf",   # 黑体
        r"C:\Windows\Fonts\simsun.ttc",   # 宋体
    ):
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
    _FONT_CACHE[size] = font
    return font


def draw_texts(img, items):
    """把一批中文文字一次画到 img（BGR ndarray）上，原地修改。

    items: [(text, x, y, size, color_bgr), ...]
    """
    if not items:
        return
    if _font(16) is None:  # 找不到中文字体就退回 cv2（中文会显示 ?，但不报错）
        for text, x, y, size, color in items:
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size / 22.0, color, 1)
        return
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for text, x, y, size, color in items:
        draw.text((x, y), text, font=_font(size), fill=(color[2], color[1], color[0]))
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
