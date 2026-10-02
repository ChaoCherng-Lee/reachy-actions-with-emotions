"""OpenCV window: camera view on the left, readout + virtual Reachy on the right."""

import math

import cv2
import numpy as np

from reachy_emotion import EMOTIONS, EMOTION_TO_LIGHT
from reachy_emotion.robot import POSES

PANEL_W = 360
BG = (32, 32, 32)
TEXT = (235, 235, 235)
MUTED = (150, 150, 150)
LIGHT_COLOURS = {"RED": (70, 70, 230), "AMBER": (0, 190, 255), "GREEN": (90, 200, 90), None: (120, 120, 120)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def render(frame, face, reading, feedback, mic_status, heard):
    h, w = frame.shape[:2]
    view = frame.copy()
    light = reading["light"] if reading else None

    if face:
        x, y, fw, fh = face[0]
        cv2.rectangle(view, (x, y), (x + fw, y + fh), LIGHT_COLOURS[light], 2)
    if heard:
        _caption(view, f'Heard: "{heard}"')

    panel = np.full((max(h, 540), PANEL_W, 3), BG, np.uint8)
    y = _traffic_light(panel, light, 20)
    y = _virtual_reachy(panel, POSES[light], LIGHT_COLOURS[light], y + 10)
    y = _emotion_bars(panel, reading, y + 10)
    y = _wrapped(panel, feedback or "Looking for a face...", 16, y + 14, TEXT, 0.5, PANEL_W - 32)

    sources = "+".join(reading["sources"]) if reading else "none"
    cv2.putText(panel, f"mic: {mic_status}   using: {sources}", (16, panel.shape[0] - 14), FONT, 0.42, MUTED, 1, cv2.LINE_AA)

    if panel.shape[0] > h:
        view = cv2.copyMakeBorder(view, 0, panel.shape[0] - h, 0, 0, cv2.BORDER_CONSTANT, value=BG)
    return np.hstack([view, panel])


def _traffic_light(img, light, top):
    for i, name in enumerate(("RED", "AMBER", "GREEN")):
        colour = LIGHT_COLOURS[name] if name == light else tuple(c // 4 for c in LIGHT_COLOURS[name])
        cv2.circle(img, (40 + i * 44, top + 20), 16, colour, -1, cv2.LINE_AA)
    label = light or "NO ONE"
    cv2.putText(img, label, (180, top + 28), FONT, 0.8, LIGHT_COLOURS[light], 2, cv2.LINE_AA)
    return top + 44


def _virtual_reachy(img, pose, colour, top):
    """Front view of a Reachy Mini, posed exactly as the real one would be"""
    cx = PANEL_W // 2
    body_top = top + 130
    # we see the robot from the front, so its positive roll tips the head to our left
    roll = math.radians(-pose["roll"])
    head = (cx, int(top + 80 - pose["z"] * 1.5))

    def to_panel(px, py):
        """head-frame point -> panel pixel, applying head roll"""
        c, s = math.cos(roll), math.sin(roll)
        return int(head[0] + px * c - py * s), int(head[1] + px * s + py * c)

    # body
    cv2.ellipse(img, (cx, body_top + 40), (70, 45), 0, 180, 360, (200, 200, 200), -1, cv2.LINE_AA)
    cv2.rectangle(img, (cx - 70, body_top + 40), (cx + 70, body_top + 55), (200, 200, 200), -1)
    cv2.line(img, (cx, body_top + 5), head, (170, 170, 170), 10, cv2.LINE_AA)

    # antennas are (right, left) in the SDK; facing us, the robot's right is on our left.
    # 0 = straight up, positive tips towards our right (matches the MuJoCo sim)
    for sign, angle in ((-1, pose["antennas"][0]), (1, pose["antennas"][1])):
        base = (sign * 30, -30)
        a = math.radians(angle)
        tip = (base[0] + 55 * math.sin(a), base[1] - 55 * math.cos(a))
        cv2.line(img, to_panel(*base), to_panel(*tip), colour, 4, cv2.LINE_AA)
        cv2.circle(img, to_panel(*tip), 6, colour, -1, cv2.LINE_AA)

    # head and eyes (pitch moves the gaze up/down)
    cv2.ellipse(img, head, (65, 40), -pose["roll"], 0, 360, (235, 235, 235), -1, cv2.LINE_AA)
    gaze = pose["pitch"] * 0.6
    for sign in (-1, 1):
        cv2.circle(img, to_panel(sign * 26, gaze), 15, (40, 40, 40), -1, cv2.LINE_AA)
        cv2.circle(img, to_panel(sign * 26 + 4, gaze - 4), 4, (220, 220, 220), -1, cv2.LINE_AA)
    return body_top + 60


def _emotion_bars(img, reading, top):
    for i, emotion in enumerate(EMOTIONS):
        y = top + i * 20
        score = reading["scores"][emotion] if reading else 0.0
        is_top = reading is not None and emotion == reading["emotion"]
        cv2.putText(img, emotion, (16, y + 12), FONT, 0.45, TEXT if is_top else MUTED, 1, cv2.LINE_AA)
        cv2.rectangle(img, (100, y + 2), (300, y + 14), (60, 60, 60), -1)
        cv2.rectangle(img, (100, y + 2), (100 + int(200 * score), y + 14), LIGHT_COLOURS[EMOTION_TO_LIGHT[emotion]], -1)
        cv2.putText(img, f"{score:.0%}", (308, y + 12), FONT, 0.4, MUTED, 1, cv2.LINE_AA)
    return top + len(EMOTIONS) * 20


def _wrapped(img, text, x, y, colour, scale, max_w):
    line = ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if cv2.getTextSize(trial, FONT, scale, 1)[0][0] > max_w and line:
            cv2.putText(img, line, (x, y), FONT, scale, colour, 1, cv2.LINE_AA)
            y, line = y + 22, word
        else:
            line = trial
    if line:
        cv2.putText(img, line, (x, y), FONT, scale, colour, 1, cv2.LINE_AA)
    return y + 22


def _caption(img, text):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 34), (w, h), (0, 0, 0), -1)
    cv2.putText(img, text[:90], (10, h - 12), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
