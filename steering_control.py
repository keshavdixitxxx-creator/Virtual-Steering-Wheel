"""
Virtual Steering Wheel (Tasks API version)
-------------------------------------------
Use your two hands (held apart like on a steering wheel) in front of your
webcam to control a racing game. Tilt your hands like turning a wheel to
steer left/right. Bring your hands down/together to brake, and hold them
up normally to accelerate.

This version uses MediaPipe's newer "Tasks" API (HandLandmarker) instead of
the older `mediapipe.solutions` API. Several recent mediapipe pip releases
(0.10.3x) ship broken/missing `mediapipe.solutions` on some platforms, so
the Tasks API is the more reliable choice right now.

On first run, this script will automatically download the small hand
landmark model file (~10MB) into the same folder.

Controls sent to the OS:
    W  -> accelerate
    S  -> brake
    A  -> steer left
    D  -> steer right

Run:
    python steering_control.py

Press 'q' with the preview window focused to quit.
"""

import math
import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import keyboard  # pip install keyboard

try:
    import pygetwindow as gw  # pip install pygetwindow
    HAS_WINDOW_DETECTION = True
except Exception:
    HAS_WINDOW_DETECTION = False

# --------------------------------------------------------------------------
# Config - tweak these to taste
# --------------------------------------------------------------------------
CAM_INDEX = 0
STEER_DEADZONE_DEG = 8        # angle within this range = "straight"
STEER_FULL_DEG = 45           # angle at/beyond this = fully committed turn
SMOOTHING = 0.35              # 0 = no smoothing, closer to 1 = smoother/slower
KEY_ACCELERATE = ("w", "up")     # send both - whichever the game listens for will work
KEY_BRAKE = ("s", "down")
KEY_LEFT = ("a", "left")
KEY_RIGHT = ("d", "right")

# Only send key presses when the foreground (focused) window's title
# contains any of these substrings (case-insensitive). Add more if your
# browser tab title differs, e.g. your browser name or "CrazyGames".
TARGET_WINDOW_KEYWORDS = ["racing limits", "crazygames", "crazy games"]
REQUIRE_WINDOW_FOCUS = True  # set False to always send keys regardless of focus


def foreground_window_title():
    if not HAS_WINDOW_DETECTION:
        return None
    try:
        win = gw.getActiveWindow()
        return win.title if win else None
    except Exception:
        return None


def is_target_window_focused():
    if not REQUIRE_WINDOW_FOCUS:
        return True
    title = foreground_window_title()
    if title is None:
        # If we can't detect windows (e.g. pygetwindow missing/unsupported),
        # fail open so the script still works, just without auto-guarding.
        return True
    title_lower = title.lower()
    return any(k in title_lower for k in TARGET_WINDOW_KEYWORDS)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (first run only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded to", MODEL_PATH)


class KeyState:
    """Tracks which keys are currently 'held down' so we don't spam press/release.
    Accepts either a single key name or a tuple/list of key names (all are
    held/released together, so you can send e.g. both 'w' and 'up' for
    maximum compatibility with whatever the game actually listens for)."""

    def __init__(self):
        self._held = set()

    @staticmethod
    def _as_list(key):
        return [key] if isinstance(key, str) else list(key)

    def hold(self, key):
        for k in self._as_list(key):
            if k not in self._held:
                keyboard.press(k)
                self._held.add(k)

    def release(self, key):
        for k in self._as_list(key):
            if k in self._held:
                keyboard.release(k)
                self._held.discard(k)

    def release_all(self):
        for key in list(self._held):
            keyboard.release(key)
            self._held.discard(key)


def smooth(prev, new, factor):
    if prev is None:
        return new
    return prev * factor + new * (1 - factor)


def hand_center_px(hand_landmarks, w, h):
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]
    return (sum(xs) / len(xs) * w, sum(ys) / len(ys) * h)


FINGER_TIPS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
WRIST = 0
THUMB_MCP = 2
THUMB_TIP = 4
MIDDLE_MCP = 9
YELLOW = (0, 255, 255)   # BGR
CYAN = (255, 255, 0)     # BGR
GRAY = (170, 170, 170)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

# Thumb-gesture pedal control:
#   Thumb standing up, roughly perpendicular to the back of the fist -> BRAKE
#   Thumb laying flat/tucked, roughly parallel to the back of the fist -> GAS
# Angle is measured between the thumb (MCP->TIP) and the hand's own axis
# (WRIST->MIDDLE_MCP), so it works regardless of how you're holding your
# hand toward the camera.
THUMB_PARALLEL_MAX_DEG = 40   # <= this (or >= 180-this) => thumb laying flat => GAS
THUMB_PERP_MIN_DEG = 60       # between this and (180-this) => thumb standing => BRAKE
THUMB_PERP_MAX_DEG = 120


def thumb_gesture_state(hand_landmarks):
    """Returns ('ACCEL' | 'BRAKE' | None, angle_degrees)."""

    def vec(a, b):
        return (hand_landmarks[b].x - hand_landmarks[a].x,
                 hand_landmarks[b].y - hand_landmarks[a].y)

    hand_axis = vec(WRIST, MIDDLE_MCP)
    thumb_axis = vec(THUMB_MCP, THUMB_TIP)

    mag_h = math.hypot(*hand_axis)
    mag_t = math.hypot(*thumb_axis)
    if mag_h == 0 or mag_t == 0:
        return None, 0.0

    dot = hand_axis[0] * thumb_axis[0] + hand_axis[1] * thumb_axis[1]
    cos_angle = max(-1.0, min(1.0, dot / (mag_h * mag_t)))
    angle = math.degrees(math.acos(cos_angle))  # 0..180

    if angle <= THUMB_PARALLEL_MAX_DEG or angle >= (180 - THUMB_PARALLEL_MAX_DEG):
        return "ACCEL", angle
    if THUMB_PERP_MIN_DEG <= angle <= THUMB_PERP_MAX_DEG:
        return "BRAKE", angle
    return None, angle  # ambiguous transition zone, ignore


def draw_dashboard(frame, angle, accel, brake, fps, hands_detected, window_focused, thumb_angle=None):
    h, w = frame.shape[:2]

    # Top-left: tracking status
    cv2.putText(frame, f"TRACKING {hands_detected}/2", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, YELLOW, 2)

    # Top-right: FPS
    fps_text = f"{fps:.0f} FPS"
    (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(frame, fps_text, (w - tw - 10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Bottom-center: big angle readout (draw manual degree-circle since
    # OpenCV's Hershey font has no ° glyph)
    angle_text = f"{angle:+.0f}"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3
    (tw, th), _ = cv2.getTextSize(angle_text, font, scale, thick)
    text_x = w // 2 - (tw + 14) // 2
    text_y = h - 20
    cv2.putText(frame, angle_text, (text_x, text_y), font, scale, YELLOW, thick)
    deg_cx = text_x + tw + 10
    deg_cy = text_y - th + 2
    cv2.circle(frame, (deg_cx, deg_cy), 4, YELLOW, 2)

    # Bottom-left / bottom-right: LEFT / RIGHT indicators, active one lit up
    going_left = angle < -STEER_DEADZONE_DEG
    going_right = angle > STEER_DEADZONE_DEG
    cv2.putText(frame, "< LEFT" if not going_left else "<< LEFT", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, CYAN if going_left else GRAY, 2)
    right_text = "RIGHT >" if not going_right else "RIGHT >>"
    (tw, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(frame, right_text, (w - tw - 10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, CYAN if going_right else GRAY, 2)

    # Small accel/brake status under the tracking label
    status = f"ACCEL {'ON' if accel else 'off'}  BRAKE {'ON' if brake else 'off'}"
    cv2.putText(frame, status, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Thumb-gesture readout
    if thumb_angle is not None:
        gesture = "STANDING (BRAKE)" if brake else "LAYING (GAS)" if accel else "..."
        gcolor = RED if brake else GREEN if accel else GRAY
        thumb_text = f"THUMB {thumb_angle:.0f} - {gesture}"
        cv2.putText(frame, thumb_text, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gcolor, 1)

    # Focus / connection status, centered near the top
    if window_focused:
        conn_text = "GAME CONNECTED"
        conn_color = (0, 255, 0)
    else:
        conn_text = "WAITING FOR GAME WINDOW..."
        conn_color = (0, 0, 255)
    (tw, _), _ = cv2.getTextSize(conn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, conn_text, (w // 2 - tw // 2, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, conn_color, 1)


def draw_landmarks(frame, hand_landmarks, w, h):
    """Yellow fan/mesh style: wrist connected to each fingertip, fingertips
    connected to their neighbors, matching a lightweight tracking-overlay look."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    wrist = pts[WRIST]
    tips = [pts[i] for i in FINGER_TIPS]

    for tip in tips:
        cv2.line(frame, wrist, tip, YELLOW, 1)
    for a, b in zip(tips, tips[1:]):
        cv2.line(frame, a, b, YELLOW, 1)

    for x, y in pts:
        cv2.circle(frame, (x, y), 3, YELLOW, -1)

    # Label the thumb tip, like a tracked-point readout
    tx, ty = pts[FINGER_TIPS[0]]
    cv2.putText(frame, "THUMB_TIP", (tx + 8, ty - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW, 1)


def main():
    ensure_model()

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open webcam. Check CAM_INDEX or camera permissions.")
        return

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    keys = KeyState()
    smoothed_angle = None
    start_time = time.time()
    prev_frame_time = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            angle = 0.0
            accelerate = False
            brake = False
            thumb_angle_display = None
            pedal_state = None  # 'ACCEL' | 'BRAKE' | None

            hands_landmarks = result.hand_landmarks  # list of lists of landmarks
            hands_detected = len(hands_landmarks) if hands_landmarks else 0
            if hands_landmarks:
                for hl in hands_landmarks:
                    draw_landmarks(frame, hl, w, h)

                # --- Pedals: thumb gesture on any detected hand ---
                # Brake takes priority for safety - if ANY hand shows a
                # standing thumb, brake overrides an accelerate reading.
                for hl in hands_landmarks:
                    state, th_angle = thumb_gesture_state(hl)
                    if state == "BRAKE":
                        pedal_state = "BRAKE"
                        thumb_angle_display = th_angle
                        break
                    elif state == "ACCEL" and pedal_state is None:
                        pedal_state = "ACCEL"
                        thumb_angle_display = th_angle

                accelerate = pedal_state == "ACCEL"
                brake = pedal_state == "BRAKE"

                # --- Steering: two-hand tilt, or single-hand x-position fallback ---
                centers = [hand_center_px(hl, w, h) for hl in hands_landmarks]

                if len(centers) == 2:
                    (x1, y1), (x2, y2) = centers
                    if x1 > x2:
                        (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
                    dx, dy = (x2 - x1), (y2 - y1)
                    raw_angle = math.degrees(math.atan2(dy, dx))
                    angle = max(-STEER_FULL_DEG, min(STEER_FULL_DEG, raw_angle))

                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)

                elif len(centers) == 1:
                    x, y = centers[0]
                    norm_x = (x - w / 2) / (w / 2)
                    angle = max(-STEER_FULL_DEG, min(STEER_FULL_DEG, norm_x * STEER_FULL_DEG))
            else:
                angle = 0.0

            smoothed_angle = smooth(smoothed_angle, angle, SMOOTHING)

            window_focused = is_target_window_focused()

            if window_focused:
                if smoothed_angle < -STEER_DEADZONE_DEG:
                    keys.hold(KEY_LEFT)
                    keys.release(KEY_RIGHT)
                elif smoothed_angle > STEER_DEADZONE_DEG:
                    keys.hold(KEY_RIGHT)
                    keys.release(KEY_LEFT)
                else:
                    keys.release(KEY_LEFT)
                    keys.release(KEY_RIGHT)

                if brake:
                    keys.hold(KEY_BRAKE)
                    keys.release(KEY_ACCELERATE)
                elif accelerate:
                    keys.hold(KEY_ACCELERATE)
                    keys.release(KEY_BRAKE)
                else:
                    keys.release(KEY_ACCELERATE)
                    keys.release(KEY_BRAKE)
            else:
                # Game window isn't focused (e.g. you tabbed away) -
                # release everything so keys don't leak into other apps.
                keys.release_all()

            now = time.time()
            frame_dt = now - prev_frame_time
            prev_frame_time = now
            if frame_dt > 0:
                fps = smooth(fps, 1.0 / frame_dt, 0.9) if fps else 1.0 / frame_dt

            draw_dashboard(frame, smoothed_angle, accelerate, brake, fps, hands_detected, window_focused, thumb_angle_display)
            cv2.putText(frame, "Press 'q' to quit", (10, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

            cv2.imshow("Virtual Steering Wheel", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        keys.release_all()
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()