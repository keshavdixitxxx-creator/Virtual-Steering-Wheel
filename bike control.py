"""
Virtual Bike Controller (upgraded from the working car steering script)
--------------------------------------------------------------------------
Same proven foundation as the car version (HandLandmarker Tasks API,
window-focus auto-connect, key-holding mechanism) - only the gesture
logic has changed for bike games like Traffic Rider:

    Both hands FISTS, held roughly level   -> RUNNING (accelerate)
    Either hand OPEN PALM facing camera    -> BRAKE (always overrides)
    Both hands FISTS, tilted to one side   -> TURN toward the lower side
                                               (keeps accelerating while turning)

On first run, this script will automatically download the small hand
landmark model file (~10MB) into the same folder.

Controls sent to the OS (sends BOTH key sets at once for compatibility):
    W / UP     -> accelerate / running
    S / DOWN   -> brake
    A / LEFT   -> turn left
    D / RIGHT  -> turn right

Run:
    python bike_control.py

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
TILT_DEADZONE_DEG = 10        # tilt within this range = "level" -> just accelerate
TILT_FULL_DEG = 45            # tilt at/beyond this = fully committed turn
SMOOTHING = 0.35              # 0 = no smoothing, closer to 1 = smoother/slower
OPEN_HAND_FINGER_THRESHOLD = 3  # >= this many extended fingers counts as "open palm"
EXTENDED_RATIO = 1.2          # fingertip must be this much farther from wrist than its MCP to count as "extended"

KEY_ACCELERATE = ("w", "up")     # send both - whichever the game listens for will work
KEY_BRAKE = ("s", "down")
KEY_LEFT = ("a", "left")
KEY_RIGHT = ("d", "right")

# Only send key presses when the foreground (focused) window's title
# contains any of these substrings (case-insensitive). Add more if your
# browser tab title differs, e.g. your browser name or "CrazyGames".
TARGET_WINDOW_KEYWORDS = ["traffic rider", "racing limits", "crazygames", "crazy games", "bluestacks"]
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


WRIST = 0
FINGER_TIPS = [4, 8, 12, 16, 20]           # thumb, index, middle, ring, pinky
NON_THUMB_FINGERS = [(5, 8), (9, 12), (13, 16), (17, 20)]  # (MCP, TIP) pairs
MIDDLE_MCP = 9
YELLOW = (0, 255, 255)   # BGR
CYAN = (255, 255, 0)     # BGR
GRAY = (170, 170, 170)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def is_hand_open(hand_landmarks):
    """Counts extended (non-thumb) fingers. Returns (is_open: bool, extended_count: int)."""
    wrist = hand_landmarks[WRIST]

    def dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    extended = 0
    for mcp_idx, tip_idx in NON_THUMB_FINGERS:
        mcp = hand_landmarks[mcp_idx]
        tip = hand_landmarks[tip_idx]
        if dist(wrist, tip) > dist(wrist, mcp) * EXTENDED_RATIO:
            extended += 1

    return extended >= OPEN_HAND_FINGER_THRESHOLD, extended


def hand_axis_angle(hand_landmarks):
    """Angle (degrees) of the WRIST->MIDDLE_MCP axis relative to vertical
    (straight up on screen = 0). Positive = tilted right, negative = tilted left."""
    wrist = hand_landmarks[WRIST]
    mid = hand_landmarks[MIDDLE_MCP]
    dx = mid.x - wrist.x
    dy = mid.y - wrist.y
    return math.degrees(math.atan2(dx, -dy))


def draw_dashboard(frame, mode, tilt_angle, fps, hands_detected, window_focused, open_counts):
    h, w = frame.shape[:2]

    # Top-left: tracking status
    cv2.putText(frame, f"TRACKING {hands_detected}/2", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, YELLOW, 2)

    # Top-right: FPS
    fps_text = f"{fps:.0f} FPS"
    (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(frame, fps_text, (w - tw - 10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Mode banner, big and centered
    mode_colors = {
        "RUNNING": GREEN,
        "BRAKE": RED,
        "TURN LEFT": CYAN,
        "TURN RIGHT": CYAN,
        "IDLE": GRAY,
    }
    color = mode_colors.get(mode, GRAY)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
    (tw, th), _ = cv2.getTextSize(mode, font, scale, thick)
    cv2.putText(frame, mode, (w // 2 - tw // 2, h - 20), font, scale, color, thick)

    # Tilt angle readout, bottom-left
    cv2.putText(frame, f"TILT {tilt_angle:+.0f}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1)

    # Open-finger counts, under tracking label
    if open_counts:
        counts_text = "FINGERS " + " / ".join(str(c) for c in open_counts)
        cv2.putText(frame, counts_text, (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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


def draw_landmarks(frame, hand_landmarks, w, h, open_hand):
    """Yellow fan/mesh style: wrist connected to each fingertip, fingertips
    connected to their neighbors. Turns cyan when the hand is classified open."""
    color = CYAN if open_hand else YELLOW
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    wrist = pts[WRIST]
    tips = [pts[i] for i in FINGER_TIPS]

    for tip in tips:
        cv2.line(frame, wrist, tip, color, 1)
    for a, b in zip(tips, tips[1:]):
        cv2.line(frame, a, b, color, 1)

    for x, y in pts:
        cv2.circle(frame, (x, y), 3, color, -1)


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
    smoothed_tilt = None
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

            tilt_angle = 0.0
            mode = "IDLE"
            open_counts = []

            hands_landmarks = result.hand_landmarks  # list of lists of landmarks
            hands_detected = len(hands_landmarks) if hands_landmarks else 0

            if hands_landmarks:
                open_flags = []
                for hl in hands_landmarks:
                    is_open, count = is_hand_open(hl)
                    open_flags.append(is_open)
                    open_counts.append(count)
                    draw_landmarks(frame, hl, w, h, is_open)

                any_open = any(open_flags)

                if any_open:
                    # --- BRAKE: any open palm overrides everything ---
                    mode = "BRAKE"
                    tilt_angle = 0.0
                else:
                    # --- Fists: compute tilt for turning, else running ---
                    if hands_detected == 2:
                        (x1, y1), (x2, y2) = [hand_center_px(hl, w, h) for hl in hands_landmarks]
                        if x1 > x2:
                            (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
                        dx, dy = (x2 - x1), (y2 - y1)
                        raw_tilt = math.degrees(math.atan2(dy, dx))
                        tilt_angle = max(-TILT_FULL_DEG, min(TILT_FULL_DEG, raw_tilt))
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)
                    else:
                        raw_tilt = hand_axis_angle(hands_landmarks[0])
                        tilt_angle = max(-TILT_FULL_DEG, min(TILT_FULL_DEG, raw_tilt))

                    if abs(tilt_angle) < TILT_DEADZONE_DEG:
                        mode = "RUNNING"
                    elif tilt_angle < 0:
                        mode = "TURN LEFT"
                    else:
                        mode = "TURN RIGHT"

            smoothed_tilt = smooth(smoothed_tilt, tilt_angle, SMOOTHING)

            window_focused = is_target_window_focused()

            if window_focused:
                if mode == "BRAKE":
                    keys.hold(KEY_BRAKE)
                    keys.release(KEY_ACCELERATE)
                    keys.release(KEY_LEFT)
                    keys.release(KEY_RIGHT)
                elif mode == "RUNNING":
                    keys.hold(KEY_ACCELERATE)
                    keys.release(KEY_BRAKE)
                    keys.release(KEY_LEFT)
                    keys.release(KEY_RIGHT)
                elif mode == "TURN LEFT":
                    keys.hold(KEY_ACCELERATE)
                    keys.hold(KEY_LEFT)
                    keys.release(KEY_BRAKE)
                    keys.release(KEY_RIGHT)
                elif mode == "TURN RIGHT":
                    keys.hold(KEY_ACCELERATE)
                    keys.hold(KEY_RIGHT)
                    keys.release(KEY_BRAKE)
                    keys.release(KEY_LEFT)
                else:  # IDLE - no hands visible
                    keys.release_all()
            else:
                # Game window isn't focused (e.g. you tabbed away) -
                # release everything so keys don't leak into other apps.
                keys.release_all()

            now = time.time()
            frame_dt = now - prev_frame_time
            prev_frame_time = now
            if frame_dt > 0:
                fps = smooth(fps, 1.0 / frame_dt, 0.9) if fps else 1.0 / frame_dt

            draw_dashboard(frame, mode, smoothed_tilt, fps, hands_detected, window_focused, open_counts)
            cv2.putText(frame, "Press 'q' to quit", (10, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

            cv2.imshow("Virtual Bike Controller", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        keys.release_all()
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()