# Virtual Steering Wheel 🎮🖐️

Control racing games with your hands via webcam — no hardware wheel needed.
Hold your hands up like you're gripping a steering wheel, tilt them to turn,
and drop them down to brake.

This is the same idea as the "hand-tracking steering wheel" clip: MediaPipe
detects your hand landmarks, we compute the tilt angle between your two
hands, and that angle is mapped to `A` / `D` key presses (with `W` / `S` for
gas/brake).

## How it works

1. **OpenCV** grabs frames from your webcam.
2. **MediaPipe Hands** detects up to 2 hands and their 21 landmarks each.
3. We take the center point of each hand and compute the angle of the line
   connecting them relative to horizontal — just like the angle of a real
   steering wheel.
4. That angle is smoothed (to avoid jitter) and mapped to keyboard input:
   - Tilt left → hold `A`
   - Tilt right → hold `D`
   - Hands roughly level → release both (go straight)
   - Hands raised → hold `W` (accelerate)
   - Hands dropped low → hold `S` (brake)
5. The **`keyboard`** library sends these as real, system-wide key events,
   so any game listening for keyboard input (e.g. browser racing games,
   most PC racing games) will respond.

A single-hand fallback mode is included too — if only one hand is visible,
its horizontal position steers and vertical position controls gas/brake.

## Setup

```bash
pip install -r requirements.txt
```

> **Linux users:** the `keyboard` library needs root privileges to inject
> global key events, so run with `sudo python3 steering_control.py` (or add
> your user to the `input` group).
>
> **macOS users:** grant your terminal/IDE Accessibility + Input Monitoring
> permissions in System Settings → Privacy & Security, or key events will be
> silently ignored.
>
> **Windows users:** should just work. If a specific game doesn't respond to
> the `keyboard` library's synthetic key presses (some games only accept
> DirectInput), swap it for `pydirectinput` — see the note below.

## Run

```bash
python steering_control.py
```

1. Point your webcam at yourself, sit back far enough that both hands are
   visible in frame.
2. Raise both hands like you're holding a wheel — this starts "accelerate".
3. Tilt both hands together (like turning a wheel) to steer left/right.
4. Drop your hands down near your lap to brake.
5. Press `q` in the preview window to quit (this also releases any held keys).

Open your racing game in another window so it has keyboard focus — the
script sends OS-level key events, it doesn't need to be in the foreground,
but the *game* does need focus to receive them.

## Run it side-by-side with the browser game (Windows 11)

To play a browser racing game like Racing Limits on CrazyGames while
watching the camera preview:

1. Open the game in your browser and let it load.
2. Snap the browser window to one half of the screen: click and hold its
   title bar, drag to the left or right edge (or press `Win + Left` /
   `Win + Right`).
3. Run `python steering_control.py` — snap its preview window to the other
   half the same way.
4. Click into the browser/game so it has focus, then raise your hands to
   start driving.

### Auto-connect to the game window

The script only sends key presses when the **focused window's title**
matches one of `TARGET_WINDOW_KEYWORDS` at the top of `steering_control.py`
(by default: `"racing limits"`, `"crazygames"`, `"crazy games"`). This means:

- If you tab away to another app, keys stop being sent automatically — no
  more steering leaking into your code editor or chat app.
- The HUD shows **`GAME CONNECTED`** (green) when your active window
  matches, or **`WAITING FOR GAME WINDOW...`** (red) when it doesn't.
- If your browser's tab title doesn't contain those words, add whatever
  your window title actually shows (e.g. your browser's name, or just the
  site name) to the `TARGET_WINDOW_KEYWORDS` list.
- Set `REQUIRE_WINDOW_FOCUS = False` if you'd rather disable this and just
  always send keys (matches the old behavior).

This needs the `pygetwindow` package (already in `requirements.txt`), which
reads window titles via the Windows API — it only works on Windows.



All the important knobs are at the top of `steering_control.py`:

| Setting | What it does |
|---|---|
| `STEER_DEADZONE_DEG` | How many degrees of tilt count as "straight" |
| `STEER_FULL_DEG` | Angle at which steering is considered fully committed |
| `SMOOTHING` | Higher = smoother but slower to react (0–1) |
| `BRAKE_HAND_DROP_RATIO` | How far down your hands must drop to trigger braking |
| `KEY_ACCELERATE` / `KEY_BRAKE` / `KEY_LEFT` / `KEY_RIGHT` | Remap keys for a different game's control scheme |

## If your game doesn't respond to key presses

Some games (especially ones using DirectInput on Windows) ignore the
synthetic events from the `keyboard` library. Swap it for `pydirectinput`:

```bash
pip install pydirectinput
```

```python
import pydirectinput as keyboard_backend
# replace keyboard.press(key) / keyboard.release(key) calls with
# keyboard_backend.keyDown(key) / keyboard_backend.keyUp(key)
```

## Troubleshooting

- **No camera window appears** — check `CAM_INDEX` (try `1` if you have
  multiple cameras), and make sure no other app is using the webcam.
- **Laggy / jittery detection** — lower your camera resolution, ensure good
  lighting, or increase `SMOOTHING`.
- **Keys not reaching the game** — see the permissions notes above; this is
  almost always an OS permission issue, not a bug in the script.
