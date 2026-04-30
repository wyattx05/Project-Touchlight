"""
Project Touchlight

Setup:
1. Create and activate a virtual environment:
   python3 -m venv .venv
   source .venv/bin/activate
2. Install dependencies:
   pip install -r requirements.txt
3. Grant Accessibility permissions to your terminal or Python app in:
   System Settings -> Privacy & Security -> Accessibility
4. Choose one of two startup modes:
   - Continuity Camera touchscreen mode: use your iPhone camera as a gesture-driven touchscreen.
   - MacBook camera air-mouse mode: use the built-in camera to move the
     mouse with your finger and pinch to click.
5. For Continuity Camera, USB is preferred because it is typically lower latency than WiFi.
6. Run:
   python project_touchlight.py

Controls:
- Press `c` in the preview window to run calibration.
- Press `r` to delete the saved calibration and recalibrate.
- Press `q` to quit.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.error
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands
import pyautogui


# macOS mouse control should not trigger PyAutoGUI's corner failsafe while we
# are intentionally moving the cursor to the edges during calibration/use.
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


WINDOW_NAME = "Project Touchlight"
CORNER_NAMES = ["top_left", "top_right", "bottom_right", "bottom_left"]
LEGACY_CALIBRATION_FILE = Path(__file__).with_name("gesture_calibration.json")
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_LANDMARKER_MODEL_FILE = (
    Path.home()
    / "Library"
    / "Caches"
    / "Project Touchlight"
    / "hand_landmarker.task"
)

# Pinch threshold is measured in normalized image coordinates. Start slightly
# conservative so accidental near-pinches do not click too often.
PINCH_DOWN_THRESHOLD = 0.040
PINCH_UP_THRESHOLD = 0.060
CLICK_DEBOUNCE_SECONDS = 0.45
DRAG_HOLD_SECONDS = 0.20
TAP_FAST_MOVE_THRESHOLD = 0.018
TAP_STILL_THRESHOLD = 0.006
TAP_MAX_DRIFT = 0.025
TAP_SETTLE_SECONDS = 0.07
TAP_DEBOUNCE_SECONDS = 0.35

# Exponential moving average factor. Lower = smoother but slower.
SMOOTHING_ALPHA = 0.18
CAMERA_WARMUP_FRAMES = 20
DEFAULT_CAMERA_SCAN_LIMIT = 10
CONTINUITY_CAMERA_SCAN_LIMIT = 25
CAMERA_PROBE_FRAMES = 20
CAMERA_PROBE_DELAY_SECONDS = 0.03
CAMERA_PREVIEW_TIMEOUT_SECONDS = 8.0
ZOOM_DISTANCE_THRESHOLD = 0.012
ZOOM_SCROLL_SCALE = 2200
SCROLL_DELTA_THRESHOLD = 0.010
SCROLL_SCALE = 1700
MISSION_CONTROL_SWIPE_THRESHOLD = 0.08
DESKTOP_SWIPE_THRESHOLD = 0.16
MISSION_CONTROL_COOLDOWN_SECONDS = 1.2
CURSOR_MOVE_DEADZONE_PX = 1.5
PROCESS_FRAME_WIDTH = 960

MODE_CONTINUITY = "continuity_touchscreen"
MODE_MACBOOK = "macbook_air_mouse"
MODE_INFO = {
    MODE_CONTINUITY: {
        "label": "Continuity Camera touchscreen",
        "description": "Use your iPhone camera as a low-latency gesture touchscreen.",
        "default_hint": "On macOS this is often a higher camera index than the built-in webcam.",
    },
    MODE_MACBOOK: {
        "label": "MacBook camera air-mouse",
        "description": (
            "Use your built-in camera to move the cursor with your finger "
            "and pinch to click."
        ),
        "default_hint": "On macOS this is commonly camera index 0.",
    },
}


class CameraUnavailableError(RuntimeError):
    """Raised when OpenCV cannot access any usable camera."""


def calibration_file_for_mode(mode: str) -> Path:
    """Keep a separate saved calibration per input mode/camera setup."""
    return Path(__file__).with_name(f"gesture_calibration_{mode}.json")


def default_calibration() -> Dict[str, List[float]]:
    """Fallback mapping that treats the full camera frame as the gesture zone."""
    return {
        "top_left": [0.0, 0.0],
        "top_right": [1.0, 0.0],
        "bottom_right": [1.0, 1.0],
        "bottom_left": [0.0, 1.0],
    }


def camera_backends() -> List[int]:
    """Return camera backends to try, in preference order."""
    if platform.system() == "Darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def configure_camera(capture: cv2.VideoCapture) -> None:
    """Apply common capture settings."""
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, PROCESS_FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def open_camera(index: int) -> cv2.VideoCapture:
    """Open a camera with a backend appropriate for the current platform."""
    fallback = cv2.VideoCapture()
    for backend in camera_backends():
        capture = cv2.VideoCapture(index, backend)
        configure_camera(capture)
        if capture.isOpened():
            return capture
        capture.release()
    return fallback


def close_camera_window(capture: cv2.VideoCapture, *, all_windows: bool = False) -> None:
    """Release a capture and close the OpenCV preview window."""
    capture.release()
    try:
        if all_windows:
            cv2.destroyAllWindows()
        else:
            cv2.destroyWindow(WINDOW_NAME)
    except cv2.error:
        pass


def camera_scan_limit(mode: str) -> int:
    """Continuity Camera can appear at higher indexes than built-in webcams."""
    if mode == MODE_CONTINUITY:
        return CONTINUITY_CAMERA_SCAN_LIMIT
    return DEFAULT_CAMERA_SCAN_LIMIT


def macos_camera_names() -> List[str]:
    """Ask macOS which camera devices are currently visible."""
    if platform.system() != "Darwin":
        return []

    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    names: List[str] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line.endswith(":") or line == "Camera:":
            continue
        names.append(line[:-1])
    return names


def has_continuity_camera(names: List[str]) -> bool:
    """Return whether the macOS camera list includes an iPhone/Continuity camera."""
    return any(
        "iphone" in name.lower() or "continuity" in name.lower()
        for name in names
    )


def print_continuity_camera_hint(names: Optional[List[str]] = None) -> None:
    """Print a targeted hint when macOS does not expose an iPhone camera."""
    if names is None:
        names = macos_camera_names()

    if names:
        print("macOS visible cameras:", ", ".join(names))
    else:
        print("macOS did not report any visible cameras.")

    if not has_continuity_camera(names):
        print("I do not see an iPhone/Continuity Camera in macOS right now.")
        print("Check that the iPhone is nearby, unlocked, on the same Apple ID,")
        print("and has Continuity Camera enabled in Settings -> General -> AirPlay & Continuity.")
        print("Also grant Camera permission to Terminal/Python/Codex in macOS Privacy settings.")


def open_continuity_camera_helper() -> None:
    """Open FaceTime so macOS can wake/register Continuity Camera."""
    if platform.system() != "Darwin":
        print("Automatic Continuity Camera wake-up is only available on macOS.")
        return

    try:
        subprocess.run(["open", "-a", "FaceTime"], check=False)
    except OSError as exc:
        print(f"Could not open FaceTime automatically: {exc}")
        return

    print(
        "Opened FaceTime. If the iPhone camera appears there, select it, "
        "then come back here and press r to rescan."
    )


def camera_returns_frames(capture: cv2.VideoCapture, attempts: int = CAMERA_PROBE_FRAMES) -> bool:
    """Give slow virtual cameras a moment to start returning frames."""
    for _ in range(attempts):
        ok, frame = capture.read()
        if ok and frame is not None:
            return True
        time.sleep(CAMERA_PROBE_DELAY_SECONDS)
    return False


def camera_index_returns_frames(index: int) -> bool:
    """Probe all supported backends for a readable camera index."""
    for backend in camera_backends():
        capture = cv2.VideoCapture(index, backend)
        configure_camera(capture)
        if capture.isOpened() and camera_returns_frames(capture):
            capture.release()
            return True
        capture.release()
    return False


def should_mirror_camera(mode: str) -> bool:
    """
    Mirror only the built-in camera mode.

    Air-mouse control feels natural as a mirrored selfie preview, but a camera
    aimed at the display should preserve left/right so touches map correctly.
    """
    return mode == MODE_MACBOOK


def prepare_frame_for_tracking(frame, mode: str):
    """Apply mode-specific camera orientation before preview/tracking."""
    if should_mirror_camera(mode):
        return cv2.flip(frame, 1)
    return frame


def choose_mode() -> str:
    """Prompt for the interaction style when the app starts."""
    print("Project Touchlight")
    print("Choose a control mode:")
    print("1. Continuity Camera touchscreen")
    print("2. MacBook camera air-mouse")

    while True:
        choice = input("Select mode [1]: ").strip()
        if choice in ("", "1"):
            return MODE_CONTINUITY
        if choice == "2":
            return MODE_MACBOOK
        print("Enter 1 or 2.")


def choose_camera_index(mode: str, max_tested: Optional[int] = None) -> int:
    """Prompt the user to pick a camera after probing several indexes."""
    visible_camera_names: List[str] = []
    if max_tested is None:
        max_tested = camera_scan_limit(mode)
        if mode == MODE_CONTINUITY and platform.system() == "Darwin":
            visible_camera_names = macos_camera_names()
            if visible_camera_names and not has_continuity_camera(visible_camera_names):
                max_tested = len(visible_camera_names)
    elif mode == MODE_CONTINUITY and platform.system() == "Darwin":
        visible_camera_names = macos_camera_names()

    print(f"Scanning camera indexes 0-{max_tested - 1}...")
    if mode == MODE_CONTINUITY:
        print_continuity_camera_hint(visible_camera_names)

    available: List[int] = []

    for index in range(max_tested):
        if camera_index_returns_frames(index):
            available.append(index)

    if not available:
        if mode != MODE_CONTINUITY:
            raise CameraUnavailableError(
                "No working cameras were found. On macOS, grant Camera permission to "
                "your terminal/Python app in System Settings -> Privacy & Security -> Camera, "
                "then restart the app."
            )

        print("No cameras returned frames during the scan.")
        print("Continuity Camera may still work if macOS exposes it after selection.")
        while True:
            raw_value = input(
                "Enter an index to try, o to open FaceTime, r to rescan, "
                "or q to quit [1]: "
            ).strip()
            if raw_value.lower() == "q":
                raise SystemExit(0)
            if raw_value.lower() == "o":
                open_continuity_camera_helper()
                continue
            if raw_value.lower() == "r":
                return choose_camera_index(mode)
            if raw_value == "":
                return 1
            if raw_value.isdigit():
                return int(raw_value)
            print("Enter a camera index number, o to open FaceTime, r to rescan, or q to quit.")

    default_index = available[0]
    if platform.system() == "Darwin":
        if mode == MODE_CONTINUITY and len(available) > 1:
            default_index = max(available)
        if mode == MODE_MACBOOK and 0 in available:
            default_index = 0

    print(f"Mode: {MODE_INFO[mode]['label']}")
    print(MODE_INFO[mode]["description"])
    print("Available camera indexes:", ", ".join(str(i) for i in available))
    print(MODE_INFO[mode]["default_hint"])
    if (
        mode == MODE_CONTINUITY
        and visible_camera_names
        and not has_continuity_camera(visible_camera_names)
    ):
        print("Only the built-in camera is exposed right now.")
        print(
            "Press o to open FaceTime, choose the iPhone camera there "
            "if it appears, then press r here."
        )
    print(f"Default camera index: {default_index}")

    while True:
        if mode == MODE_CONTINUITY:
            raw_value = input(
                f"Choose camera index [{default_index}], "
                "o to open FaceTime, r to rescan: "
            ).strip()
        else:
            raw_value = input(f"Choose camera index [{default_index}]: ").strip()
        if not raw_value:
            return default_index
        if raw_value.lower() == "o" and mode == MODE_CONTINUITY:
            open_continuity_camera_helper()
            continue
        if raw_value.lower() == "r" and mode == MODE_CONTINUITY:
            return choose_camera_index(mode)
        if raw_value.isdigit() and int(raw_value) in available:
            return int(raw_value)
        if raw_value.isdigit() and mode == MODE_CONTINUITY:
            print(f"Trying undetected Continuity Camera index {raw_value}.")
            return int(raw_value)
        print("Please enter one of the detected camera indexes.")


def confirm_camera_selection(mode: str, camera_index: int) -> bool:
    """
    Show a live preview before starting tracking.

    This helps catch Continuity Camera cases where macOS/OpenCV falls back to a
    different webcam than expected.
    """
    capture = open_camera(camera_index)
    if not capture.isOpened():
        print(f"Could not open camera index {camera_index} for preview.")
        return False

    for _ in range(CAMERA_WARMUP_FRAMES):
        ok, _ = capture.read()
        if not ok:
            time.sleep(CAMERA_PROBE_DELAY_SECONDS)

    preview_started_time = time.time()
    while True:
        ok, frame = capture.read()
        if not ok:
            if (time.time() - preview_started_time) > CAMERA_PREVIEW_TIMEOUT_SECONDS:
                print(f"Camera index {camera_index} opened but did not return preview frames.")
                close_camera_window(capture)
                return False
            time.sleep(CAMERA_PROBE_DELAY_SECONDS)
            continue

        frame = prepare_frame_for_tracking(frame, mode)
        draw_instruction(frame, f"Preview: {MODE_INFO[mode]['label']}")
        draw_instruction(frame, f"Camera index: {camera_index}", 1)
        draw_instruction(frame, "Press y to use this camera, n to choose another, q to quit", 2)
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("y"):
            close_camera_window(capture)
            return True
        if key == ord("n"):
            close_camera_window(capture)
            return False
        if key == ord("q"):
            close_camera_window(capture, all_windows=True)
            raise SystemExit(0)


def load_calibration(calibration_file: Path, mode: str) -> Optional[Dict[str, List[float]]]:
    """Load previously saved calibration points from disk."""
    candidate_files = [calibration_file]
    if mode == MODE_CONTINUITY and calibration_file.name != LEGACY_CALIBRATION_FILE.name:
        candidate_files.append(LEGACY_CALIBRATION_FILE)

    data = None
    for candidate in candidate_files:
        if not candidate.exists():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                break
        except (OSError, json.JSONDecodeError):
            continue

    if data is None:
        return None

    if not all(name in data for name in CORNER_NAMES):
        return None
    return data


def save_calibration(calibration_file: Path, calibration: Dict[str, List[float]]) -> None:
    """Persist gesture-zone corner points so calibration survives restarts."""
    with calibration_file.open("w", encoding="utf-8") as handle:
        json.dump(calibration, handle, indent=2)


def remove_calibration(calibration_file: Path) -> None:
    """Delete the saved calibration file if it exists."""
    if calibration_file.exists():
        calibration_file.unlink()


def ensure_hand_landmarker_model() -> Path:
    """Download the hand landmarker model once and reuse it from cache."""
    if HAND_LANDMARKER_MODEL_FILE.exists():
        return HAND_LANDMARKER_MODEL_FILE

    HAND_LANDMARKER_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmarker model to {HAND_LANDMARKER_MODEL_FILE}...")
    try:
        with urllib.request.urlopen(HAND_LANDMARKER_MODEL_URL) as response:
            with HAND_LANDMARKER_MODEL_FILE.open("wb") as handle:
                handle.write(response.read())
    except (OSError, urllib.error.URLError) as exc:
        if HAND_LANDMARKER_MODEL_FILE.exists():
            HAND_LANDMARKER_MODEL_FILE.unlink()
        raise RuntimeError(
            "Could not download the MediaPipe hand landmarker model. "
            f"Open {HAND_LANDMARKER_MODEL_URL} in a browser to verify network access."
        ) from exc

    return HAND_LANDMARKER_MODEL_FILE


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def draw_instruction(frame, text: str, line: int = 0) -> None:
    """Render instructions in a consistent style on the debug window."""
    origin = (20, 35 + line * 28)
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_hand_landmarks(frame, hand_landmarks) -> None:
    """Draw Tasks API hand landmarks with MediaPipe's classic drawing helper."""
    landmark_list = landmark_pb2.NormalizedLandmarkList()
    landmark_list.landmark.extend(
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=getattr(landmark, "z", 0.0))
        for landmark in hand_landmarks
    )
    mp_drawing.draw_landmarks(frame, landmark_list, mp_hands.HAND_CONNECTIONS)


def calibrate_gesture_zone(
    capture: cv2.VideoCapture,
    landmarker: mp.tasks.vision.HandLandmarker,
    calibration_file: Path,
    mode: str,
    start_timestamp_ms: int,
) -> Tuple[Optional[Dict[str, List[float]]], int]:
    """
    Ask the user to point at four corners of the usable gesture zone.

    The saved points form an axis-aligned rectangle in normalized camera space.
    """
    print("\nCalibration started.")
    print(f"Mode: {MODE_INFO[mode]['label']}")
    print("Point your INDEX fingertip at each requested corner and press SPACE to capture it.")
    print("Press ESC to cancel calibration.\n")

    calibration: Dict[str, List[float]] = {}
    current_corner = 0
    timestamp_ms = start_timestamp_ms

    while current_corner < len(CORNER_NAMES):
        ok, frame = capture.read()
        if not ok:
            continue

        frame = prepare_frame_for_tracking(frame, mode)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms += 1
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        index_tip: Optional[Tuple[float, float]] = None
        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            draw_hand_landmarks(frame, hand_landmarks)
            index_landmark = hand_landmarks[8]
            index_tip = (index_landmark.x, index_landmark.y)

            h, w = frame.shape[:2]
            px = int(index_landmark.x * w)
            py = int(index_landmark.y * h)
            cv2.circle(frame, (px, py), 10, (0, 255, 0), -1)

        draw_instruction(frame, f"{MODE_INFO[mode]['label']} calibration")
        draw_instruction(
            frame,
            f"Point to {CORNER_NAMES[current_corner].replace('_', ' ')} and press SPACE",
            1,
        )
        draw_instruction(frame, "ESC cancels", 2)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("Calibration cancelled.")
            return None, timestamp_ms

        if key == 32:
            if index_tip is None:
                print("No hand detected. Try again.")
                continue
            calibration[CORNER_NAMES[current_corner]] = [index_tip[0], index_tip[1]]
            print(f"Captured {CORNER_NAMES[current_corner]}: {index_tip}")
            current_corner += 1

    save_calibration(calibration_file, calibration)
    print(f"Calibration saved to {calibration_file}")
    return calibration, timestamp_ms


def run_calibration_safely(
    capture: cv2.VideoCapture,
    landmarker: mp.tasks.vision.HandLandmarker,
    calibration_file: Path,
    mode: str,
    start_timestamp_ms: int,
) -> Tuple[Optional[Dict[str, List[float]]], int]:
    """Keep calibration failures from crashing the whole app."""
    try:
        return calibrate_gesture_zone(
            capture,
            landmarker,
            calibration_file,
            mode,
            start_timestamp_ms,
        )
    except Exception as exc:
        print(f"Calibration failed: {exc}")
        traceback.print_exc()
        return None, start_timestamp_ms


def calibration_bounds(calibration: Dict[str, List[float]]) -> Tuple[float, float, float, float]:
    """Convert four captured corners into rectangle bounds in normalized space."""
    xs = [calibration[name][0] for name in CORNER_NAMES]
    ys = [calibration[name][1] for name in CORNER_NAMES]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Guard against a collapsed calibration rectangle.
    if math.isclose(min_x, max_x):
        max_x = min_x + 1e-6
    if math.isclose(min_y, max_y):
        max_y = min_y + 1e-6
    return min_x, max_x, min_y, max_y


def map_to_screen(
    point: Tuple[float, float],
    calibration: Dict[str, List[float]],
    screen_width: int,
    screen_height: int,
) -> Tuple[float, float]:
    """
    Linearly interpolate normalized fingertip coordinates onto screen space.

    Calibration defines the gesture zone rectangle that maps to the full screen.
    """
    min_x, max_x, min_y, max_y = calibration_bounds(calibration)
    normalized_x = (point[0] - min_x) / (max_x - min_x)
    normalized_y = (point[1] - min_y) / (max_y - min_y)

    normalized_x = clamp(normalized_x, 0.0, 1.0)
    normalized_y = clamp(normalized_y, 0.0, 1.0)

    screen_x = normalized_x * screen_width
    screen_y = normalized_y * screen_height
    return screen_x, screen_y


def ema(
    previous: Optional[Tuple[float, float]],
    current: Tuple[float, float],
    alpha: float,
) -> Tuple[float, float]:
    """Smooth cursor motion with an exponential moving average."""
    if previous is None:
        return current
    return (
        previous[0] + alpha * (current[0] - previous[0]),
        previous[1] + alpha * (current[1] - previous[1]),
    )


def normalized_distance(a, b) -> float:
    """Distance between two MediaPipe landmarks in normalized image space."""
    return math.hypot(a.x - b.x, a.y - b.y)


def finger_extended(hand_landmarks, tip_index: int, pip_index: int) -> bool:
    """Approximate whether a finger is raised by comparing tip and joint height."""
    return hand_landmarks[tip_index].y < hand_landmarks[pip_index].y


def extended_finger_state(hand_landmarks) -> Dict[str, bool]:
    """Summarize which non-thumb fingers look extended."""
    return {
        "index": finger_extended(hand_landmarks, 8, 6),
        "middle": finger_extended(hand_landmarks, 12, 10),
        "ring": finger_extended(hand_landmarks, 16, 14),
        "pinky": finger_extended(hand_landmarks, 20, 18),
    }


def average_point(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Average a short list of normalized landmark points."""
    count = max(len(points), 1)
    return (
        sum(point[0] for point in points) / count,
        sum(point[1] for point in points) / count,
    )


def pinch_is_active(distance: float, was_active: bool) -> bool:
    """Use hysteresis so the pinch state does not flicker around one threshold."""
    if was_active:
        return distance < PINCH_UP_THRESHOLD
    return distance < PINCH_DOWN_THRESHOLD


def detect_index_tap(
    current_point: Tuple[float, float],
    previous_point: Optional[Tuple[float, float]],
    candidate_point: Optional[Tuple[float, float]],
    candidate_started_time: Optional[float],
    last_tap_time: float,
    now: float,
) -> Tuple[bool, Optional[Tuple[float, float]], Optional[float]]:
    """
    Detect a screen-tap-like gesture from fingertip motion.

    A tap is treated as a quick move into position followed by a short, steady
    settle at nearly the same point. This works with ordinary 2D hand landmarks,
    where real touch depth is not reliable enough to use by itself.
    """
    if previous_point is None:
        return False, None, None

    movement = math.hypot(
        current_point[0] - previous_point[0],
        current_point[1] - previous_point[1],
    )

    if movement >= TAP_FAST_MOVE_THRESHOLD:
        return False, current_point, now

    if candidate_point is None or candidate_started_time is None:
        return False, None, None

    drift = math.hypot(
        current_point[0] - candidate_point[0],
        current_point[1] - candidate_point[1],
    )
    if drift > TAP_MAX_DRIFT:
        return False, None, None

    settled = movement <= TAP_STILL_THRESHOLD
    old_enough = (now - candidate_started_time) >= TAP_SETTLE_SECONDS
    debounced = (now - last_tap_time) >= TAP_DEBOUNCE_SECONDS
    if settled and old_enough and debounced:
        return True, None, None

    return False, candidate_point, candidate_started_time


def perform_zoom_scroll(amount: int) -> None:
    """Zoom by pairing a scroll event with the macOS Command modifier."""
    if amount == 0:
        return
    pyautogui.keyDown("command")
    try:
        pyautogui.scroll(amount)
    finally:
        pyautogui.keyUp("command")


def main() -> None:
    mode = choose_mode()
    calibration_file = calibration_file_for_mode(mode)

    while True:
        camera_index = choose_camera_index(mode)
        if confirm_camera_selection(mode, camera_index):
            break
        print("Choose another camera index.\n")

    capture = open_camera(camera_index)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    for _ in range(CAMERA_WARMUP_FRAMES):
        ok, _ = capture.read()
        if not ok:
            break

    screen_width, screen_height = pyautogui.size()
    calibration = load_calibration(calibration_file, mode)

    model_path = ensure_hand_landmarker_model()
    mp_vision = mp.tasks.vision
    mp_base = mp.tasks.BaseOptions

    smoothed_cursor: Optional[Tuple[float, float]] = None
    last_sent_cursor: Optional[Tuple[float, float]] = None
    pinch_active = False
    last_click_time = 0.0
    timestamp_ms = 0
    two_finger_previous_center: Optional[Tuple[float, float]] = None
    two_finger_previous_distance: Optional[float] = None
    four_finger_previous_center: Optional[Tuple[float, float]] = None
    four_finger_accumulated_dx = 0.0
    four_finger_accumulated_dy = 0.0
    last_mission_control_time = 0.0
    pinch_started_time: Optional[float] = None
    mouse_dragging = False
    previous_index_point: Optional[Tuple[float, float]] = None
    tap_candidate_point: Optional[Tuple[float, float]] = None
    tap_candidate_started_time: Optional[float] = None
    last_tap_time = 0.0

    hand_landmarker_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_base(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    try:
        with mp_vision.HandLandmarker.create_from_options(hand_landmarker_options) as landmarker:
            if calibration is None:
                calibration, timestamp_ms = run_calibration_safely(
                    capture,
                    landmarker,
                    calibration_file,
                    mode,
                    timestamp_ms,
                )
                if calibration is None:
                    calibration = default_calibration()
                    print(
                        "Calibration skipped. Using full-frame fallback mapping "
                        "until you calibrate with 'c'."
                    )

            while True:
                ok, frame = capture.read()
                if not ok:
                    continue

                frame = prepare_frame_for_tracking(frame, mode)
                debug_frame = frame.copy()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms += 1
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                mapped_cursor: Optional[Tuple[float, float]] = None
                pinch_now = False
                gesture_status = "None"
                gesture_override_active = False

                if result.hand_landmarks:
                    hand_landmarks = result.hand_landmarks[0]
                    draw_hand_landmarks(debug_frame, hand_landmarks)

                    index_tip = hand_landmarks[8]
                    thumb_tip = hand_landmarks[4]
                    finger_state = extended_finger_state(hand_landmarks)

                    if mode == MODE_MACBOOK:
                        two_finger_mode = (
                            finger_state["index"]
                            and finger_state["middle"]
                            and not finger_state["ring"]
                            and not finger_state["pinky"]
                        )
                        four_finger_mode = all(finger_state.values())

                        if four_finger_mode:
                            four_finger_points = [
                                (hand_landmarks[8].x, hand_landmarks[8].y),
                                (hand_landmarks[12].x, hand_landmarks[12].y),
                                (hand_landmarks[16].x, hand_landmarks[16].y),
                                (hand_landmarks[20].x, hand_landmarks[20].y),
                            ]
                            current_center = average_point(four_finger_points)

                            if four_finger_previous_center is not None:
                                delta_x = current_center[0] - four_finger_previous_center[0]
                                delta_y = four_finger_previous_center[1] - current_center[1]
                                ready = (
                                    time.time() - last_mission_control_time
                                ) > MISSION_CONTROL_COOLDOWN_SECONDS
                                four_finger_accumulated_dx += delta_x
                                four_finger_accumulated_dy += delta_y

                                if (
                                    ready
                                    and abs(four_finger_accumulated_dx) > DESKTOP_SWIPE_THRESHOLD
                                    and abs(four_finger_accumulated_dx)
                                    > abs(four_finger_accumulated_dy)
                                ):
                                    if four_finger_accumulated_dx > 0:
                                        pyautogui.hotkey("ctrl", "right")
                                        gesture_status = "Next desktop"
                                    else:
                                        pyautogui.hotkey("ctrl", "left")
                                        gesture_status = "Previous desktop"
                                    last_mission_control_time = time.time()
                                    four_finger_accumulated_dx = 0.0
                                    four_finger_accumulated_dy = 0.0
                                elif (
                                    ready
                                    and four_finger_accumulated_dy
                                    > MISSION_CONTROL_SWIPE_THRESHOLD
                                ):
                                    pyautogui.hotkey("ctrl", "up")
                                    last_mission_control_time = time.time()
                                    gesture_status = "Mission Control"
                                    four_finger_accumulated_dx = 0.0
                                    four_finger_accumulated_dy = 0.0
                            four_finger_previous_center = current_center
                            two_finger_previous_center = None
                            two_finger_previous_distance = None
                            gesture_override_active = True
                            smoothed_cursor = None
                        elif two_finger_mode:
                            two_finger_points = [
                                (hand_landmarks[8].x, hand_landmarks[8].y),
                                (hand_landmarks[12].x, hand_landmarks[12].y),
                            ]
                            two_finger_center = average_point(two_finger_points)
                            two_finger_distance = normalized_distance(
                                hand_landmarks[8],
                                hand_landmarks[12],
                            )

                            if (
                                two_finger_previous_center is not None
                                and two_finger_previous_distance is not None
                            ):
                                delta_x = two_finger_center[0] - two_finger_previous_center[0]
                                delta_y = two_finger_previous_center[1] - two_finger_center[1]
                                delta_distance = two_finger_distance - two_finger_previous_distance

                                if (
                                    abs(delta_distance) > ZOOM_DISTANCE_THRESHOLD
                                    and abs(delta_distance) > abs(delta_y)
                                ):
                                    zoom_scroll = int(delta_distance * ZOOM_SCROLL_SCALE)
                                    if zoom_scroll != 0:
                                        perform_zoom_scroll(zoom_scroll)
                                        gesture_status = "Two-finger zoom"
                                else:
                                    if abs(delta_y) > SCROLL_DELTA_THRESHOLD:
                                        pyautogui.scroll(int(delta_y * SCROLL_SCALE))
                                        gesture_status = "Two-finger vertical scroll"
                                    if abs(delta_x) > SCROLL_DELTA_THRESHOLD:
                                        pyautogui.hscroll(int(delta_x * SCROLL_SCALE))
                                        gesture_status = "Two-finger horizontal scroll"

                            two_finger_previous_center = two_finger_center
                            two_finger_previous_distance = two_finger_distance
                            four_finger_previous_center = None
                            four_finger_accumulated_dx = 0.0
                            four_finger_accumulated_dy = 0.0
                            gesture_override_active = True
                            smoothed_cursor = None
                        else:
                            two_finger_previous_center = None
                            two_finger_previous_distance = None
                            four_finger_previous_center = None
                            four_finger_accumulated_dx = 0.0
                            four_finger_accumulated_dy = 0.0
                            last_sent_cursor = None

                    if not gesture_override_active:
                        index_point = (index_tip.x, index_tip.y)
                        mapped_cursor = map_to_screen(
                            index_point,
                            calibration,
                            screen_width,
                            screen_height,
                        )
                        smoothed_cursor = ema(smoothed_cursor, mapped_cursor, SMOOTHING_ALPHA)
                        if (
                            last_sent_cursor is None
                            or math.hypot(
                                smoothed_cursor[0] - last_sent_cursor[0],
                                smoothed_cursor[1] - last_sent_cursor[1],
                            ) >= CURSOR_MOVE_DEADZONE_PX
                        ):
                            pyautogui.moveTo(
                                smoothed_cursor[0],
                                smoothed_cursor[1],
                                _pause=False,
                            )
                            last_sent_cursor = smoothed_cursor

                        now = time.time()
                        if mode == MODE_CONTINUITY:
                            (
                                tapped,
                                tap_candidate_point,
                                tap_candidate_started_time,
                            ) = detect_index_tap(
                                index_point,
                                previous_index_point,
                                tap_candidate_point,
                                tap_candidate_started_time,
                                last_tap_time,
                                now,
                            )
                            if tapped:
                                pyautogui.click()
                                last_tap_time = now
                                gesture_status = "Tap click"
                            elif tap_candidate_point is not None:
                                gesture_status = "Tap target"
                            else:
                                gesture_status = "Cursor control"

                            pinch_active = False
                            pinch_started_time = None
                            if mouse_dragging:
                                pyautogui.mouseUp()
                                mouse_dragging = False
                        else:
                            pinch_distance = normalized_distance(index_tip, thumb_tip)
                            pinch_now = pinch_is_active(pinch_distance, pinch_active)

                            if pinch_now and not pinch_active:
                                pinch_started_time = now

                            if pinch_now and pinch_started_time is not None:
                                pinch_elapsed = time.time() - pinch_started_time
                                if not mouse_dragging and pinch_elapsed >= DRAG_HOLD_SECONDS:
                                    pyautogui.mouseDown()
                                    mouse_dragging = True
                                    gesture_status = "Pinch drag"

                            if not pinch_now and pinch_active:
                                pinch_elapsed = 0.0
                                if pinch_started_time is not None:
                                    pinch_elapsed = now - pinch_started_time

                                if mouse_dragging:
                                    pyautogui.mouseUp()
                                    mouse_dragging = False
                                    gesture_status = "Drop"
                                elif (
                                    pinch_elapsed < DRAG_HOLD_SECONDS
                                    and (now - last_click_time) > CLICK_DEBOUNCE_SECONDS
                                ):
                                    pyautogui.click()
                                    last_click_time = now
                                    gesture_status = "Pinch click"
                                pinch_started_time = None

                            pinch_active = pinch_now
                            if pinch_now:
                                if mouse_dragging:
                                    gesture_status = "Pinch drag"
                                elif (
                                    pinch_started_time is not None
                                    and (now - pinch_started_time) < DRAG_HOLD_SECONDS
                                ):
                                    gesture_status = "Pinch hold"
                            elif gesture_status == "None":
                                gesture_status = "Cursor control"
                            tap_candidate_point = None
                            tap_candidate_started_time = None

                        previous_index_point = index_point
                    else:
                        if mouse_dragging:
                            pyautogui.mouseUp()
                            mouse_dragging = False
                        pinch_active = False
                        pinch_started_time = None
                        last_sent_cursor = None
                        previous_index_point = None
                        tap_candidate_point = None
                        tap_candidate_started_time = None

                    h, w = debug_frame.shape[:2]
                    index_px = (int(index_tip.x * w), int(index_tip.y * h))
                    thumb_px = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                    cv2.circle(debug_frame, index_px, 9, (0, 255, 0), -1)
                    cv2.circle(debug_frame, thumb_px, 9, (255, 0, 0), -1)
                    cv2.line(debug_frame, index_px, thumb_px, (255, 255, 0), 2)

                    min_x, max_x, min_y, max_y = calibration_bounds(calibration)
                    zone_start = (int(min_x * w), int(min_y * h))
                    zone_end = (int(max_x * w), int(max_y * h))
                    cv2.rectangle(debug_frame, zone_start, zone_end, (0, 200, 255), 2)
                else:
                    if mouse_dragging:
                        pyautogui.mouseUp()
                        mouse_dragging = False
                    pinch_active = False
                    pinch_started_time = None
                    two_finger_previous_center = None
                    two_finger_previous_distance = None
                    four_finger_previous_center = None
                    four_finger_accumulated_dx = 0.0
                    four_finger_accumulated_dy = 0.0
                    last_sent_cursor = None
                    previous_index_point = None
                    tap_candidate_point = None
                    tap_candidate_started_time = None

                draw_instruction(debug_frame, "Project Touchlight")
                draw_instruction(
                    debug_frame,
                    "Keys: c = calibrate, r = reset calibration, q = quit",
                    1,
                )
                draw_instruction(
                    debug_frame,
                    f"Mode: {MODE_INFO[mode]['label']} | Camera: {camera_index}",
                    2,
                )
                draw_instruction(
                    debug_frame,
                    (
                        f"Pinch: {'ON' if pinch_now else 'OFF'} | "
                        f"Drag: {'ON' if mouse_dragging else 'OFF'} | "
                        f"Gesture: {gesture_status}"
                    ),
                    3,
                )

                if smoothed_cursor is not None:
                    draw_instruction(
                        debug_frame,
                        f"Cursor: ({int(smoothed_cursor[0])}, {int(smoothed_cursor[1])})",
                        4,
                    )

                if mapped_cursor is not None:
                    h, w = debug_frame.shape[:2]
                    cursor_preview = (
                        int((mapped_cursor[0] / screen_width) * w),
                        int((mapped_cursor[1] / screen_height) * h),
                    )
                    cv2.circle(debug_frame, cursor_preview, 8, (0, 0, 255), -1)

                cv2.imshow(WINDOW_NAME, debug_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                if key == ord("c"):
                    new_calibration, timestamp_ms = run_calibration_safely(
                        capture,
                        landmarker,
                        calibration_file,
                        mode,
                        timestamp_ms,
                    )
                    if new_calibration is not None:
                        calibration = new_calibration
                        smoothed_cursor = None
                        last_sent_cursor = None
                        pinch_active = False
                        pinch_started_time = None
                    else:
                        print("Calibration canceled. Continuing with the current mapping.")
                if key == ord("r"):
                    remove_calibration(calibration_file)
                    print("Saved calibration removed.")
                    calibration, timestamp_ms = run_calibration_safely(
                        capture,
                        landmarker,
                        calibration_file,
                        mode,
                        timestamp_ms,
                    )
                    smoothed_cursor = None
                    last_sent_cursor = None
                    pinch_active = False
                    pinch_started_time = None
                    if calibration is None:
                        calibration = default_calibration()
                        print(
                            "Calibration canceled. Continuing with the "
                            "full-frame fallback mapping."
                        )
    finally:
        if mouse_dragging:
            pyautogui.mouseUp()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutdown by user.")
    except CameraUnavailableError as exc:
        print(f"\nCamera setup issue: {exc}")
    except Exception as exc:
        print(f"\nCrash: {type(exc).__name__}: {exc}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)
