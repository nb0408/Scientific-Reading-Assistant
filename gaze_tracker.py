"""
Phase 7b: Gaze/Attention Tracking (MediaPipe Tasks API)
-------------------------------------------------------------
Analyzes a single webcam frame (snapshot) and estimates an attention
score, using MediaPipe's current Face Landmarker task:
  - Blink detection: via the model's built-in blendshape scores
    (eyeBlinkLeft / eyeBlinkRight) -- these are trained outputs, not a
    hand-rolled eye-aspect-ratio calculation.
  - Head pose (yaw/pitch/roll): decoded from the facial transformation
    matrix the model returns directly -- no manual solvePnP needed.

API NOTE: MediaPipe removed its older `mp.solutions.face_mesh` API.
This module uses the current `mediapipe.tasks` API, which requires a
downloaded .task model file (see setup below) -- the model is NOT
bundled in the pip package itself.

SCOPING NOTE -- read this before demoing:
Streamlit has no native continuous video pipeline; a live, frame-by-frame
webcam feed would need `streamlit-webrtc`, which adds WebRTC/TURN-server
dependencies that can be flaky across networks and firewalls. This module
instead analyzes periodic SNAPSHOTS (taken via Streamlit's built-in
st.camera_input widget), not continuous video. This is a legitimate,
honest scoping decision for a course project -- say so explicitly in
your report rather than implying continuous real-time tracking.

Attention score here is a heuristic combining these two signals, NOT a
validated cognitive-attention measure -- real gaze-based confusion
detection (see your own Lit Review [5], Ito et al. 2026) uses much more
sophisticated multimodal fusion models trained on labeled data. This
module produces a reasonable proxy signal, not a research-grade one.

SETUP (one-time):
  pip install mediapipe opencv-python-headless
  Download the face landmarker model (~4MB):
    curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
  (or open that URL in a browser and save the file as face_landmarker.task
  in this same project folder)
"""

import os
import math
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

_detector = None

BLINK_SCORE_THRESHOLD = 0.5
YAW_LOOKING_AWAY_THRESHOLD = 25.0    # degrees
PITCH_LOOKING_AWAY_THRESHOLD = 20.0  # degrees


def _get_detector():
    global _detector
    if _detector is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"Face landmarker model not found at {_MODEL_PATH}.\n"
                f"Download it once with:\n"
                f'  curl -L -o face_landmarker.task "{_MODEL_URL}"\n'
                f"or open that URL in a browser and save the file as "
                f"'face_landmarker.task' in this project folder."
            )
        base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        _detector = mp_vision.FaceLandmarker.create_from_options(options)
    return _detector


def _rotation_matrix_to_euler(R: np.ndarray):
    """Standard rotation-matrix -> (yaw, pitch, roll) in degrees."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    pitch = math.degrees(math.atan2(-R[2, 0], sy))
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    return yaw, pitch, roll


def _score_from_signals(looking_away: bool, is_blinking: bool) -> float:
    score = 1.0
    if looking_away:
        score -= 0.5
    if is_blinking:
        score -= 0.1  # a single blink is normal; mild penalty only
    return round(max(0.0, min(1.0, score)), 2)


def analyze_frame(frame_bgr: np.ndarray) -> dict:
    """
    frame_bgr: a BGR image as a numpy array (e.g. from cv2.imread, or a
               decoded Streamlit camera_input snapshot).
    Returns: {face_detected, is_blinking, yaw, pitch, roll,
              looking_away, attention_score}
    """
    detector = _get_detector()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    result = detector.detect(mp_image)

    if not result.face_landmarks:
        return {
            "face_detected": False, "is_blinking": False,
            "yaw": None, "pitch": None, "roll": None,
            "looking_away": True, "attention_score": 0.0,
        }

    blendshapes = {b.category_name: b.score for b in result.face_blendshapes[0]}
    eye_blink_left = blendshapes.get("eyeBlinkLeft", 0.0)
    eye_blink_right = blendshapes.get("eyeBlinkRight", 0.0)
    is_blinking = ((eye_blink_left + eye_blink_right) / 2.0) > BLINK_SCORE_THRESHOLD

    transform = np.array(result.facial_transformation_matrixes[0])
    yaw, pitch, roll = _rotation_matrix_to_euler(transform[:3, :3])
    looking_away = abs(yaw) > YAW_LOOKING_AWAY_THRESHOLD or abs(pitch) > PITCH_LOOKING_AWAY_THRESHOLD

    return {
        "face_detected": True,
        "is_blinking": is_blinking,
        "yaw": round(yaw, 1),
        "pitch": round(pitch, 1),
        "roll": round(roll, 1),
        "looking_away": looking_away,
        "attention_score": _score_from_signals(looking_away, is_blinking),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python gaze_tracker.py <path_to_image.jpg>")
        sys.exit(1)
    frame = cv2.imread(sys.argv[1])
    if frame is None:
        print(f"Could not read image: {sys.argv[1]}")
        sys.exit(1)
    print(analyze_frame(frame))
