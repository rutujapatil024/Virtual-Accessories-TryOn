"""
overlay_engine.py — Accessory Overlay Engine

Handles face landmark detection via MediaPipe FaceLandmarker (Task API)
and overlays accessories (glasses, hats, earrings) on webcam frames
with proper scaling, rotation, and alpha blending.

Compatible with MediaPipe >= 0.10.x (Task API).
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import os


class OverlayEngine:
    """
    Core engine for detecting face landmarks and overlaying
    accessory images onto webcam frames in real time.
    """

    # ── Landmark indices (478 total in FaceLandmarker) ────────────────
    # Eyes
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263

    # Forehead / top of head
    FOREHEAD_TOP = 10

    # Ears
    LEFT_EAR = 234
    RIGHT_EAR = 454

    # Face shape measurement points
    CHIN = 152
    LEFT_CHEEK = 234
    RIGHT_CHEEK = 454
    LEFT_FOREHEAD = 71
    RIGHT_FOREHEAD = 301
    LEFT_JAW = 150
    RIGHT_JAW = 379

    # ── Scale factors per category ────────────────────────────────────
    SCALE_GLASSES = 1.8
    SCALE_HAT = 2.2
    SCALE_EARRING = 0.35

    def __init__(self, model_path: str = None):
        """
        Initialize MediaPipe FaceLandmarker (Task API).

        Args:
            model_path: Path to the face_landmarker.task model file.
                        Defaults to 'face_landmarker.task' in the same directory.
        """
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'face_landmarker.task'
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe model not found at: {model_path}\n"
                "Download it from: https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            )

        # Create FaceLandmarker with VIDEO running mode
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        # Current accessory state
        self.accessory_img = None       # Loaded PNG with alpha
        self.accessory_category = None  # 'glasses', 'hats', or 'earrings'
        self.accessory_path = None      # Path for reference

        # Face shape measurement buffer for smoothing (last N measurements)
        self._face_measurements_buffer = []
        self._FACE_BUFFER_SIZE = 5

    # ── Public API ────────────────────────────────────────────────────

    def load_accessory(self, path: str, category: str) -> bool:
        """
        Load an accessory PNG image with alpha channel.

        Args:
            path: File path to the PNG accessory image.
            category: One of 'glasses', 'hats', or 'earrings'.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not os.path.exists(path):
            print(f"[OverlayEngine] File not found: {path}")
            return False

        # Load with alpha channel preserved
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[OverlayEngine] Failed to read image: {path}")
            return False

        # If image has no alpha channel, add one (fully opaque)
        if len(img.shape) == 2:
            # Grayscale — convert to BGRA
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            alpha = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype) * 255
            img = np.concatenate([img, alpha], axis=2)

        self.accessory_img = img
        self.accessory_category = category
        self.accessory_path = path
        print(f"[OverlayEngine] Loaded {category} accessory: {path}")
        return True

    def clear_accessory(self):
        """Remove the current accessory overlay."""
        self.accessory_img = None
        self.accessory_category = None
        self.accessory_path = None

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single webcam frame:
        1. Detect face landmarks
        2. Overlay the selected accessory (if any)
        3. Return the composited frame

        Args:
            frame: BGR webcam frame from OpenCV.

        Returns:
            Processed BGR frame with accessory overlay.
        """
        if frame is None:
            return frame

        # Flip horizontally for a mirror-like experience
        frame = cv2.flip(frame, 1)

        # If no accessory is loaded, return the plain frame
        if self.accessory_img is None:
            return frame

        # Convert BGR → RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect face landmarks
        try:
            results = self.landmarker.detect(mp_image)
        except Exception as e:
            print(f"[OverlayEngine] Detection error: {e}")
            return frame

        # If no face detected, return frame as-is
        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return frame

        face_landmarks = results.face_landmarks[0]
        h, w, _ = frame.shape

        # Convert normalized landmarks to pixel coordinates
        points = {}
        for idx in [
            self.LEFT_EYE_OUTER, self.LEFT_EYE_INNER,
            self.RIGHT_EYE_INNER, self.RIGHT_EYE_OUTER,
            self.FOREHEAD_TOP, self.LEFT_EAR, self.RIGHT_EAR,
            self.CHIN, self.LEFT_CHEEK, self.RIGHT_CHEEK,
            self.LEFT_FOREHEAD, self.RIGHT_FOREHEAD,
            self.LEFT_JAW, self.RIGHT_JAW,
        ]:
            lm = face_landmarks[idx]
            points[idx] = (int(lm.x * w), int(lm.y * h))

        # Dispatch to the correct overlay method
        if self.accessory_category == 'glasses':
            frame = self._overlay_glasses(frame, points)
        elif self.accessory_category == 'hats':
            frame = self._overlay_hat(frame, points)
        elif self.accessory_category == 'earrings':
            frame = self._overlay_earrings(frame, points)

        return frame

    def analyze_face_shape(self, frame: np.ndarray) -> dict:
        """
        Analyze the face shape from a single frame.

        Returns:
            dict with 'shape' and 'suggestion' keys, or None if no face found.
        """
        if frame is None:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        try:
            results = self.landmarker.detect(mp_image)
        except Exception as e:
            print(f"[OverlayEngine] Detection error: {e}")
            return None

        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return None

        face_landmarks = results.face_landmarks[0]
        h, w, _ = frame.shape

        # Extract measurement points
        def pt(idx):
            lm = face_landmarks[idx]
            return (lm.x * w, lm.y * h)

        forehead_top = pt(self.FOREHEAD_TOP)
        chin = pt(self.CHIN)
        left_cheek = pt(self.LEFT_CHEEK)
        right_cheek = pt(self.RIGHT_CHEEK)
        left_forehead = pt(self.LEFT_FOREHEAD)
        right_forehead = pt(self.RIGHT_FOREHEAD)
        left_jaw = pt(self.LEFT_JAW)
        right_jaw = pt(self.RIGHT_JAW)

        # Calculate proportions
        face_height = math.dist(forehead_top, chin)
        face_width = math.dist(left_cheek, right_cheek)
        forehead_width = math.dist(left_forehead, right_forehead)
        jawline_width = math.dist(left_jaw, right_jaw)

        if face_height == 0:
            return None

        ratio = face_width / face_height
        forehead_jaw_ratio = forehead_width / jawline_width if jawline_width > 0 else 1.0
        forehead_cheek_ratio = forehead_width / face_width if face_width > 0 else 1.0
        jaw_cheek_ratio = jawline_width / face_width if face_width > 0 else 1.0

        # Store this measurement in the buffer
        self._face_measurements_buffer.append({
            'ratio': ratio,
            'forehead_jaw_ratio': forehead_jaw_ratio,
            'forehead_cheek_ratio': forehead_cheek_ratio,
            'jaw_cheek_ratio': jaw_cheek_ratio,
            'face_width': face_width,
            'face_height': face_height,
            'forehead_width': forehead_width,
            'jawline_width': jawline_width,
        })

        # Keep only the last N measurements
        if len(self._face_measurements_buffer) > self._FACE_BUFFER_SIZE:
            self._face_measurements_buffer = self._face_measurements_buffer[-self._FACE_BUFFER_SIZE:]

        # Average over the buffer for stability
        buf = self._face_measurements_buffer
        avg = {key: sum(m[key] for m in buf) / len(buf) for key in buf[0]}

        ratio = avg['ratio']
        forehead_jaw_ratio = avg['forehead_jaw_ratio']
        forehead_cheek_ratio = avg['forehead_cheek_ratio']
        jaw_cheek_ratio = avg['jaw_cheek_ratio']

        # Classification rules (refined thresholds)
        forehead_jaw_diff = abs(avg['forehead_width'] - avg['jawline_width']) / max(avg['forehead_width'], avg['jawline_width'])

        if ratio > 0.80 and forehead_jaw_diff < 0.15:
            shape = "Square"
            suggestion = "Round or oval frames will soften your angular features. Try aviator-style or round glasses."
        elif ratio > 0.80:
            shape = "Round"
            suggestion = "Angular or rectangular frames will add definition to your face. Try wayfarer or square frames."
        elif ratio < 0.68:
            shape = "Oblong"
            suggestion = "Oversized or wide frames will add width and balance your face length. Try big round or butterfly frames."
        elif forehead_jaw_ratio > 1.20:
            shape = "Heart"
            suggestion = "Bottom-heavy or rimless frames will balance your wider forehead. Try aviators or round frames."
        elif jaw_cheek_ratio < 0.80 and forehead_cheek_ratio < 0.85:
            shape = "Diamond"
            suggestion = "Oval or cat-eye frames will complement your cheekbones beautifully. Try rimless or semi-rimless styles."
        else:
            shape = "Oval"
            suggestion = "Lucky you! Most frame styles work well with your face shape. Try classic aviators or cat-eye frames."

        return {
            "shape": shape,
            "suggestion": suggestion,
            "measurements": {
                "face_width": round(avg['face_width'], 1),
                "face_height": round(avg['face_height'], 1),
                "ratio": round(ratio, 2),
                "forehead_width": round(avg['forehead_width'], 1),
                "jawline_width": round(avg['jawline_width'], 1),
            }
        }

    # ── Private overlay methods ───────────────────────────────────────

    def _overlay_glasses(self, frame: np.ndarray, points: dict) -> np.ndarray:
        """Place glasses between the eye landmarks."""
        left_eye = points[self.LEFT_EYE_OUTER]
        right_eye = points[self.RIGHT_EYE_OUTER]

        # Calculate scale based on inter-eye distance
        eye_dist = math.dist(left_eye, right_eye)
        target_width = int(eye_dist * self.SCALE_GLASSES)

        # Calculate rotation angle from eye tilt
        angle = self._compute_rotation(left_eye, right_eye)

        # Center point between the eyes
        center_x = (left_eye[0] + right_eye[0]) // 2
        center_y = (left_eye[1] + right_eye[1]) // 2

        # Scale the accessory
        accessory = self._resize_accessory(self.accessory_img, target_width)
        if accessory is None:
            return frame

        # Rotate the accessory
        accessory = self._rotate_image(accessory, angle)
        if accessory is None:
            return frame

        # Position: center on the midpoint of the eyes
        x = center_x - accessory.shape[1] // 2
        y = center_y - accessory.shape[0] // 2

        return self._alpha_blend(frame, accessory, x, y)

    def _overlay_hat(self, frame: np.ndarray, points: dict) -> np.ndarray:
        """Place hat above the forehead landmark."""
        left_eye = points[self.LEFT_EYE_OUTER]
        right_eye = points[self.RIGHT_EYE_OUTER]
        forehead = points[self.FOREHEAD_TOP]

        # Calculate scale based on face width
        eye_dist = math.dist(left_eye, right_eye)
        target_width = int(eye_dist * self.SCALE_HAT)

        # Calculate rotation angle
        angle = self._compute_rotation(left_eye, right_eye)

        # Center horizontally on the forehead
        center_x = (left_eye[0] + right_eye[0]) // 2

        # Scale the accessory
        accessory = self._resize_accessory(self.accessory_img, target_width)
        if accessory is None:
            return frame

        # Rotate the accessory
        accessory = self._rotate_image(accessory, angle)
        if accessory is None:
            return frame

        # Position: bottom of the hat near the forehead
        x = center_x - accessory.shape[1] // 2
        y = forehead[1] - int(accessory.shape[0] * 0.85)

        return self._alpha_blend(frame, accessory, x, y)

    def _overlay_earrings(self, frame: np.ndarray, points: dict) -> np.ndarray:
        """Place earrings at both ear landmarks."""
        left_ear = points[self.LEFT_EAR]
        right_ear = points[self.RIGHT_EAR]
        left_eye = points[self.LEFT_EYE_OUTER]
        right_eye = points[self.RIGHT_EYE_OUTER]

        # Calculate scale based on inter-eye distance
        eye_dist = math.dist(left_eye, right_eye)
        target_width = int(eye_dist * self.SCALE_EARRING)

        # Scale the accessory
        accessory = self._resize_accessory(self.accessory_img, target_width)
        if accessory is None:
            return frame

        # Place left earring (shifted down to simulate earlobe)
        earlobe_drop = int(eye_dist * 0.25)
        left_x = left_ear[0] - accessory.shape[1] // 2
        left_y = left_ear[1] + earlobe_drop - accessory.shape[0] // 4

        # Place right earring (flip horizontally)
        right_accessory = cv2.flip(accessory, 1)
        right_x = right_ear[0] - right_accessory.shape[1] // 2
        right_y = right_ear[1] + earlobe_drop - right_accessory.shape[0] // 4

        frame = self._alpha_blend(frame, accessory, left_x, left_y)
        frame = self._alpha_blend(frame, right_accessory, right_x, right_y)

        return frame

    # ── Utility methods ───────────────────────────────────────────────

    def _compute_rotation(self, left_point: tuple, right_point: tuple) -> float:
        """
        Calculate the head tilt angle in degrees from two
        symmetric facial landmarks. Handles horizontal flip correctly.
        """
        dx = right_point[0] - left_point[0]
        dy = right_point[1] - left_point[1]
        
        # Prevent 180-degree flip if the image was horizontally mirrored
        if dx < 0:
            dx = -dx
            dy = -dy
            
        angle = math.degrees(math.atan2(dy, dx))
        return angle

    def _resize_accessory(self, img: np.ndarray, target_width: int) -> np.ndarray:
        """Resize the accessory image to the target width, preserving aspect ratio."""
        if target_width <= 0:
            return None

        h, w = img.shape[:2]
        aspect = h / w
        target_height = int(target_width * aspect)

        if target_height <= 0:
            return None

        resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return resized

    def _rotate_image(self, img: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate the accessory image by the given angle (degrees)
        around its center, expanding the canvas to avoid clipping.
        """
        if abs(angle) < 0.5:
            return img  # Skip rotation for negligible angles

        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        # Get the rotation matrix
        rot_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

        # Calculate new bounding box size
        cos = abs(rot_matrix[0, 0])
        sin = abs(rot_matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        # Adjust the rotation matrix for translation
        rot_matrix[0, 2] += (new_w / 2) - center[0]
        rot_matrix[1, 2] += (new_h / 2) - center[1]

        # Apply rotation with transparent border
        rotated = cv2.warpAffine(
            img, rot_matrix, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        return rotated

    def _alpha_blend(self, frame: np.ndarray, overlay: np.ndarray, x: int, y: int) -> np.ndarray:
        """
        Alpha-blend the overlay image onto the frame at position (x, y).

        Uses the overlay's alpha channel for per-pixel blending:
            output = alpha * overlay + (1 - alpha) * frame
        """
        h_overlay, w_overlay = overlay.shape[:2]
        h_frame, w_frame = frame.shape[:2]

        # Calculate the valid region (handle out-of-bounds)
        y1 = max(0, y)
        y2 = min(h_frame, y + h_overlay)
        x1 = max(0, x)
        x2 = min(w_frame, x + w_overlay)

        # Calculate corresponding overlay region
        oy1 = max(0, -y)
        oy2 = oy1 + (y2 - y1)
        ox1 = max(0, -x)
        ox2 = ox1 + (x2 - x1)

        # Validate dimensions
        if y1 >= y2 or x1 >= x2:
            return frame

        # Extract alpha channel (normalized to 0.0 – 1.0)
        alpha = overlay[oy1:oy2, ox1:ox2, 3] / 255.0
        alpha = np.stack([alpha] * 3, axis=-1)

        # Extract RGB channels of overlay
        overlay_rgb = overlay[oy1:oy2, ox1:ox2, :3]

        # Blend: output = alpha * overlay + (1 - alpha) * background
        frame_region = frame[y1:y2, x1:x2]
        blended = (alpha * overlay_rgb + (1 - alpha) * frame_region).astype(np.uint8)

        frame[y1:y2, x1:x2] = blended
        return frame
