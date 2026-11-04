"""
Face detection + alignment behind one interface.

Detectors are selected with settings.FACE_DETECTOR:
    haar   OpenCV Haar cascade (original behaviour; no landmarks, no confidence)
    yunet  OpenCV YuNet (needs the ONNX file at YUNET_MODEL_PATH; gives confidence
           and 5 landmarks, which enable eye-level alignment)
To add RetinaFace/MTCNN/MediaPipe, implement `detect(image_rgb) -> list[Face]` and
register it in DETECTORS.
"""

import logging
import math
import os
import threading
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from django.conf import settings

logger = logging.getLogger('deepshield.faces')


def _deduplicate_boxes(boxes, overlap_threshold=0.7):
    """Keep the largest box when two detectors report substantially the same face."""
    kept = []
    for box in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
        x, y, w, h = box
        area = w * h
        duplicate = False
        for kx, ky, kw, kh in kept:
            intersection_w = max(0, min(x + w, kx + kw) - max(x, kx))
            intersection_h = max(0, min(y + h, ky + kh) - max(y, ky))
            intersection = intersection_w * intersection_h
            if intersection / min(area, kw * kh) >= overlap_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


@dataclass
class Face:
    index: int
    x: int
    y: int
    w: int
    h: int
    confidence: Optional[float] = None          # None when the detector gives no score
    landmarks: Optional[np.ndarray] = None      # (5, 2): two eyes first, then nose, mouth corners

    @property
    def bbox(self):
        return (int(self.x), int(self.y), int(self.w), int(self.h))


class HaarDetector:
    name = 'haar'
    supports_landmarks = False

    def __init__(self):
        frontal_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        alt_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        profile_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
        eye_path = cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'

        if not os.path.exists(frontal_path):
            raise RuntimeError('OpenCV Haar cascade file is missing.')

        self._cascades = [cv2.CascadeClassifier(frontal_path)]
        if os.path.exists(alt_path):
            self._cascades.append(cv2.CascadeClassifier(alt_path))

        self._profile_cascade = cv2.CascadeClassifier(profile_path) if os.path.exists(profile_path) else None
        self._eye_cascade = cv2.CascadeClassifier(eye_path) if os.path.exists(eye_path) else None
        if self._eye_cascade is None or self._eye_cascade.empty():
            logger.warning('Haar eye cascade unavailable; false-positive face filtering is reduced.')

    def _has_eye_structure(self, gray, box, profile=False):
        if self._eye_cascade is None or self._eye_cascade.empty():
            return True
        x, y, w, h = box
        roi = gray[y:y + h, x:x + w]
        if roi.size == 0:
            return False
        upper = roi[:max(1, int(h * 0.68)), :]
        eye_min = max(4, int(min(w, h) * 0.08))
        eyes = self._eye_cascade.detectMultiScale(
            upper, scaleFactor=1.1, minNeighbors=4, minSize=(eye_min, eye_min))
        centers = []
        for ex, ey, ew, eh in eyes:
            cx, cy = ex + ew / 2, ey + eh / 2
            if 0.1 * h <= cy <= 0.65 * h:
                centers.append((cx, cy))
        if profile:
            return any(0.15 * w <= cx <= 0.85 * w for cx, _ in centers)
        return any(
            0.18 * w <= abs(x1 - x2) <= 0.75 * w and abs(y1 - y2) <= 0.25 * h
            for i, (x1, y1) in enumerate(centers)
            for x2, y2 in centers[i + 1:]
        )

    def detect(self, image_rgb: np.ndarray) -> List[Face]:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        m = settings.FACE_MIN_SIZE
        H, W = gray.shape[:2]
        image_area = H * W
        rects = []
        # Frontal detections require a more stable cascade vote and plausible eye placement.
        for c in self._cascades:
            f = c.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=settings.FACE_HAAR_MIN_NEIGHBORS,
                                   minSize=(m, m))
            if len(f) > 0:
                rects.extend((tuple(map(int, b)), False) for b in f)

        # Profile detections need one plausible eye; this rejects many background patches.
        if self._profile_cascade is not None:
            f_prof = self._profile_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=settings.FACE_HAAR_MIN_NEIGHBORS, minSize=(m, m))
            if len(f_prof) > 0:
                rects.extend((tuple(map(int, b)), True) for b in f_prof)

            gray_flipped = cv2.flip(gray, 1)
            f_prof_flipped = self._profile_cascade.detectMultiScale(
                gray_flipped, scaleFactor=1.1, minNeighbors=settings.FACE_HAAR_MIN_NEIGHBORS, minSize=(m, m))
            if len(f_prof_flipped) > 0:
                for (fx, fy, fw, fh) in f_prof_flipped:
                    x_orig = W - (fx + fw)
                    rects.append(((x_orig, fy, fw, fh), True))

        if not rects:
            return []

        # Filter by minimum area ratio and aspect ratio before duplicate suppression.
        # Faces are roughly square; reject anything with w/h < 0.5 or > 2.0.
        valid = []
        for (x, y, w, h), profile in rects:
            if w * h < image_area * settings.FACE_MIN_AREA_RATIO:
                continue
            ratio = w / h if h > 0 else 0.0
            if not (0.5 <= ratio <= 2.0):
                continue
            if settings.FACE_HAAR_VALIDATE_EYES and not self._has_eye_structure(
                    gray, (int(x), int(y), int(w), int(h)), profile):
                continue
            valid.append([int(x), int(y), int(w), int(h)])

        if not valid:
            return []

        final_boxes = _deduplicate_boxes(valid)
        return [Face(i, *b) for i, b in enumerate(final_boxes[:settings.AI_MAX_FACES])]


_YUNET = {}                       # path -> loaded cv2.FaceDetectorYN (the ONNX model is read only once)
_YUNET_LOCK = threading.Lock()    # one detection at a time per process (the detector object is not thread-safe)


class YuNetDetector:
    name = 'yunet'
    supports_landmarks = True

    def __init__(self):
        path = settings.YUNET_MODEL_PATH
        if not os.path.isfile(path):
            raise RuntimeError('FACE_DETECTOR=yunet but the YuNet model file was not found.')
        self._path = path
        cv2.setNumThreads(1)       # OpenCV must not use every CPU core, or the web server stops answering

    # Long side of the picture YuNet looks at. Big photos are shrunk (never enlarged); if nothing is
    # found a second, smaller pass helps with very large faces. Boxes are mapped back to full size.
    WORK_SIDES = (960, 480)

    def _net(self):
        net = _YUNET.get(self._path)
        if net is None:
            net = cv2.FaceDetectorYN.create(self._path, '', (320, 320), settings.FACE_CONFIDENCE_THRESHOLD, 0.3, 50)
            _YUNET[self._path] = net
        return net

    def _detect_at(self, image_rgb: np.ndarray, max_side: int):
        h, w = image_rgb.shape[:2]
        scale = min(1.0, max_side / float(max(h, w)))
        if scale < 1.0:
            work = cv2.resize(image_rgb, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                              interpolation=cv2.INTER_AREA)
        else:
            work, scale = image_rgb, 1.0
        wh, ww = work.shape[:2]
        with _YUNET_LOCK:
            net = self._net()
            net.setInputSize((ww, wh))       # same loaded model, new input size
            _, rows = net.detect(cv2.cvtColor(work, cv2.COLOR_RGB2BGR))
        if rows is None:
            return []
        out = []
        for r in rows:
            vals = r.astype(np.float64).copy()
            vals[:14] = vals[:14] / scale
            out.append(vals)
        return out

    def detect(self, image_rgb: np.ndarray) -> List[Face]:
        h, w = image_rgb.shape[:2]
        image_area = h * w
        rows = []
        for side in self.WORK_SIDES:
            if side > max(h, w) * 1.01 and side != self.WORK_SIDES[0]:
                continue                                   # never upscale beyond the original size
            rows = self._detect_at(image_rgb, side)
            if rows:
                break
        if not rows:
            return []
        boxes = []
        confidences = []
        landmarks = []
        for r in sorted(rows, key=lambda r: r[2] * r[3], reverse=True):
            x, y, bw, bh = (int(round(v)) for v in r[:4])
            x, y = max(0, x), max(0, y)
            bw, bh = min(bw, w - x), min(bh, h - y)
            if min(bw, bh) < settings.FACE_MIN_SIZE:
                continue
            confidence = float(r[14])
            if confidence < settings.FACE_CONFIDENCE_THRESHOLD:
                continue
            # YuNet's confidence is reliable, so only a very small area floor applies (a face in a
            # 12 MP photo can legitimately be well under 0.5 % of the frame).
            if bw * bh < image_area * min(settings.FACE_MIN_AREA_RATIO, 0.0005):
                continue
            ratio = bw / bh if bh > 0 else 0.0
            if not (0.5 <= ratio <= 2.0):
                continue
            boxes.append([x, y, bw, bh])
            confidences.append(confidence)
            landmarks.append(r[4:14].reshape(5, 2).copy())
        boxes = _deduplicate_boxes(boxes)
        faces = []
        for box in boxes[:settings.AI_MAX_FACES]:
            best = max(
                (i for i, candidate in enumerate(boxes) if candidate == box),
                key=lambda i: confidences[i],
            )
            faces.append(Face(len(faces), *box, confidences[best], landmarks[best]))
        return faces


DETECTORS = {'haar': HaarDetector, 'yunet': YuNetDetector}


def get_face_detector(name: Optional[str] = None):
    name = (name or settings.FACE_DETECTOR).lower()
    if name == 'auto':          # best available: YuNet (box + landmarks) when its file exists, else Haar
        name = 'yunet' if os.path.isfile(settings.YUNET_MODEL_PATH) else 'haar'
    if name not in DETECTORS:
        raise RuntimeError(f'Unknown FACE_DETECTOR {name!r}; options: {sorted(DETECTORS)}')
    return DETECTORS[name]()


# ------------------------------------------------------------------ alignment
def eye_angle_degrees(landmarks: np.ndarray) -> float:
    """Angle (cv2 convention: positive = counter-clockwise rotation to level the eyes)."""
    e1, e2 = sorted(landmarks[:2], key=lambda p: p[0])
    return math.degrees(math.atan2(e2[1] - e1[1], e2[0] - e1[0]))


def quality_flags(face: Face, image_shape) -> List[str]:
    H, W = image_shape[:2]
    flags = []
    if min(face.w, face.h) < settings.FACE_QUALITY_MIN_SIZE:
        flags.append('face_too_small')
    margin = max(2, int(0.01 * max(H, W)))
    if face.x <= margin or face.y <= margin or face.x + face.w >= W - margin or face.y + face.h >= H - margin:
        flags.append('face_near_image_edge')      # possibly partially visible
    return flags


def align_and_crop(image_rgb: np.ndarray, face: Face, padding: Optional[float] = None):
    """Return (crop_rgb, aligned: bool). Rotates the image about the face centre to level
    the eyes when landmarks exist (resampling the pixels); otherwise a plain padded crop."""
    padding = settings.FACE_PADDING if padding is None else padding
    H, W = image_rgb.shape[:2]
    cx, cy = face.x + face.w / 2.0, face.y + face.h / 2.0
    src, aligned = image_rgb, False
    if face.landmarks is not None:
        angle = eye_angle_degrees(face.landmarks)
        if abs(angle) >= 1.0:
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            src = cv2.warpAffine(image_rgb, M, (W, H), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REFLECT)
            aligned = True
    px, py = int(face.w * padding), int(face.h * padding)
    x1, y1 = max(0, face.x - px), max(0, face.y - py)
    x2, y2 = min(W, face.x + face.w + px), min(H, face.y + face.h + py)
    return src[y1:y2, x1:x2].copy(), aligned


def crop_geometry(image_shape, face: Face, padding: Optional[float] = None) -> dict:
    """Where align_and_crop() took its crop from, in original-image coordinates, so the UI can map
    anything drawn on the crop (heatmaps, suspicious regions) back onto the photo.
    x1..y2 is the crop rectangle in the (possibly eye-levelled) image; angle/cx/cy describe the
    rotation applied about the face centre (angle 0 = none)."""
    padding = settings.FACE_PADDING if padding is None else padding
    H, W = image_shape[:2]
    cx, cy = face.x + face.w / 2.0, face.y + face.h / 2.0
    angle = 0.0
    if face.landmarks is not None:
        a = eye_angle_degrees(face.landmarks)
        if abs(a) >= 1.0:
            angle = float(a)
    px, py = int(face.w * padding), int(face.h * padding)
    return {'x1': max(0, face.x - px), 'y1': max(0, face.y - py),
            'x2': min(W, face.x + face.w + px), 'y2': min(H, face.y + face.h + py),
            'angle': round(angle, 3), 'cx': round(cx, 2), 'cy': round(cy, 2)}
