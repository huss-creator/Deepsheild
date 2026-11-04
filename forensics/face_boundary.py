"""
forensics/face_boundary.py — Face Boundary / Blending Analysis

What does this look for?
    Classic face-swap (and many deepfake) pipelines render or warp a face
    and then blend it into the target photo along the face's outline.
    Even good blending tends to leave one of three traces right at that
    seam:
        1. A sharpness mismatch — the swapped face is often a different
           resolution/blur level than the frame it was pasted into.
        2. A colour/illumination jump — the pasted face's lighting was
           captured (or rendered) under different conditions than the
           background it now sits in.
        3. An unnatural gradient spike exactly along the boundary
           contour — real face-to-background transitions are gradual;
           a hard paste leaves a thin ring of unusually high edge energy.

Algorithm:
    1. Detect the face with the same OpenCV Haar Cascade the rest of the
       project already uses (ai_models/preprocessing.py).
    2. Build three concentric zones around the detected face box:
       inner (face interior), boundary ring (just outside the box), and
       outer (background, further out).
    3. Compare Laplacian variance (focus/sharpness), Sobel gradient
       energy, and mean colour between the three zones.
    4. Combine the three signals into one 0-100 anomaly score.

IMPORTANT DISCLAIMER:
    A real photo of a real face will also show SOME boundary contrast —
    hair, ears, and shadows aren't uniform. This module flags unusually
    STRONG boundary signals relative to the image's own statistics, not
    the mere presence of an edge. It is supporting evidence only.

Requirements:
    pip install opencv-python numpy
"""

import os
import uuid
import logging

logger = logging.getLogger(__name__)

BOUNDARY_LOW_THRESHOLD    = 30.0
BOUNDARY_MEDIUM_THRESHOLD = 55.0


def run_face_boundary_analysis(image_path, save_folder):
    """
    Analyse the boundary around a detected face for blending artifacts.

    Args:
        image_path (str):  Absolute path to the image.
        save_folder (str): Folder to save the annotated visualization.

    Returns:
        dict:
            face_boundary_image_path (str or None)
            face_boundary_score (float):   0-100 anomaly score
            face_boundary_level (str):     LOW / MEDIUM / HIGH / UNAVAILABLE
            face_detected (bool)
            face_boundary_sharpness_ratio (float)
            face_boundary_color_jump (float)
            face_boundary_interpretation (str)
            success (bool)
            error (str)
    """
    result = {
        'face_boundary_image_path': None,
        'face_boundary_score': 0.0,
        'face_boundary_level': 'UNAVAILABLE',
        'face_detected': False,
        'face_boundary_sharpness_ratio': 0.0,
        'face_boundary_color_jump': 0.0,
        'face_boundary_interpretation': 'No face was detected, so boundary analysis could not run.',
        'success': False,
        'error': '',
    }

    try:
        import cv2  # noqa: F401  (availability probe)
        import numpy as np  # noqa: F401
    except ImportError as e:
        result['error'] = f"Face boundary analysis requires OpenCV and NumPy: {e}"
        return result

    try:
        from ai_models.preprocessing import detect_faces
        face_result = detect_faces(image_path)
    except Exception as e:
        result['error'] = f"Face detection step failed: {e}"
        return result

    if not face_result.get('success') or not face_result.get('face_detected'):
        result['success'] = True  # ran fine, there just wasn't a face
        return result

    try:
        image_bgr = face_result['image_bgr']
        if image_bgr is None:
            image_bgr = __import__('cv2').imread(image_path)

        # use the largest detected face (by area) if there's more than one
        faces = face_result['faces']
        bbox = max(faces, key=lambda f: f[2] * f[3])

        scores = _score_boundary(image_bgr, bbox)

        overall = scores['boundary_score']
        # Approximate method: contrast across a ring around the detector's RECTANGLE, not the true
        # face contour, so natural sharpness/lighting/background changes also score. Level is capped
        # at MEDIUM and thresholds are uncalibrated defaults.
        level = 'LOW' if overall < BOUNDARY_LOW_THRESHOLD else 'MEDIUM'

        vis_path = _save_visualization(image_bgr, bbox, scores, level, save_folder)

        interpretation = _get_interpretation(level, scores)

        result.update({
            'face_boundary_image_path': vis_path,
            'face_boundary_score': round(overall, 2),
            'face_boundary_level': level,
            'face_detected': True,
            'face_boundary_sharpness_ratio': round(scores['sharpness_ratio'], 3),
            'face_boundary_color_jump': round(scores['color_jump'], 3),
            'face_boundary_interpretation': interpretation,
            'success': True,
            'error': '',
        })

        logger.info(f"Face boundary analysis complete: score={overall:.1f}, level={level}")

    except Exception as e:
        msg = f"Face boundary analysis failed: {e}"
        logger.error(msg)
        result['error'] = msg

    return result


def _score_boundary(image_bgr, bbox):
    """
    Core scoring logic — kept separate from face-detection so it can be
    unit-tested directly against a known bounding box.

    Args:
        image_bgr: OpenCV BGR image array.
        bbox:      (x, y, w, h) face rectangle.

    Returns:
        dict with sharpness_ratio, color_jump, gradient_ratio, boundary_score
    """
    import cv2
    import numpy as np

    x, y, w, h = [int(v) for v in bbox]
    H, W = image_bgr.shape[:2]

    # Padding for the boundary ring / outer background zone, scaled to face size
    pad = max(6, int(round(0.18 * max(w, h))))
    outer_pad = pad * 2

    def clamp_box(x0, y0, x1, y1):
        return (max(0, x0), max(0, y0), min(W, x1), min(H, y1))

    inner_box   = clamp_box(x + int(0.08*w), y + int(0.08*h), x + w - int(0.08*w), y + h - int(0.08*h))
    boundary_out = clamp_box(x - pad, y - pad, x + w + pad, y + h + pad)
    outer_out    = clamp_box(x - outer_pad, y - outer_pad, x + w + outer_pad, y + h + outer_pad)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    def region(arr, box):
        bx0, by0, bx1, by1 = box
        return arr[by0:by1, bx0:bx1]

    inner_gray = region(gray, inner_box)

    # "ring" regions = outer box minus inner box, extracted via masks
    def ring_mask(outer_box, inner_box, shape):
        mask = np.zeros(shape, dtype=np.uint8)
        ox0, oy0, ox1, oy1 = outer_box
        ix0, iy0, ix1, iy1 = inner_box
        mask[oy0:oy1, ox0:ox1] = 1
        mask[iy0:iy1, ix0:ix1] = 0
        return mask

    boundary_mask = ring_mask(boundary_out, (x, y, x + w, y + h), gray.shape)
    outer_mask    = ring_mask(outer_out, boundary_out, gray.shape)

    # --- 1. Sharpness (Laplacian variance = focus measure) ---
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    inner_lap_var = float(np.var(region(lap, inner_box))) if inner_gray.size else 0.0
    outer_pixels = lap[outer_mask == 1]
    outer_lap_var = float(np.var(outer_pixels)) if outer_pixels.size > 20 else inner_lap_var

    sharpness_ratio = (inner_lap_var + 1e-6) / (outer_lap_var + 1e-6)
    # anomaly grows the further this ratio is from 1.0 (either much sharper OR much blurrier)
    sharpness_anomaly = min(1.0, abs(np.log(sharpness_ratio + 1e-6)) / np.log(6.0))

    # --- 2. Gradient energy right on the boundary ring vs elsewhere ---
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.hypot(sobel_x, sobel_y)

    boundary_pixels = grad_mag[boundary_mask == 1]
    outer_grad_pixels = grad_mag[outer_mask == 1]
    boundary_grad_mean = float(np.mean(boundary_pixels)) if boundary_pixels.size > 20 else 0.0
    outer_grad_mean = float(np.mean(outer_grad_pixels)) if outer_grad_pixels.size > 20 else (boundary_grad_mean + 1e-6)

    gradient_ratio = (boundary_grad_mean + 1e-6) / (outer_grad_mean + 1e-6)
    gradient_anomaly = min(1.0, max(0.0, (gradient_ratio - 1.0) / 2.5))

    # --- 3. Colour / illumination jump across the boundary ---
    inner_bgr = region(image_bgr, inner_box).reshape(-1, 3).astype(np.float32)
    outer_bgr_pixels = image_bgr[outer_mask == 1].astype(np.float32)

    if inner_bgr.size and outer_bgr_pixels.size:
        inner_mean_color = inner_bgr.mean(axis=0)
        outer_mean_color = outer_bgr_pixels.mean(axis=0)
        color_jump = float(np.linalg.norm(inner_mean_color - outer_mean_color))
    else:
        color_jump = 0.0

    # Normalize color jump against overall image color variability so a
    # naturally colourful scene doesn't get flagged just for being colourful
    overall_std = float(np.std(image_bgr.astype(np.float32)))
    color_anomaly = min(1.0, color_jump / max(1.0, overall_std * 1.5))

    # --- combine ---
    boundary_score = 100.0 * (
        0.40 * sharpness_anomaly +
        0.35 * gradient_anomaly +
        0.25 * color_anomaly
    )

    return {
        'sharpness_ratio': sharpness_ratio,
        'gradient_ratio': gradient_ratio,
        'color_jump': color_jump,
        'boundary_score': boundary_score,
    }


def _save_visualization(image_bgr, bbox, scores, level, save_folder):
    try:
        import cv2
        x, y, w, h = [int(v) for v in bbox]
        vis = image_bgr.copy()

        color = {
            'LOW':    (0, 200, 0),
            'MEDIUM': (0, 200, 255),
            'HIGH':   (0, 0, 255),
        }.get(level, (200, 200, 200))

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        pad = max(6, int(round(0.18 * max(w, h))))
        cv2.rectangle(vis, (x - pad, y - pad), (x + w + pad, y + h + pad), color, 1)

        label = f"boundary score {scores['boundary_score']:.0f}"
        cv2.putText(vis, label, (x, max(15, y - pad - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        os.makedirs(save_folder, exist_ok=True)
        filename = f"faceboundary_{uuid.uuid4().hex[:10]}.jpg"
        save_path = os.path.join(save_folder, filename)
        cv2.imwrite(save_path, vis)
        return save_path
    except Exception as e:
        logger.error(f"Could not save face boundary visualization: {e}")
        return None


def _get_interpretation(level, scores):
    base = (
        f"Face Boundary Score: {scores['boundary_score']:.1f}/100 "
        f"(sharpness ratio {scores['sharpness_ratio']:.2f}, "
        f"gradient ratio {scores['gradient_ratio']:.2f}, "
        f"colour jump {scores['color_jump']:.1f}). "
    )

    if level == 'LOW':
        return (
            base +
            "The transition between the face region and its surroundings looks "
            "consistent with a naturally captured photo — no strong sharpness, "
            "edge, or colour discontinuity was found at the face boundary. "
            "This is supporting evidence only."
        )
    elif level == 'MEDIUM':
        return (
            base +
            "Some discontinuity was found between the face region and its "
            "surroundings (in sharpness, edge strength, or colour). This can "
            "occur naturally (makeup, lighting, depth-of-field) but is also "
            "consistent with a blended/composited face. This is supporting "
            "evidence only."
        )
    else:
        return (
            base +
            "A strong discontinuity was found right at the face boundary — "
            "the face region differs noticeably from its surroundings in "
            "sharpness and/or colour and/or edge energy. This pattern is "
            "commonly seen where a face has been rendered or captured "
            "separately and blended into the frame. This is supporting "
            "evidence only, not conclusive proof."
        )
