"""
ai_models/preprocessing.py — Image loading, face detection, and preprocessing.

This file handles everything that happens BEFORE sending the image to any model:
  1. Open and validate the image
  2. Detect faces using OpenCV Haar Cascade
  3. Crop the largest face
  4. Save the cropped face to disk
  5. Generate a SHA-256 hash of the original image

We use OpenCV (cv2) for face detection because it is:
  - Free and open source
  - Works on Windows without any extra setup
  - Lightweight compared to deep learning face detectors
  - Good enough for academic projects

NOTE: cv2 and numpy are imported INSIDE functions (lazy imports) so that
Django can start up even if these packages are not yet installed.
Install them with: pip install opencv-python numpy Pillow
"""

import os
import logging

# Set up a simple logger so we can see what is happening in the terminal
logger = logging.getLogger(__name__)


# ============================================================
# FUNCTION 1: Calculate SHA-256 hash
# ============================================================
def calculate_sha256(file_path):
    """
    Generate a SHA-256 hash (digital fingerprint) of a file.

    SHA-256 produces a unique 64-character string for any given file.
    If even one pixel changes, the hash is completely different.

    This is used for:
      - Detecting exact duplicate uploads (same file, same hash)
      - Image integrity verification

    IMPORTANT: This does NOT detect whether an image is fake or real.
    It only checks if two files are byte-for-byte identical.

    Args:
        file_path (str): Absolute path to the image file.

    Returns:
        str: 64-character hex string, or empty string if an error occurs.
    """
    try:
        import hashlib as _hashlib
        # Create a SHA-256 hasher object
        sha256_hasher = _hashlib.sha256()

        # Read the file in chunks to handle large files without using too much RAM
        # 8192 bytes = 8 KB chunks
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256_hasher.update(chunk)

        return sha256_hasher.hexdigest()  # Returns the 64-character hash string

    except Exception as e:
        logger.error(f"SHA-256 calculation failed for {file_path}: {e}")
        return ""  # Return empty string on failure (do not crash)


# ============================================================
# FUNCTION 2: Detect faces in an image
# ============================================================
def detect_faces(image_path):
    """
    Use OpenCV Haar Cascade to find faces in an image.

    Haar Cascade is a classic computer vision algorithm for face detection.
    It looks for patterns of light and dark areas that match a face shape.

    Args:
        image_path (str): Absolute path to the image file.

    Returns:
        dict with keys:
          - success (bool):       True if processing worked (even if no face found)
          - face_detected (bool): True if at least one face was found
          - face_count (int):     Number of faces found
          - faces (list):         List of (x, y, w, h) rectangles for each face
          - error (str):          Error message if something went wrong
    """
    result = {
        'success': False,
        'face_detected': False,
        'face_count': 0,
        'faces': [],
        'image_bgr': None,
        'error': '',
    }

    try:
        import cv2  # Lazy import — only imported when this function is called
    except ImportError:
        result['error'] = "OpenCV (cv2) is not installed. Please install it with: pip install opencv-python"
        return result

    # --- Step 1: Load the image using OpenCV ---
    try:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            result['error'] = "Could not open image. The file may be corrupted or in an unsupported format."
            return result
    except Exception as e:
        result['error'] = f"Error loading image: {str(e)}"
        return result

    # --- Step 2: Load Haar Cascade face detectors (frontal + profile) ---
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    alt_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
    profile_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'

    if not os.path.exists(cascade_path):
        result['error'] = "OpenCV Haar Cascade file not found. Please reinstall opencv-python."
        return result

    cascades = [cv2.CascadeClassifier(cascade_path)]
    if os.path.exists(alt_path):
        cascades.append(cv2.CascadeClassifier(alt_path))

    prof_cascade = cv2.CascadeClassifier(profile_path) if os.path.exists(profile_path) else None

    # --- Step 3: Convert to grayscale for face detection ---
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    H, W = image_gray.shape[:2]

    # --- Step 4: Detect faces (frontal + profile left & right) ---
    rects = []
    for c in cascades:
        f = c.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(f) > 0:
            rects.extend(f)

    if prof_cascade is not None:
        f_prof = prof_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(f_prof) > 0:
            rects.extend(f_prof)

        gray_flipped = cv2.flip(image_gray, 1)
        f_prof_flipped = prof_cascade.detectMultiScale(gray_flipped, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(f_prof_flipped) > 0:
            for (fx, fy, fw, fh) in f_prof_flipped:
                x_orig = W - (fx + fw)
                rects.append([x_orig, fy, fw, fh])

    result['success'] = True
    result['image_bgr'] = image_bgr

    if not rects:
        result['face_detected'] = False
        result['face_count'] = 0
        result['faces'] = []
        return result

    # Deduplicate with NMS
    import numpy as np
    boxes = [[int(x), int(y), int(w), int(h)] for x, y, w, h in rects]
    scores = [float(w * h) for x, y, w, h in rects]
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=0.3)

    final_boxes = []
    if len(indices) > 0:
        for idx in np.asarray(indices).flatten():
            final_boxes.append(boxes[idx])

    final_boxes = sorted(final_boxes, key=lambda b: b[2] * b[3], reverse=True)

    result['face_detected'] = len(final_boxes) > 0
    result['face_count'] = len(final_boxes)
    result['faces'] = final_boxes
    return result


# ============================================================
# FUNCTION 3: Crop the largest face from the image
# ============================================================
def crop_largest_face(image_bgr, faces, padding_ratio=0.2):
    """
    Crop the largest face from the image with some padding around it.

    When multiple faces are found, we use the largest one for analysis.
    We add padding (extra space around the face) so the model gets
    some context around the face, not just the face outline alone.

    Args:
        image_bgr (ndarray): Original image as OpenCV array (BGR format).
        faces (list):        List of (x, y, w, h) face rectangles.
        padding_ratio (float): How much extra space to add (0.2 = 20% on each side).

    Returns:
        ndarray: Cropped face image, or None if cropping fails.
    """

    if len(faces) == 0:
        return None

    # Find the largest face by area (width * height)
    # In Python, max() with key= picks the item with the highest key value
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face

    # Add padding around the face
    img_height, img_width = image_bgr.shape[:2]

    padding_x = int(w * padding_ratio)
    padding_y = int(h * padding_ratio)

    # Calculate new boundaries, making sure they don't go outside the image
    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(img_width, x + w + padding_x)
    y2 = min(img_height, y + h + padding_y)

    # Crop the face region from the original image
    face_crop = image_bgr[y1:y2, x1:x2]

    return face_crop


# ============================================================
# FUNCTION 4: Save cropped face to disk
# ============================================================
def save_cropped_face(face_bgr, save_folder, filename_prefix="face"):
    """
    Save a cropped face image (OpenCV BGR array) to a folder.

    Args:
        face_bgr (ndarray): The cropped face as an OpenCV image array.
        save_folder (str):  Folder where the face image will be saved.
        filename_prefix (str): Prefix for the filename.

    Returns:
        str: Full path to the saved face image, or None if saving failed.
    """
    import uuid
    import cv2

    if face_bgr is None:
        return None

    # Create the folder if it does not exist
    os.makedirs(save_folder, exist_ok=True)

    # Generate a unique filename for this face crop
    filename = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.jpg"
    save_path = os.path.join(save_folder, filename)

    try:
        # cv2.imwrite saves the image to disk
        success = cv2.imwrite(save_path, face_bgr)
        if success:
            return save_path
        else:
            logger.error(f"cv2.imwrite failed to save face to {save_path}")
            return None
    except Exception as e:
        logger.error(f"Error saving face image: {e}")
        return None


# ============================================================
# FUNCTION 5: Prepare image for a model (resize + normalize)
# ============================================================
def prepare_image_for_model(image_path_or_array, target_size=(224, 224), normalize=True):
    """
    Resize and normalize an image for input to a deep learning model.

    Most models expect:
      - A specific input size (e.g. 224x224 pixels)
      - Pixel values in range [0, 1] instead of [0, 255]

    Args:
        image_path_or_array: Either a file path (str) or an OpenCV array (ndarray).
        target_size (tuple): (width, height) to resize to.
        normalize (bool): If True, divide pixel values by 255 to get [0, 1] range.

    Returns:
        ndarray: Preprocessed image as a NumPy array with shape (1, H, W, 3)
                 The leading '1' is the batch dimension required by Keras/TF.
        None: If preprocessing fails.
    """
    try:
        import numpy as np
        import cv2
        from PIL import Image

        # --- Load the image ---
        if isinstance(image_path_or_array, str):
            # It's a file path — load with PIL (handles JPG, PNG, etc.)
            img = Image.open(image_path_or_array).convert('RGB')
        elif isinstance(image_path_or_array, np.ndarray):
            # It's an OpenCV array (BGR) — convert to RGB then PIL
            img_rgb = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
        else:
            logger.error("prepare_image_for_model: unknown input type")
            return None

        # --- Resize ---
        width, height = target_size
        img = img.resize((width, height), Image.LANCZOS)

        # --- Convert to NumPy array ---
        img_array = np.array(img, dtype=np.float32)

        # --- Normalize ---
        if normalize:
            img_array = img_array / 255.0  # Scale from [0, 255] to [0.0, 1.0]

        # --- Add batch dimension ---
        # Most Keras/TF models expect shape: (batch_size, height, width, channels)
        # np.expand_dims adds an extra dimension at position 0
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, H, W, 3)

        return img_array

    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return None


# ============================================================
# FUNCTION 6: Full preprocessing pipeline
# ============================================================
def run_full_preprocessing(image_path, faces_save_folder):
    """
    Run the complete preprocessing pipeline for one uploaded image.

    Steps:
      1. Calculate SHA-256 hash
      2. Detect faces
      3. Crop the largest face
      4. Save the cropped face

    Args:
        image_path (str):        Full path to the uploaded original image.
        faces_save_folder (str): Folder to save cropped face images.

    Returns:
        dict with all preprocessing results.
    """
    output = {
        'sha256_hash': '',
        'face_detected': False,
        'face_count': 0,
        'cropped_face_path': None,
        'face_bgr': None,
        'error': '',
        'success': False,
    }

    # Step 1: SHA-256
    output['sha256_hash'] = calculate_sha256(image_path)

    # Step 2: Face detection
    detection = detect_faces(image_path)
    if not detection['success']:
        output['error'] = detection['error']
        return output

    output['face_detected'] = detection['face_detected']
    output['face_count'] = detection['face_count']

    if not detection['face_detected']:
        output['error'] = "No human face was detected in this image."
        output['success'] = True  # Preprocessing itself succeeded, just no face
        return output

    # Step 3: Crop the largest face
    face_bgr = crop_largest_face(detection['image_bgr'], detection['faces'])
    if face_bgr is None:
        output['error'] = "Face was detected but could not be cropped."
        return output

    output['face_bgr'] = face_bgr

    # Step 4: Save the cropped face
    saved_path = save_cropped_face(face_bgr, faces_save_folder)
    output['cropped_face_path'] = saved_path
    output['success'] = True

    return output
