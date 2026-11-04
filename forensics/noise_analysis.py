"""
forensics/noise_analysis.py — Image Noise Residual Analysis

What is Noise Analysis?
    Real photographs always contain a consistent noise pattern caused by the
    camera sensor. This is called "sensor noise" or "photo response non-uniformity".
    
    When an image is digitally generated or heavily processed:
        - The noise pattern may be missing entirely (too smooth)
        - The noise pattern may be inconsistent across regions (splicing)
        - The noise distribution may differ from natural camera noise
    
    Algorithm:
        1. Convert image to grayscale
        2. Apply Gaussian blur to approximate the "clean" image
        3. Subtract blurred from original → this isolates the noise residual
        4. Analyse the noise statistics:
            - Mean absolute noise (low = too smooth = suspicious)
            - Standard deviation (should follow certain distributions for real images)
            - Spatial consistency across image blocks

IMPORTANT DISCLAIMER:
    Noise analysis is supporting forensic evidence ONLY.
    - Over-sharpened real images can show high noise.
    - Heavily compressed images can show artificially low noise.
    - This is not a standalone deepfake detector.

Requirements:
    pip install numpy Pillow
    (Both already in requirements.txt)
"""

import os
import uuid
import logging

logger = logging.getLogger(__name__)

# Noise level thresholds (based on mean absolute noise value)
NOISE_LOW_MEAN    = 3.0    # Very smooth — possibly AI-generated or heavily processed
NOISE_HIGH_MEAN   = 25.0   # High noise — possibly manipulated or added noise
NOISE_IDEAL_MIN   = 3.0    # Natural image noise is usually above this
NOISE_IDEAL_MAX   = 20.0   # And below this

# Consistency threshold: std of block noise means
CONSISTENCY_THRESHOLD = 5.0  # Higher = more inconsistent noise across blocks


def run_noise_analysis(image_path, save_folder):
    """
    Run noise residual analysis on an image.

    Args:
        image_path (str):   Absolute path to the image file.
        save_folder (str):  Folder to save the noise visualization image.

    Returns:
        dict:
            noise_image_path (str or None): Path to saved noise visualization.
            noise_score (float):            Anomaly score (0-100).
            noise_level (str):              'LOW' | 'MEDIUM' | 'HIGH' | 'UNAVAILABLE'
            noise_consistency (str):        'CONSISTENT' | 'INCONSISTENT' | 'N/A'
            noise_mean (float):             Mean noise residual value.
            noise_std (float):              Noise standard deviation.
            noise_interpretation (str):     Human-readable explanation.
            success (bool):
            error (str):
    """
    result = {
        'noise_image_path': None,
        'noise_score': 0.0,
        'noise_level': 'UNAVAILABLE',
        'noise_consistency': 'N/A',
        'noise_mean': 0.0,
        'noise_std': 0.0,
        'noise_interpretation': 'Noise analysis was not available.',
        'success': False,
        'error': '',
    }

    try:
        import numpy as np
        from PIL import Image, ImageFilter

        # --- Step 1: Open and convert to grayscale ---
        img = Image.open(image_path).convert('L')

        # Resize for consistent analysis (avoid very large images)
        img = img.resize((512, 512), Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32)

        # --- Step 2: Apply Gaussian blur to get "smooth" version ---
        # The blur radius controls how much noise we extract
        # Radius 1 = capture fine noise, Radius 2 = coarser residual
        img_pil = Image.fromarray(img_array.astype(np.uint8))
        blurred_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=1))
        blurred_array = np.array(blurred_pil, dtype=np.float32)

        # --- Step 3: Compute noise residual ---
        # noise = original - blurred (difference between original and smooth)
        noise_residual = img_array - blurred_array

        # --- Step 4: Compute noise statistics ---
        noise_mean = float(np.mean(np.abs(noise_residual)))
        noise_std  = float(np.std(noise_residual))

        # --- Step 5: Analyse spatial consistency of noise ---
        # Divide the image into 4x4 blocks and compute mean noise per block
        block_size = 128  # 512/4 = 128 pixels per block
        block_means = []

        for row in range(0, 512, block_size):
            for col in range(0, 512, block_size):
                block = noise_residual[row:row+block_size, col:col+block_size]
                block_means.append(float(np.mean(np.abs(block))))

        block_std = float(np.std(block_means)) if block_means else 0.0

        # High block_std = noise is very different across regions = inconsistent
        noise_consistency = 'INCONSISTENT' if block_std > CONSISTENCY_THRESHOLD else 'CONSISTENT'

        # --- Step 6: Calculate anomaly score ---
        # Two factors contribute to anomaly:
        #   a) Noise mean too low (over-smooth, potentially AI-generated)
        #   b) Inconsistent noise across regions (potentially spliced)
        
        # Factor a: "too smooth" penalty
        smooth_penalty = 0.0
        if noise_mean < NOISE_LOW_MEAN:
            # Very smooth image: score based on how far below threshold
            smooth_penalty = min(60.0, (NOISE_LOW_MEAN - noise_mean) / NOISE_LOW_MEAN * 60.0)

        # Factor b: inconsistency penalty  
        consistency_penalty = min(40.0, (block_std / CONSISTENCY_THRESHOLD) * 40.0)

        noise_score = min(100.0, smooth_penalty + consistency_penalty)
        noise_score = round(noise_score, 2)

        # --- Step 7: Classify noise level ---
        if noise_mean < NOISE_LOW_MEAN:
            noise_level = 'HIGH'    # Suspiciously smooth
        elif noise_mean > NOISE_HIGH_MEAN:
            noise_level = 'HIGH'    # Unusually noisy
        elif noise_consistency == 'INCONSISTENT':
            noise_level = 'MEDIUM'
        elif noise_mean < NOISE_IDEAL_MAX:
            noise_level = 'LOW'     # Normal range
        else:
            noise_level = 'MEDIUM'

        # --- Step 8: Create noise visualization ---
        # Scale residual to visible range (0-255)
        noise_vis = np.abs(noise_residual)
        if noise_vis.max() > 0:
            noise_vis = (noise_vis / noise_vis.max() * 255).astype(np.uint8)
        else:
            noise_vis = noise_vis.astype(np.uint8)

        noise_image_path = _save_noise_image(noise_vis, save_folder)

        # --- Step 9: Generate interpretation ---
        noise_interpretation = _get_noise_interpretation(
            noise_level, noise_score, noise_mean, noise_consistency
        )

        result.update({
            'noise_image_path':    noise_image_path,
            'noise_score':         noise_score,
            'noise_level':         noise_level,
            'noise_consistency':   noise_consistency,
            'noise_mean':          round(noise_mean, 3),
            'noise_std':           round(noise_std, 3),
            'noise_interpretation': noise_interpretation,
            'success':             True,
            'error':               '',
        })

        logger.info(f"Noise analysis complete: mean={noise_mean:.2f}, level={noise_level}")

    except ImportError as e:
        msg = f"Noise analysis requires NumPy and Pillow: {e}"
        logger.error(msg)
        result['error'] = msg

    except Exception as e:
        msg = f"Noise analysis failed: {e}"
        logger.error(msg)
        result['error'] = msg

    return result


def _save_noise_image(noise_array, save_folder):
    """Save noise residual visualization. Returns path or None."""
    try:
        from PIL import Image
        os.makedirs(save_folder, exist_ok=True)
        filename  = f"noise_{uuid.uuid4().hex[:10]}.jpg"
        save_path = os.path.join(save_folder, filename)
        noise_img = Image.fromarray(noise_array, mode='L')
        noise_img = noise_img.resize((512, 512), Image.NEAREST)
        noise_img.save(save_path, 'JPEG', quality=90)
        return save_path
    except Exception as e:
        logger.error(f"Could not save noise image: {e}")
        return None


def _get_noise_interpretation(noise_level, noise_score, noise_mean, consistency):
    """Return a human-readable interpretation of noise analysis results."""
    base = f"Noise Anomaly Score: {noise_score:.1f}/100. Mean Residual: {noise_mean:.2f}. "

    if noise_mean < NOISE_LOW_MEAN:
        content = (
            "The image shows very low noise residuals, suggesting it may have been "
            "heavily smoothed, generated by an AI model, or processed by a denoising filter. "
            "Very smooth images with almost no sensor noise can be a characteristic of "
            "synthetic (AI-generated) images."
        )
    elif noise_mean > NOISE_HIGH_MEAN:
        content = (
            "The image exhibits high noise residuals. This may indicate added artificial "
            "noise, heavy sharpening, or image compression artifacts. High noise can "
            "sometimes be used to obscure manipulation traces."
        )
    else:
        content = (
            "The image shows noise residual levels within a typical range for "
            "natural photographic content."
        )

    consistency_note = ""
    if consistency == 'INCONSISTENT':
        consistency_note = (
            " Additionally, the noise pattern is spatially inconsistent — "
            "different regions of the image show significantly different noise levels. "
            "This may suggest that different regions have different processing histories, "
            "which could be consistent with image splicing or compositing."
        )

    return base + content + consistency_note + " This is supporting evidence only."
