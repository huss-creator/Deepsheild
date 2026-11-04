"""
forensics/copy_move.py — Copy-Move Forgery Detection

What is Copy-Move Forgery?
    Someone copies a region of an image and pastes it elsewhere in the
    SAME image — usually to hide an object, or to duplicate something
    (e.g. cloning extra people into a crowd photo). Because the pasted
    patch comes from the same image, it shares the exact same noise,
    lighting, and compression history as the rest of the photo, which
    is exactly what makes it invisible to ELA and hard for the eye to
    spot — but not invisible to keypoint matching.

Algorithm (a real, standard technique — not a toy heuristic):
    1. Detect ORB keypoints + binary descriptors across the whole image.
    2. Match every keypoint against every OTHER keypoint in the same
       image (self-matching) using a Hamming-distance matcher.
    3. Throw away matches between keypoints that are simply close
       neighbours (textures repeat locally all the time — a brick wall
       is not a forgery). Only keep matches where the two points are
       far enough apart to be a genuine "this patch reappears elsewhere".
    4. For every surviving match, compute the shift vector (dx, dy)
       between the two keypoint locations. A copy-pasted region moves
       as a rigid block, so ALL of its matched keypoints share
       (approximately) the same shift vector. Genuine coincidental
       texture matches don't — their shift vectors point every which way.
    5. Cluster the shift vectors (simple grid-based binning). A cluster
       with many members = strong evidence of a duplicated, translated
       region. A weak/empty cluster = no copy-move pattern found.

VERIFICATION (this version): matches pass Lowe's ratio test (so repetitive textures that match
many places are discarded), then a RANSAC similarity transform (translation + rotation +
scale) must explain >= CM_MIN_INLIERS keypoint pairs spread over a non-trivial area before a
region pair is reported. Output includes source/target bounding boxes.

IMPORTANT DISCLAIMER:
    This flags PIXEL-LEVEL duplication within one image. It will not
    catch a copy-move that was also rotated/rescaled/heavily
    recompressed after pasting — the ORB descriptors become too
    different across large transforms. Like every module in this
    project, this is supporting evidence, not proof.

Requirements:
    pip install opencv-python numpy
"""

import logging
import os
import uuid

import cv2
import numpy as np

from forensics.common import cfg

logger = logging.getLogger('deepshield.forensics')

ORB_FEATURES = 3000
RATIO_TEST = 0.75           # Lowe ratio: best / second-best descriptor distance
MIN_SEPARATION = 24         # px — closer matches are neighbouring texture, not a relocated patch
RANSAC_REPROJ = 3.0         # px
MAX_REGIONS = 3
LIMITATIONS = ('Natural repetition (tiles, fences, crowds, symmetric objects) and stock textures can '
               'produce matching regions. Supporting indicator; visually verify the marked regions.')


def _verified_regions(keypoints, descriptors, min_inliers, min_hull_area):
    """Ratio-test matching + iterative RANSAC similarity verification."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(descriptors, descriptors, k=3)
    pairs = {}
    for group in knn:
        group = [m for m in group if m.trainIdx != m.queryIdx]      # drop the self match
        if len(group) < 2 or group[0].distance >= RATIO_TEST * group[1].distance:
            continue                                                 # ambiguous (repetitive) -> discard
        m = group[0]
        p, q = np.array(keypoints[m.queryIdx].pt), np.array(keypoints[m.trainIdx].pt)
        if np.hypot(*(p - q)) < MIN_SEPARATION:
            continue
        pairs[tuple(sorted((m.queryIdx, m.trainIdx)))] = (p, q, m.distance)
    regions = []
    remaining = dict(pairs)
    for _ in range(MAX_REGIONS):
        if len(remaining) < min_inliers:
            break
        ids = list(remaining)
        src = np.float32([remaining[i][0] for i in ids] + [remaining[i][1] for i in ids])
        dst = np.float32([remaining[i][1] for i in ids] + [remaining[i][0] for i in ids])
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ)
        if M is None:
            break
        inl = inl.ravel().astype(bool)
        k = len(ids)
        used = {ids[i % k] for i in range(2 * k) if inl[i]}
        a = np.float32([remaining[i][0] for i in used])      # one end of each pair
        b = np.float32([remaining[i][1] for i in used])
        if len(used) >= min_inliers:
            # orient: source = the set the forward transform maps FROM
            fwd = np.hypot(*(cv2.transform(a[None], M)[0] - b).T) < RANSAC_REPROJ
            S = np.where(fwd[:, None], a, b)
            T = np.where(fwd[:, None], b, a)
            hull_s = cv2.contourArea(cv2.convexHull(S))
            hull_t = cv2.contourArea(cv2.convexHull(T))
            if min(hull_s, hull_t) >= min_hull_area:            # reject collinear / tiny coincidences
                scale = float(np.hypot(M[0, 0], M[1, 0]))
                regions.append({
                    'inliers': len(used), 'scale': round(scale, 3),
                    'rotation_deg': round(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), 1),
                    'source_points': S, 'target_points': T,
                    'mean_descriptor_distance': round(float(np.mean([remaining[i][2] for i in used])), 1)})
        for i in used:
            remaining.pop(i, None)
    return regions, len(pairs)


def _bbox(points, scale, pad=6):
    x0, y0 = points.min(axis=0) / scale - pad
    x1, y1 = points.max(axis=0) / scale + pad
    return [int(max(0, x0)), int(max(0, y0)), int(x1 - x0), int(y1 - y0)]


def run_copy_move_analysis(image_path, save_folder):
    result = {'copy_move_image_path': None, 'copy_move_score': 0.0, 'copy_move_level': 'UNAVAILABLE',
              'copy_move_cluster_size': 0, 'copy_move_region_count': 0,
              'copy_move_interpretation': 'Copy-move analysis was not available.',
              'success': False, 'error': '', 'details': {'regions': []}}
    try:
        bgr = cv2.imread(image_path)
        if bgr is None:
            result['error'] = 'Could not open image for copy-move analysis.'
            return result
        h, w = bgr.shape[:2]
        scale = 1000.0 / max(h, w) if max(h, w) > 1000 else 1.0
        work = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale != 1.0 else bgr
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        kps, desc = cv2.ORB_create(nfeatures=ORB_FEATURES).detectAndCompute(gray, None)
        min_inl = int(cfg('CM_MIN_INLIERS', 10))
        if desc is None or len(kps) < 30:
            result.update(success=True, copy_move_level='LOW', copy_move_interpretation=(
                'Too few distinctive keypoints for reliable copy-move matching (image too smooth or small). '
                'No duplication evidence to report.'))
            return result
        regions, n_pairs = _verified_regions(kps, desc, min_inl, float(cfg('CM_MIN_HULL_AREA', 900)))
        best = max((r['inliers'] for r in regions), default=0)
        level = 'LOW' if best < min_inl else ('MEDIUM' if best < int(cfg('CM_HIGH_INLIERS', 30)) else 'HIGH')
        vis_path, report = None, []
        vis = work.copy()
        for r in regions:
            report.append({'source_bbox': _bbox(r['source_points'], scale), 'target_bbox': _bbox(r['target_points'], scale),
                           'inliers': r['inliers'], 'scale': r['scale'], 'rotation_deg': r['rotation_deg'],
                           'mean_descriptor_distance': r['mean_descriptor_distance']})
            for pts, col in ((r['source_points'], (0, 255, 255)), (r['target_points'], (0, 0, 255))):
                x, y, bw, bh = cv2.boundingRect(pts.astype(np.int32))
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), col, 2)
            for a, b in zip(r['source_points'], r['target_points']):
                cv2.line(vis, tuple(int(v) for v in a), tuple(int(v) for v in b), (0, 255, 0), 1, cv2.LINE_AA)
        if regions:
            os.makedirs(save_folder, exist_ok=True)
            vis_path = os.path.join(save_folder, f'copymove_{uuid.uuid4().hex[:10]}.png')
            cv2.imwrite(vis_path, vis)
        result.update(
            copy_move_image_path=vis_path, copy_move_score=round(min(100.0, best / int(cfg('CM_HIGH_INLIERS', 30)) * 100), 1),
            copy_move_level=level, copy_move_cluster_size=best, copy_move_region_count=len(regions), success=True,
            copy_move_interpretation=(
                (f'{len(regions)} region pair(s) are explained by one geometric transform '
                 f'(best: {best} verified keypoint pairs). ' if regions else
                 'No region pair was geometrically verified as a duplicate. ') + LIMITATIONS),
            details={'regions': report, 'candidate_pairs': n_pairs, 'min_inliers': min_inl,
                     'reliability': 'moderate', 'limitations': LIMITATIONS})
    except Exception:
        logger.exception('copy-move analysis failed')
        result['error'] = 'Copy-move analysis failed.'
    return result
