"""What each forensic test measures, cannot prove, needs, and how it fails (audit record).
Used verbatim in reports/UI so wording stays consistent. `reliability` reflects the synthetic
benchmark in docs/forensic_validation.md (not real-world accuracy)."""

CATALOG = {
    'ela': dict(
        name='Error Level Analysis (ELA)', reliability='low', max_severity='info',
        measures='How strongly regions change when a JPEG is re-compressed once at fixed quality.',
        cannot_prove='Manipulation. Texture, edges and original save quality also change the error level.',
        inputs='JPEG images only (not applicable to PNG/lossless).',
        failure='Non-JPEG input; very small images.',
        false_positive='Textured/edge-rich areas and mixed-quality sources look anomalous.',
        false_negative='Edits followed by recompression at similar quality leave no contrast.',
        cost='Low (one JPEG encode + block statistics).'),
    'frequency': dict(
        name='Frequency-domain (FFT) analysis', reliability='low', max_severity='low',
        measures='Narrow periodic peaks in the 2-D spectrum (e.g. up-sampling traces).',
        cannot_prove='That an image is generated; many natural/processed images contain peaks.',
        inputs='Any image >=128 px; best on lossless input.',
        failure='Tiny images.',
        false_positive='Regular textures, sharpening, screen patterns.',
        false_negative='Generator artefacts at JPEG-grid frequencies are invisible in JPEG files; resized images.',
        cost='Medium (2-D FFT + median filter on a <=512 px window).'),
    'noise': dict(
        name='Noise residual analysis', reliability='low', max_severity='low',
        measures='Statistics/consistency of the high-pass noise residual across the image.',
        cannot_prove='Manipulation; denoised, compressed or phone-processed photos have unusual noise.',
        inputs='Any image.', failure='Very small or flat images.',
        false_positive='Heavily denoised/compressed phone images.', false_negative='Noise-matched edits.',
        cost='Low.'),
    'copy_move': dict(
        name='Copy-move (clone) detection', reliability='moderate', max_severity='moderate',
        measures='Region pairs explained by one geometric transform (RANSAC-verified keypoint matches).',
        cannot_prove='Intent; regions can repeat naturally (tiles, fences, symmetric objects).',
        inputs='Textured images with enough keypoints.', failure='Smooth/small images (reported as no evidence).',
        false_positive='Repeated real-world structures.', false_negative='Clones of smooth regions; heavy retouching.',
        cost='Medium (ORB + matching + RANSAC).'),
    'resampling': dict(
        name='Resampling / interpolation analysis', reliability='low', max_severity='low',
        measures='Periodic correlations left by resizing/rotating (second-derivative periodicity).',
        cannot_prove='Manipulation; ordinary editing, screenshots and thumbnails are resampled too.',
        inputs='Any image >=96 px.', failure='Tiny images.',
        false_positive='JPEG/sharpening/regular textures.', false_negative='Down-scaling, strong compression.',
        cost='Low.'),
    'face_boundary': dict(
        name='Face boundary contrast (approximate)', reliability='low', max_severity='low',
        measures='Sharpness/edge/colour contrast across a ring around the detected face rectangle.',
        cannot_prove='Blending: it analyses the detector rectangle, not the true face contour.',
        inputs='A detected face.', failure='No face detected.',
        false_positive='Depth of field, lighting and makeup produce natural boundaries.',
        false_negative='Well-blended swaps.', cost='Low.'),
    'metadata': dict(
        name='Metadata (EXIF)', reliability='context', max_severity='info',
        measures='Camera/software/time tags embedded in the file.',
        cannot_prove='Authenticity or manipulation: metadata is trivially added, edited or stripped.',
        inputs='Files with metadata (JPEG/TIFF mainly).', failure='Stripped files (very common).',
        false_positive='Camera apps and editors both write software tags.', false_negative='Forged tags.',
        cost='Negligible.'),
    'c2pa': dict(
        name='Content Credentials (C2PA)', reliability='high_when_present', max_severity='info',
        measures='Whether a signed provenance manifest exists, parses and validates.',
        cannot_prove='That depicted content is genuine, or that an image without credentials is fake.',
        inputs='Files with an embedded manifest.', failure='Library not installed; unreadable manifest.',
        false_positive='n/a', false_negative='Most real images carry no credentials.', cost='Low.'),
    'sha256': dict(
        name='SHA-256 file hash', reliability='exact', max_severity='none',
        measures='Exact byte-level identity of the uploaded file.',
        cannot_prove='Anything about manipulation: it is an integrity/identification tool, not a detector.',
        inputs='Any file.', failure='None.', false_positive='n/a', false_negative='n/a', cost='Negligible.'),
    'gradcam': dict(
        name='Grad-CAM explanation', reliability='visual_aid', max_severity='none',
        measures='Regions that influenced a CNN model output.',
        cannot_prove='Manipulation or ground-truth localisation.', inputs='CNN models with gradients.',
        failure='Transformers/ONNX/SVM models (unavailable).',
        false_positive='Highlights can sit on irrelevant regions.', false_negative='n/a', cost='Medium.'),
}
