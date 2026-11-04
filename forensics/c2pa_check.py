"""
forensics/c2pa_check.py — C2PA Content Credentials Verification

What is C2PA?
    The Coalition for Content Provenance and Authenticity (C2PA) is an
    industry standard (backed by Adobe, Microsoft, camera makers, news
    organizations, etc.) for attaching a cryptographically signed
    "manifest" to an image — recording where it came from, what
    software touched it, and whether it was AI-generated. Unlike ELA or
    noise analysis, this isn't a statistical guess: if a valid manifest
    is present and its signature checks out, that's real, verifiable
    provenance information, not an inference.

    The catch: C2PA is still an emerging standard. Most images —
    including the vast majority of casual photos and screenshots you'll
    test this project with — simply don't have a manifest at all yet.
    That is NOT a sign of manipulation; it just means no provenance
    data exists to check. Absence of a manifest is reported honestly
    as "not present", never upgraded into an anomaly score.

This module uses the official `c2pa-python` library (the same C2PA SDK
used by Adobe/Content Authenticity Initiative tools) to actually read
and validate any embedded manifest — it is a real integration, not a
placeholder.

Requirements:
    pip install c2pa-python
    (If this package is not installed, the module reports that clearly
    instead of pretending to have checked anything.)
"""

import json
import logging

logger = logging.getLogger(__name__)


def run_c2pa_check(image_path):
    """
    Check an image for an embedded C2PA Content Credentials manifest.

    Args:
        image_path (str): Absolute path to the image file.

    Returns:
        dict:
            c2pa_status (str):         'VALID' | 'INVALID' | 'NOT_PRESENT' | 'LIBRARY_MISSING' | 'ERROR'
            c2pa_has_manifest (bool)
            c2pa_is_valid (bool or None)
            c2pa_claim_generator (str)
            c2pa_validation_state (str)
            c2pa_manifest_summary (str)
            c2pa_interpretation (str)
            success (bool)
            error (str)
    """
    result = {
        'c2pa_status': 'ERROR',
        'c2pa_has_manifest': False,
        'c2pa_is_valid': None,
        'c2pa_claim_generator': '',
        'c2pa_validation_state': '',
        'c2pa_signer_trusted': False,
        'c2pa_manifest_summary': '',
        'c2pa_interpretation': '',
        'success': False,
        'error': '',
    }

    try:
        import c2pa
    except ImportError:
        result.update({
            'c2pa_status': 'LIBRARY_MISSING',
            'success': True,
            'c2pa_interpretation': (
                "The c2pa-python library is not installed, so Content Credentials "
                "could not be checked. Install it with: pip install c2pa-python"
            ),
        })
        return result

    try:
        with c2pa.Reader(image_path) as reader:
            manifest_json = reader.json()
            validation_state = reader.get_validation_state()
            is_valid = reader.is_valid()

        manifest = json.loads(manifest_json)
        active_manifest = _get_active_manifest(manifest)
        claim_generator = active_manifest.get('claim_generator', '') if active_manifest else ''
        title = active_manifest.get('title', '') if active_manifest else ''

        status = 'VALID' if is_valid else 'INVALID'
        trusted = str(validation_state or '').lower() == 'trusted'

        summary = _summarize_manifest(active_manifest, title, claim_generator)

        result.update({
            'c2pa_status': status,
            'c2pa_has_manifest': True,
            'c2pa_is_valid': bool(is_valid),
            'c2pa_claim_generator': claim_generator or 'Unknown',
            'c2pa_validation_state': validation_state or 'UNKNOWN',
            'c2pa_signer_trusted': trusted,
            'c2pa_manifest_summary': summary,
            'c2pa_interpretation': _get_interpretation(status, claim_generator, validation_state),
            'success': True,
            'error': '',
        })

        logger.info(f"C2PA check complete: status={status}, generator={claim_generator}")

    except Exception as e:
        status, has_manifest = classify_c2pa_error(e)
        if status == 'NOT_PRESENT':
            result.update({'c2pa_status': 'NOT_PRESENT', 'c2pa_has_manifest': False, 'success': True,
                           'c2pa_interpretation': (
                               'No valid Content Credentials were found. This is the normal case for most '
                               'images today and says nothing about whether the image is genuine or '
                               'manipulated.')})
        elif status == 'INVALID':
            result.update({'c2pa_status': 'INVALID', 'c2pa_has_manifest': True, 'c2pa_is_valid': False,
                           'c2pa_validation_state': 'Unreadable', 'success': True,
                           'c2pa_interpretation': (
                               'Content Credentials data is present but could not be parsed or validated. '
                               'The credential is therefore not usable as provenance evidence.')})
        else:
            logger.error('C2PA check failed: %s', type(e).__name__)
            result['error'] = 'C2PA check failed.'
            result['c2pa_interpretation'] = 'C2PA verification could not be completed due to an unexpected error.'
    return result


def classify_c2pa_error(exc):
    """Map a c2pa exception to ('NOT_PRESENT'|'INVALID'|'ERROR', has_manifest).
    Only explicit "no manifest" errors mean NOT_PRESENT; a corrupt/unreadable manifest is
    INVALID (it exists but cannot be trusted); anything else is a tool ERROR."""
    name, text = type(exc).__name__.lower(), str(exc).lower()
    if ('manifestnotfound' in name or 'manifestnotfound' in text or 'no jumbf' in text
            or 'no manifest' in text or 'no c2pa' in text or 'jumbf not found' in text):
        return 'NOT_PRESENT', False
    if any(t in text for t in ('claim', 'jumbf', 'cbor', 'signature', 'malformed', 'corrupt',
                               'assertion', 'hash mismatch', 'hashmismatch', 'manifest')):
        return 'INVALID', True
    return 'ERROR', False


def _get_active_manifest(manifest_store):
    """Pull out the active manifest dict from the manifest store JSON."""
    active_label = manifest_store.get('active_manifest')
    manifests = manifest_store.get('manifests', {})
    if active_label and active_label in manifests:
        return manifests[active_label]
    # fall back to the first manifest present, if any
    if manifests:
        return next(iter(manifests.values()))
    return None


def _summarize_manifest(active_manifest, title, claim_generator):
    if not active_manifest:
        return "Manifest present but could not be summarized."

    assertions = active_manifest.get('assertions', [])
    labels = [a.get('label', '') for a in assertions if isinstance(a, dict)]

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if claim_generator:
        parts.append(f"Generated by: {claim_generator}")
    if labels:
        parts.append(f"Assertions: {', '.join(labels[:6])}")

    return " | ".join(parts) if parts else "Manifest present, no readable fields."


def _get_interpretation(status, claim_generator, validation_state):
    if status == 'VALID':
        trusted = str(validation_state or '').lower() == 'trusted'
        return (
            f"A Content Credentials manifest was found and its cryptographic signature and file "
            f"binding validated (tool: {claim_generator or 'unknown'}; validation state: {validation_state}). "
            + ("The signer is on the configured trust list (verified). " if trusted else
               "The signer is NOT confirmed against a trust list, so this shows the record is intact, "
               "not who stands behind it. ")
            + "A valid credential documents the recorded history of the file; it does not by itself show "
              "that the depicted content is genuine. Read the recorded actions (e.g. AI generation, edits)."
        )
    return (
        f"A Content Credentials manifest was found but failed validation (state: {validation_state}). "
        "This can mean the file was modified after signing or the signature could not be verified; the "
        "credential cannot be used as provenance evidence."
    )
