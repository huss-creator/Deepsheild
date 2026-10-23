"""Result page — INTERFACE ONLY (UI slice).

Shows how the final DeepShield result screen looks. The verdict, scores and forensic levels below are FIXED SAMPLE
values and the page says so. Only the file facts (name, size, type, SHA-256) are real, taken from the upload step.
The real detection pipeline lives in the main DeepShield project.
"""
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

SAMPLE = {
    'is_fake': True,
    'percent': 82,
    'face_found': True,
    'votes': {'authentic': 0, 'manipulation': 3, 'executed': 3, 'unavailable': 3},
    'models': [
        {'name': 'EfficientNet-B4', 'type': 'CNN', 'prediction': 'FAKE', 'score': 0.87},
        {'name': 'XceptionNet', 'type': 'CNN', 'prediction': 'FAKE', 'score': 0.81},
        {'name': 'ViT-B/16', 'type': 'Transformer', 'prediction': 'FAKE', 'score': 0.78},
    ],
    'forensics': [
        {'label': 'Error level analysis (ELA)', 'icon': 'bi-layers', 'value': 62, 'level': 'Medium'},
        {'label': 'Frequency analysis', 'icon': 'bi-soundwave', 'value': 71, 'level': 'High'},
        {'label': 'Noise consistency', 'icon': 'bi-grid-3x3', 'value': 48, 'level': 'Medium'},
        {'label': 'Face boundary', 'icon': 'bi-person-bounding-box', 'value': 55, 'level': 'Medium'},
        {'label': 'Metadata', 'icon': 'bi-file-earmark-text', 'value': 20, 'level': 'Low'},
    ],
}


def _size_text(n):
    return f'{n / 1048576:.2f} MB' if n >= 1048576 else f'{n / 1024:.1f} KB'


@login_required
def result_view(request):
    upload = request.session.get('last_upload')
    if not upload:                       # opened directly, nothing uploaded yet
        return redirect('analyze')
    ctx = {'result': SAMPLE, 'file': dict(upload, size_text=_size_text(upload['size']))}
    return render(request, 'result.html', ctx)
