"""Image upload page — INTERFACE ONLY (UI slice).

The page validates the uploaded image (real content, not only the extension), remembers a few facts about it
in the session (file name, size, real SHA-256) and sends the user to the result page.
No model is run, the image itself is not stored, and nothing is written to the database.
"""
import hashlib

from django import forms
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

MAX_BYTES = 10 * 1024 * 1024
# first bytes of real image files (extension alone is not trusted)
MAGIC = {'image/jpeg': (b'\xff\xd8\xff',), 'image/png': (b'\x89PNG\r\n\x1a\n',), 'image/webp': (b'RIFF',)}


class UploadForm(forms.Form):
    image = forms.FileField(label='Image')

    def clean_image(self):
        f = self.cleaned_data['image']
        if f.size > MAX_BYTES:
            raise forms.ValidationError('Image is larger than 10 MB.')
        head = f.read(12); f.seek(0)
        kind = next((k for k, sigs in MAGIC.items() if any(head.startswith(s) for s in sigs)), None)
        if kind is None or (kind == 'image/webp' and head[8:12] != b'WEBP'):
            raise forms.ValidationError('Only JPG, PNG or WEBP images are accepted.')
        f.content_type_checked = kind
        return f


@login_required
def analyze_view(request):
    form = UploadForm(request.POST or None, request.FILES or None)
    if request.method == 'POST' and form.is_valid():
        f = form.cleaned_data['image']
        digest = hashlib.sha256()
        for chunk in f.chunks():
            digest.update(chunk)
        # only small facts go into the session; the picture itself is previewed in the browser only
        request.session['last_upload'] = {
            'filename': f.name[:120],
            'size': f.size,
            'content_type': f.content_type_checked,
            'sha256': digest.hexdigest(),
        }
        return redirect('result')
    return render(request, 'analyze.html', {'form': form})
