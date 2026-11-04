# DeepShield: Deepfake & AI-Image Detection (FYP)

Django web app that checks whether a face photo is manipulated or AI-generated.
It finds the face (OpenCV YuNet), runs 5 AI models, runs 8 classical forensic tests
and shows an explainable result page. Scores are raw numbers, **not probabilities**.

## Models (5)
XceptionNet, EfficientNet-B4, ViT-B/16, FFT/DCT + SVM, **SDXL Detector** (counts double in the ensemble).

## Run (Python 3.12 recommended)
```
python -m venv venv
venv\Scripts\activate            # PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```
Open http://127.0.0.1:8000/ (redirects to `/public/`).
Optional (Grad-CAM): `pip install torch torchvision timm onnx2torch`.

## URLs
- `/public/` landing (dark only), `/public/login`, `/public/register`, `/public/limitations`
- `/auth/upload`, `/auth/history`, `/auth/result/<id>` ... (after login; dark + light theme)

## Folders
```
config/        Django settings and root urls
detector/      views, forms, models, templates logic, REST API, tests
ai_models/     face detection, model loading, ensemble, Grad-CAM
forensics/     the 8 forensic tests
templates/ static/
trained_models/  model files (keep face_detector/yunet.onnx!) , see its README
docs/          short notes on explainability, validation, model management
```

## Tests
`python manage.py test detector ai_models`

Disclaimer: results are AI estimates, not legal proof. Human review is recommended.
