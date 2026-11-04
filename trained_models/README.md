# trained_models/

Place your checkpoints here (they are git-ignored; never commit weights):

    xception/  efficientnet_b4/  vit_b16/
    face_detector/yunet.onnx   (optional, for FACE_DETECTOR=yunet)
    sdxl_detector/model.onnx   (optional; see sdxl_detector/README.md)

Then `cp models.example.json models.json`, edit each entry to match how the model was
trained (normalization, output_mode, fake_index, input_size, sha256) and run:

    python manage.py check_models

Supported formats: `.onnx` (any framework, recommended), `.pt/.pth` (TorchScript or
state_dict for timm/torchvision architectures — full-model pickles are refused),
`.keras/.h5`, `.joblib` (FFT/DCT+SVM, sha256 mandatory). Compute a checksum with
`sha256sum model.onnx`. A model with no file reports NOT_CONFIGURED; the system keeps
working with the others.
