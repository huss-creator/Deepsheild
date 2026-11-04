"""App configuration for the 'detector' Django app."""

from django.apps import AppConfig


class DetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detector'
    verbose_name = 'DeepShield Detector'

    def ready(self):
        # Global decompression-bomb guard for every Pillow decode in the project
        # (forensic modules included). Uploads are also limited in image_validation.
        from django.conf import settings
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = settings.PIL_MAX_IMAGE_PIXELS
        from detector import signals  # noqa: F401  (registers file-cleanup handlers)
