"""detector/context_processors.py: small values every template can use."""

from django.conf import settings


def site_context(request):
    return {
        'app_version': getattr(settings, 'DEEPSHIELD_VERSION', '2.0.0'),
        'system_name': 'DeepShield',
    }
