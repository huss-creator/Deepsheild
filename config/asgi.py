"""
ASGI config for deepshield_project project.
You do not need to change this file for basic development.
"""

import os
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
application = get_asgi_application()
