"""
config/urls.py — root routing table.

  /                  -> home page
  /register/         -> registration
  /login/             -> login
  /logout/           -> logout
  /admin/            -> Django admin panel

(This is the auth-only slice of the full project — no
upload/dashboard/history routes here, those live in the main app.)
"""

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('detector.urls')),
]
