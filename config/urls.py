"""
config/urls.py: root URL routing.

  /              sends visitors to /public/ and signed-in users to /auth/upload/
  /public/...    landing page, login, register, privacy, limitations   (anyone)
  /auth/...      upload, processing, result, history, account          (signed-in users)
  /api/...       REST API
  /media/...     private user images (ownership checked)
  /admin/        Django admin
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.urls import re_path
from detector.media import protected_media
from detector.metrics import metrics_view

from detector.api import views as api_views
from detector import views as web_views
from detector.urls import auth_patterns, public_patterns

urlpatterns = [
    path('admin/', admin.site.urls),
    path('metrics', metrics_view, name='metrics'),
    path('health/', api_views.HealthView.as_view(), name='health-check'),
    path('ready/', api_views.ReadinessView.as_view(), name='readiness-check'),
    # Direct /api/ aliases alongside versioned /api/v1/
    path('api/analyze/', api_views.AnalysisListCreateView.as_view(), name='direct-api-analyze'),
    path('api/analyze/<uuid:pk>/', api_views.AnalysisDetailView.as_view(), name='direct-api-analyze-detail'),
    path('api/analyze/<uuid:pk>/status/', api_views.AnalysisStatusView.as_view(), name='direct-api-analyze-status'),
    path('api/analyses/', api_views.AnalysisListCreateView.as_view(), name='direct-api-analyses'),
    path('api/analyses/<uuid:pk>/', api_views.AnalysisDetailView.as_view(), name='direct-api-analysis-detail'),
    path('api/profile/', api_views.AccountView.as_view(), name='direct-api-profile'),
    path('api/auth/register/', api_views.RegisterView.as_view(), name='direct-api-register'),
    path('api/auth/login/', api_views.LoginView.as_view(), name='direct-api-login'),
    path('api/auth/logout/', api_views.LogoutView.as_view(), name='direct-api-logout'),
    # Private media: ownership-checked, never a public static mapping.
    re_path(r'^media/(?P<path>.+)$', protected_media, name='media'),
    path('api/<str:version>/', include('detector.api.urls')),
    path('public/', include(public_patterns)),
    path('auth/', include(auth_patterns)),
    path('', web_views.root_view, name='root'),
]

handler404 = 'detector.api.errors.not_found'
handler500 = 'detector.api.errors.server_error'

# Static assets (CSS/JS) only; in production Nginx serves collected STATIC_ROOT.
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
