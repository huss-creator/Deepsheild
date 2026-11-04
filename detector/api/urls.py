from django.urls import path
from drf_spectacular.views import SpectacularAPIView, SpectacularRedocView, SpectacularSwaggerView

from detector.api import views as v

urlpatterns = [
    path('auth/register/', v.RegisterView.as_view(), name='api-register'),
    path('auth/login/', v.LoginView.as_view(), name='api-login'),
    path('auth/refresh/', v.RefreshView.as_view(), name='api-refresh'),
    path('auth/logout/', v.LogoutView.as_view(), name='api-logout'),
    path('auth/account/', v.AccountView.as_view(), name='api-account'),
    path('profile/', v.AccountView.as_view(), name='api-profile'),

    path('analyses/', v.AnalysisListCreateView.as_view(), name='api-analyses'),
    path('analyses/<uuid:pk>/', v.AnalysisDetailView.as_view(), name='api-analysis'),
    path('analyze/', v.AnalysisListCreateView.as_view(), name='api-analyze'),
    path('analyze/<uuid:pk>/', v.AnalysisDetailView.as_view(), name='api-analyze-detail'),
    path('analyses/<uuid:pk>/status/', v.AnalysisStatusView.as_view(), name='api-status'),
    path('analyze/<uuid:pk>/status/', v.AnalysisStatusView.as_view(), name='api-analyze-status'),
    path('analyses/<uuid:pk>/cancel/', v.AnalysisCancelView.as_view(), name='api-cancel'),
    path('analyze/<uuid:pk>/cancel/', v.AnalysisCancelView.as_view(), name='api-analyze-cancel'),
    path('analyses/<uuid:pk>/faces/', v.AnalysisFacesView.as_view(), name='api-faces'),
    path('analyses/<uuid:pk>/models/', v.AnalysisModelsView.as_view(), name='api-analysis-models'),
    path('analyses/<uuid:pk>/forensics/', v.AnalysisForensicsView.as_view(), name='api-forensics'),
    path('analyses/<uuid:pk>/report/', v.AnalysisReportView.as_view(), name='api-report'),
    path('analyses/<uuid:pk>/image/', v.AnalysisImageView.as_view(), name='api-image'),
    path('analyses/<uuid:pk>/faces/<int:index>/crop/', v.FaceCropView.as_view(), name='api-face-crop'),
    path('analyses/<uuid:pk>/forensics/<str:test>/image/', v.ForensicImageView.as_view(), name='api-forensic-image'),

    path('models/', v.ModelsView.as_view(), name='api-models'),
    path('health/', v.HealthView.as_view(), name='api-direct-health'),
    path('ready/', v.ReadinessView.as_view(), name='api-direct-ready'),
    path('system/health/', v.HealthView.as_view(), name='api-health'),
    path('system/readiness/', v.ReadinessView.as_view(), name='api-readiness'),

    path('schema/', SpectacularAPIView.as_view(), name='api-schema'),
    path('docs/', SpectacularSwaggerView.as_view(url_name='api-schema'), name='api-docs'),
    path('redoc/', SpectacularRedocView.as_view(url_name='api-schema'), name='api-redoc'),
]
