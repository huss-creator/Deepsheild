"""URL routes for the DeepShield web application."""

from django.contrib.auth import views as auth_views
from django.urls import path

from detector import views


urlpatterns = [
    path('', views.home_view, name='home'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.history_view, name='dashboard'),
    path('analyze/', views.upload_view, name='analyze'),
    path('upload/', views.upload_view, name='upload'),
    path('history/', views.history_view, name='history'),
    path('result/<uuid:pk>/', views.result_view, name='result'),
    path('processing/<uuid:pk>/', views.processing_view, name='processing'),
    path('job-status/<uuid:pk>/', views.job_status_view, name='job_status'),
    path('cancel-job/<uuid:pk>/', views.cancel_job_view, name='cancel_job'),
    path('detail/<uuid:pk>/', views.detail_view, name='detail'),
    path('report/<uuid:pk>/', views.report_view, name='report'),
    path('privacy/', views.privacy_view, name='privacy'),
    path('account/', views.account_view, name='account'),
    path('analysis/<uuid:pk>/delete/', views.delete_analysis_view, name='delete_analysis'),
    path('account/delete/', views.delete_account_view, name='delete_account'),
    path('limitations/', views.limitations_view, name='limitations'),
    path(
        'password-reset/',
        views.ThrottledPasswordResetView.as_view(),
        name='password_reset',
    ),
    path(
        'password-reset/done/',
        auth_views.PasswordResetDoneView.as_view(
            template_name='registration/password_reset_done.html',
        ),
        name='password_reset_done',
    ),
    path(
        'password-reset/<uidb64>/<token>/',
        auth_views.PasswordResetConfirmView.as_view(
            template_name='registration/password_reset_confirm.html',
        ),
        name='password_reset_confirm',
    ),
    path(
        'password-reset/complete/',
        auth_views.PasswordResetCompleteView.as_view(
            template_name='registration/password_reset_complete.html',
        ),
        name='password_reset_complete',
    ),
]
