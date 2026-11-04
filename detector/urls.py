<<<<<<< HEAD
"""
detector/urls.py: the two halves of the site.

  /public/...  pages anyone can open (landing page, login, register, privacy, limitations)
  /auth/...    pages that need a signed-in user (upload, processing, result, history, account)

config/urls.py mounts `public_patterns` under /public/ and `auth_patterns` under /auth/.
The url *names* (home, login, upload, result ...) are what templates use with {% url 'name' %}.
"""

=======
"""URL routes for the DeepShield web application."""

>>>>>>> d4716fba9e4bbff29280890f5b5d6911349c9a3c
from django.contrib.auth import views as auth_views
from django.urls import path

from detector import views
<<<<<<< HEAD

public_patterns = [
    path('', views.home_view, name='home'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('password-reset/', views.ThrottledPasswordResetView.as_view(), name='password_reset'),
    path('password-reset/done/', auth_views.PasswordResetDoneView.as_view(
        template_name='registration/password_reset_done.html'), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(
        template_name='registration/password_reset_confirm.html', success_url='/public/reset/done/'),
        name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(
        template_name='registration/password_reset_complete.html'), name='password_reset_complete'),
    path('privacy/', views.privacy_view, name='privacy'),
    path('limitations/', views.limitations_view, name='limitations'),
]

auth_patterns = [
    path('logout/', views.logout_view, name='logout'),
    path('upload/', views.upload_view, name='upload'),
    path('processing/<int:pk>/', views.processing_view, name='processing'),
    path('processing/<int:pk>/status/', views.job_status_view, name='job_status'),
    path('processing/<int:pk>/cancel/', views.cancel_job_view, name='cancel_job'),
    path('result/<int:pk>/', views.result_view, name='result'),
    path('result/<int:pk>/delete/', views.delete_analysis_view, name='delete_analysis'),
    path('history/', views.history_view, name='history'),
    path('detail/<int:pk>/', views.detail_view, name='detail'),
    path('report/<int:pk>/', views.report_view, name='report'),
    path('account/', views.account_view, name='account'),
    path('account/delete/', views.delete_account_view, name='delete_account'),
=======


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
>>>>>>> d4716fba9e4bbff29280890f5b5d6911349c9a3c
]
