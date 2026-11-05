"""
detector/urls.py: the two halves of the site.

  /public/...  pages anyone can open (landing page, login, register, privacy, limitations)
  /auth/...    pages that need a signed-in user (upload, processing, result, history, account)

config/urls.py mounts `public_patterns` under /public/ and `auth_patterns` under /auth/.
The url *names* (home, login, upload, result ...) are what templates use with {% url 'name' %}.
"""

from django.contrib.auth import views as auth_views
from django.urls import path

from detector import views

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
]
