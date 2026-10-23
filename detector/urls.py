"""URL routes for the auth module (Member 3 wires the three pages together)."""

from django.urls import path

from detector.views_analyze import analyze_view
from detector.views_dashboard import dashboard_view
from detector.views_home import home_view
from detector.views_login import login_view, logout_view
from detector.views_register import register_view
from detector.views_result import result_view

urlpatterns = [
    path('', home_view, name='home'),
    path('register/', register_view, name='register'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('dashboard/', dashboard_view, name='dashboard'),
    path('analyze/', analyze_view, name='analyze'),
    path('result/', result_view, name='result'),
]
