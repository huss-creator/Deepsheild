"""Logged-in home page / dashboard (Member 2)."""

from django.contrib.auth.decorators import login_required
from django.shortcuts import render


@login_required            # guests are sent to /login/?next=/dashboard/
def dashboard_view(request):
    return render(request, 'dashboard.html')
