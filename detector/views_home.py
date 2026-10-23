"""Landing / home page (Member 1)."""

from django.shortcuts import render


def home_view(request):
    # Public page: shows Register/Login buttons to guests and a Logout button to logged-in users.
    return render(request, 'home.html')
