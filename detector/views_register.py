"""Registration page (Member 2)."""

from django.contrib import messages
from django.shortcuts import redirect, render

from detector.forms import UserRegistrationForm


def register_view(request):
    # Already logged in? No need to register again.
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()          # Django hashes the password and saves the user
            messages.success(request, f"Account created for {user.username}! You can now log in.")
            return redirect('login')
    else:
        form = UserRegistrationForm()

    return render(request, 'registration/register.html', {'form': form})
