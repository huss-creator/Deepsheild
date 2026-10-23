#!/usr/bin/env python
"""
Django's command-line utility for administrative tasks.
Run this file with:  python manage.py <command>
Common commands:
  python manage.py runserver       — starts the development web server
  python manage.py makemigrations  — detects database model changes
  python manage.py migrate         — applies database changes
  python manage.py createsuperuser — creates an admin account
"""

import os
import sys


def main():
    # Tell Django where our settings file is
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Make sure it is installed and that you have "
            "activated your virtual environment."
        ) from exc
    # Auto-apply migrations before starting the server, so a fresh clone never
    # fails with "no such table: auth_user" (RUN_MAIN avoids running twice
    # when the auto-reloader restarts the process).
    if 'runserver' in sys.argv and os.environ.get('RUN_MAIN') != 'true':
        import django
        from django.core.management import call_command
        django.setup()
        call_command('migrate', interactive=False, verbosity=1)
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
