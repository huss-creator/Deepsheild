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

# Auto-switch to project virtual environment (.venv) if invoked with system Python
_project_root = os.path.dirname(os.path.abspath(__file__))
_venv_python = os.path.join(_project_root, '.venv', 'Scripts', 'python.exe')
if (
    os.path.isfile(_venv_python)
    and os.path.abspath(sys.executable).lower() != os.path.abspath(_venv_python).lower()
    and os.environ.get('_DEEPSHIELD_VENV_EXEC') != '1'
):
    import subprocess
    os.environ['_DEEPSHIELD_VENV_EXEC'] = '1'
    sys.exit(subprocess.call([_venv_python] + sys.argv))

_dll_directory_handle = None
if os.name == 'nt':
    dll_directory = os.path.join(sys.prefix, 'Scripts')
    if os.path.isdir(dll_directory):
        try:
            _dll_directory_handle = os.add_dll_directory(dll_directory)
        except Exception:
            pass


def main():
    # Tell Django where our settings file is
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.development')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Make sure it is installed and that you have "
            "activated your virtual environment."
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()

