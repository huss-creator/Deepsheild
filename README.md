# DeepShield — AI-Powered Media Forensics Platform

DeepShield is a Final Year Project (FYP) that helps users verify the authenticity of digital images.
Built with Django, it provides secure user authentication and an image-analysis workflow with a
modern cyber-themed interface.

## Features

- **Secure Authentication** — User registration with email validation and duplicate-email checks,
  login/logout with safe redirect handling, and login-protected pages.
- **Image Analysis** — Upload interface that validates real image content (magic-byte inspection,
  not just file extensions), enforces a 10 MB limit, and records a SHA-256 fingerprint of each upload.
- **Result Dashboard** — Personal dashboard showing account overview and per-upload analysis results.
- **Cyber-Themed UI** — Responsive dark UI with custom CSS/JS (`static/`, `templates/`).

## Tech Stack

- Python 3 + Django
- HTML / CSS / JavaScript (no frontend framework)
- SQLite (development)

## Getting Started

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Apply migrations and run the test suite
python manage.py migrate
python manage.py test

# 4. Start the development server
python manage.py runserver
```

Open http://127.0.0.1:8000/ — register a new account, then log in to reach the dashboard.
The Django admin (`/admin/`) is available after running `python manage.py createsuperuser`.

## Project Structure

| Area | Files |
|---|---|
| Project setup | `manage.py`, `requirements.txt`, `.gitignore`, `config/` (settings, URLs, WSGI/ASGI) |
| Landing page | `templates/base.html`, `templates/home.html`, `detector/views_home.py`, `static/` |
| Registration | `detector/forms.py`, `detector/views_register.py`, `templates/registration/register.html` |
| Dashboard | `detector/views_dashboard.py`, `templates/dashboard.html` |
| Login / logout | `detector/views_login.py`, `templates/registration/login.html` |
| Analysis | `detector/views_analyze.py`, `templates/analyze.html` |
| Results | `detector/views_result.py`, `templates/result.html` |
| Routing & tests | `detector/urls.py`, `detector/tests.py` |

## Team

| Member | GitHub | Contribution |
|---|---|---|
| Hussnain | [@huss-creator](https://github.com/huss-creator) | Project setup, configuration, landing page, static assets |
| Usman | [@m4n1c0des](https://github.com/m4n1c0des) | Registration, dashboard, image-analysis upload flow |
| Bilal | [@mrzbilal](https://github.com/mrzbilal) | Login/logout, URL routing, tests, result pages |

## Notes

- `db.sqlite3`, virtual environments, and user-uploaded media are intentionally not committed
  (see `.gitignore`).
- The analysis pipeline in this slice validates uploads and fingerprints them; model inference
  lives in the full DeepShield project.
