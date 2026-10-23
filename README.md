# DeepShield — Auth Module (Demo Slice)

This is just the login/registration part of the full DeepShield FYP,
pulled out into its own runnable project — same folder layout, same
UI/theme, same Django auth logic. No image upload, AI models, or
forensics here; that lives in the main project.

## Run it

```
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt

python manage.py migrate
python manage.py runserver
```

Then open http://127.0.0.1:8000/ — Register a new account, then log in
with it (you land on the Dashboard). `/admin/` also works if you run `createsuperuser` first.

## What's inside

| Part | Files |
|---|---|
| Project setup | `manage.py`, `requirements.txt`, `.gitignore`, `config/` (settings, urls, wsgi, asgi) |
| Landing / home page | `templates/base.html`, `templates/home.html`, `detector/views_home.py`, `static/` (css + js) |
| Registration | `detector/forms.py`, `detector/views_register.py`, `templates/registration/register.html` |
| Dashboard (logged-in home page) | `detector/views_dashboard.py`, `templates/dashboard.html` |
| Login / logout | `detector/views_login.py`, `templates/registration/login.html` |
| Routing + tests | `detector/urls.py`, `detector/tests.py` |

Run the tests: `python manage.py test`
