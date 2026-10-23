# Team GitHub guide (3 members) — DeepShield auth module

## Who pushes what
| Member | Part | Files (each member commits only these) |
|---|---|---|
| **Member 1** | Project setup + Landing page | `.gitignore`, `requirements.txt`, `README.md`, `manage.py`, `config/*`, `templates/base.html`, `templates/home.html`, `static/*`, `detector/views_home.py` |
| **Member 2** | Registration + Dashboard (logged-in home page) | `detector/forms.py`, `detector/views_register.py`, `templates/registration/register.html`, `detector/views_dashboard.py`, `templates/dashboard.html` |
| **Member 3** | Login / Logout + routing + tests | `detector/views_login.py`, `templates/registration/login.html`, `detector/urls.py`, `detector/tests.py`, `detector/__init__.py`, `detector/apps.py`, `detector/models.py`, `detector/admin.py`, `detector/migrations/__init__.py` |

Rule: every member commits from **their own computer and their own GitHub account**, and must be able to explain their files.

## 0. One-time setup (each member)
```
git config --global user.name  "Your Real Name"
git config --global user.email "email-used-on-your-GitHub-account@example.com"
```
(The email must be the one added in GitHub -> Settings -> Emails, otherwise commits do not link to your profile.)

## 1. Member 1 creates the repo
1. github.com -> New repository -> name `deepshield-auth` -> Public/Private -> do NOT add README -> Create.
2. Settings -> Collaborators -> Add Member 2 and Member 3 (they accept the e-mail invite).
3. In the project folder:
```
git init
git branch -M main
git remote add origin https://github.com/<member1-username>/deepshield-auth.git
git add .gitignore requirements.txt README.md manage.py
git commit -m "Add project setup: gitignore, requirements, README"
git add config
git commit -m "Add Django settings and root URL configuration"
git add templates/base.html static
git commit -m "Add base layout and static assets (css/js)"
git add templates/home.html detector/views_home.py
git commit -m "Add landing page and home view"
git push -u origin main
```
(First push needs only `main`; the other two members work on branches.)

## 2. Member 2 (Registration + Dashboard)
```
git clone https://github.com/<member1-username>/deepshield-auth.git
cd deepshield-auth
git checkout -b feature/registration
```
Copy your 3 files into the same paths, then:
```
git add detector/forms.py
git commit -m "Add registration form with email and duplicate-email check"
git add detector/views_register.py
git commit -m "Add register view"
git add templates/registration/register.html
git commit -m "Add registration page template"
git add detector/views_dashboard.py
git commit -m "Add login-protected dashboard view"
git add templates/dashboard.html
git commit -m "Add dashboard page (account overview)"
git push -u origin feature/registration
```
GitHub -> "Compare & pull request" -> Create pull request.

## 3. Member 3 (Login / Logout / routing / tests)
```
git clone https://github.com/<member1-username>/deepshield-auth.git
cd deepshield-auth
git checkout -b feature/login
```
Copy your files, then:
```
git add detector/__init__.py detector/apps.py detector/models.py detector/admin.py detector/migrations/__init__.py
git commit -m "Add detector app skeleton"
git add detector/views_login.py
git commit -m "Add login and logout views with safe redirect"
git add templates/registration/login.html
git commit -m "Add login page template"
git add detector/urls.py
git commit -m "Add URL routes for home, register, login, logout, dashboard"
git add detector/tests.py
git commit -m "Add tests for landing, registration, login and logout"
git push -u origin feature/login
```
Open a Pull Request.

## 4. Merge
Member 1 opens each Pull Request -> reads it -> **Merge pull request** (Member 2's first, then Member 3's).
Everyone then updates: `git checkout main` and `git pull`.

## 5. Check the project runs (anyone)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py test
python manage.py runserver
```

## 6. Show who did what (for the teacher)
```
git log --format="%h | %an | %ad | %s" --date=short
git shortlog -sn
git log --author="Member Name" --stat
```
On GitHub: repo -> **Insights -> Contributors**, the **Pull requests** tab, and each file's **Blame / History**.

## Do not commit
`db.sqlite3`, `venv/`, `media/` (already in `.gitignore`). If `git status` lists them, run `git rm --cached <file>`.
