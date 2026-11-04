/* Theme: saved choice (localStorage) -> else system preference. Loaded in <head> to avoid a flash. */
(function () {
    var root = document.documentElement, KEY = 'ds-theme';
    function saved() { try { return localStorage.getItem(KEY); } catch (e) { return null; } }
    function systemLight() { return window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches; }
    function apply(t) {
        root.setAttribute('data-theme', t);
        root.setAttribute('data-bs-theme', t);
        if (document.body) {
            document.body.setAttribute('data-theme', t);
            document.body.setAttribute('data-bs-theme', t);
        }
    }
    var locked = root.getAttribute('data-lock-theme');            // public pages: dark only
    function current() { return locked || saved() || (systemLight() ? 'light' : 'dark'); }
    apply(current());
    document.addEventListener('DOMContentLoaded', function () {
        apply(current());
        var btn = document.getElementById('themeToggle');
        if (!btn) return;
        function label() {
            var l = root.getAttribute('data-theme') === 'light';
            btn.setAttribute('aria-label', l ? 'Switch to dark theme' : 'Switch to light theme');
            if (btn.firstElementChild) {
                btn.firstElementChild.className = l ? 'bi bi-moon-stars' : 'bi bi-sun';
            }
        }
        label();
        btn.addEventListener('click', function () {
            var next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
            apply(next);
            try { localStorage.setItem(KEY, next); } catch (e) {}
            label();
        });
    });
})();

