/*
 * static/js/main.js — DeepShield Global JavaScript (Upgraded)
 *
 * What this file does:
 *   1. Auto-dismiss flash alerts after 5 seconds
 *   2. Activate Bootstrap tooltips on every page
 *   3. Toast notification system  (call: DS.toast('message', 'success'))
 *   4. Scroll-reveal animation for cards
 *   5. Animated number counters for stat cards
 *   6. Active nav link highlighting fallback
 *
 * Kept simple — no frameworks, no build tools, pure vanilla JS.
 * Comments explain every section for student readability.
 */

// Wrap everything in DOMContentLoaded so the HTML is fully loaded first
document.addEventListener('DOMContentLoaded', function () {


    // ===========================================================
    // 1. AUTO-DISMISS FLASH ALERTS after 5 seconds
    // ===========================================================
    // Django's messages framework shows alerts at the top of the page.
    // We close them automatically so they don't block the view.
    var alerts = document.querySelectorAll('.ds-alert.alert');
    alerts.forEach(function (alert) {
        setTimeout(function () {
            var bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
            if (bsAlert) bsAlert.close();
        }, 5000);
    });


    // ===========================================================
    // 2. BOOTSTRAP TOOLTIPS
    // ===========================================================
    // Any element with data-bs-toggle="tooltip" gets a tooltip on hover.
    var tooltipEls = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipEls.forEach(function (el) {
        new bootstrap.Tooltip(el, { trigger: 'hover' });
    });


    // ===========================================================
    // 3. ACTIVE NAV LINK FALLBACK
    // ===========================================================
    // Django templates already add class="active" based on URL name,
    // but this provides a client-side fallback for any edge cases.
    var currentPath = window.location.pathname;
    var navLinks = document.querySelectorAll('.ds-navbar .nav-link');
    navLinks.forEach(function (link) {
        var href = link.getAttribute('href');
        if (href && href !== '/' && currentPath.startsWith(href)) {
            link.classList.add('active');
        } else if (href === '/' && currentPath === '/') {
            link.classList.add('active');
        }
    });


    // ===========================================================
    // 4. SCROLL-REVEAL ANIMATION
    // ===========================================================
    // Cards with class "ds-reveal" fade in as they scroll into view.
    // This uses IntersectionObserver (modern, no library needed).
    var revealEls = document.querySelectorAll('.ds-reveal');
    if (revealEls.length > 0 && 'IntersectionObserver' in window) {
        var revealObserver = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    // Once revealed, stop observing to save memory
                    revealObserver.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.12,        // Trigger when 12% of element is visible
            rootMargin: '0px 0px -40px 0px'  // Trigger slightly before it's fully on screen
        });

        revealEls.forEach(function (el) {
            revealObserver.observe(el);
        });
    } else {
        // Fallback: just make everything visible without animation
        revealEls.forEach(function (el) {
            el.classList.add('visible');
        });
    }


    // ===========================================================
    // 5. ANIMATED NUMBER COUNTERS
    // ===========================================================
    // Elements with class "ds-stat-number" and a data-target attribute
    // will count up from 0 to their target value on load.
    //
    // Usage in template: <div class="ds-stat-number" data-target="42">42</div>
    var counters = document.querySelectorAll('.ds-stat-number[data-target]');
    counters.forEach(function (el) {
        var target = parseInt(el.getAttribute('data-target'), 10);
        if (isNaN(target) || target === 0) return;

        var duration = 1200;   // Total animation duration in ms
        var start    = 0;
        var startTs  = null;

        function step(timestamp) {
            if (!startTs) startTs = timestamp;
            var progress = Math.min((timestamp - startTs) / duration, 1);
            // Ease-out: fast start, slow finish
            var eased = 1 - Math.pow(1 - progress, 3);
            el.textContent = Math.round(eased * target);
            if (progress < 1) requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
    });


    // ===========================================================
    // 6. COPY TO CLIPBOARD — SHA-256 hash button
    // ===========================================================
    // Any button with data-copy-target="#elementId" copies that element's text.
    // Works for the SHA-256 copy button on the result page.
    var copyBtns = document.querySelectorAll('[data-copy-target]');
    copyBtns.forEach(function (btn) {
        btn.addEventListener('click', function () {
            var targetId = btn.getAttribute('data-copy-target');
            var targetEl = document.querySelector(targetId);
            if (!targetEl) return;

            var text = targetEl.textContent.trim();

            navigator.clipboard.writeText(text).then(function () {
                // Show success state on button
                var origHTML = btn.innerHTML;
                btn.innerHTML = '<i class="bi bi-check-lg me-1"></i>Copied!';
                btn.classList.add('copied');

                // Show a toast notification
                DS.toast('SHA-256 hash copied to clipboard', 'success');

                // Reset button after 2 seconds
                setTimeout(function () {
                    btn.innerHTML = origHTML;
                    btn.classList.remove('copied');
                }, 2000);
            }).catch(function () {
                DS.toast('Could not copy — please copy manually', 'danger');
            });
        });
    });


    // ===========================================================
    // 7. MODEL CARD STAGGERED REVEAL
    // ===========================================================
    // On the result page, model cards appear one by one for a polished feel.
    var modelCards = document.querySelectorAll('.ds-model-card');
    modelCards.forEach(function (card, index) {
        card.style.animationDelay = (index * 0.08) + 's';
        card.classList.add('card-reveal');
    });


    // ===========================================================
    // 8. PROGRESS BAR ANIMATION
    // ===========================================================
    // Confidence progress bars animate to their target width on page load.
    // The bar must have style="width: X%" set in the template.
    var progressBars = document.querySelectorAll('.ds-progress-animated .progress-bar');
    progressBars.forEach(function (bar) {
        var targetWidth = bar.style.width;
        bar.style.width = '0%';
        setTimeout(function () {
            bar.style.width = targetWidth;
        }, 300);
    });


}); // End of DOMContentLoaded


// ===========================================================
// DS — Global DeepShield utilities (accessible anywhere)
// ===========================================================
var DS = (function () {

    // -------------------------------------------------------
    // TOAST NOTIFICATION SYSTEM
    // -------------------------------------------------------
    // Creates a floating toast notification.
    //
    // Usage:
    //   DS.toast('Analysis complete', 'success');
    //   DS.toast('Upload failed', 'danger');
    //   DS.toast('No face found', 'warning');
    //   DS.toast('Processing...', 'info');
    //
    // Types: 'success' | 'danger' | 'warning' | 'info'

    var TOAST_ICONS = {
        success: 'bi-check-circle-fill',
        danger:  'bi-exclamation-triangle-fill',
        warning: 'bi-exclamation-circle-fill',
        info:    'bi-info-circle-fill',
    };

    function toast(message, type) {
        type = type || 'info';

        // Create or find the toast container
        var container = document.getElementById('ds-toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'ds-toast-container';
            document.body.appendChild(container);
        }

        // Build the toast element
        var toast = document.createElement('div');
        toast.className = 'ds-toast ds-toast-' + type;
        toast.setAttribute('role', 'alert');
        toast.innerHTML =
            '<i class="bi ' + (TOAST_ICONS[type] || TOAST_ICONS.info) + ' ds-toast-icon"></i>' +
            '<span class="ds-toast-body">' + escapeHtml(message) + '</span>' +
            '<button class="ds-toast-close" aria-label="Close">' +
                '<i class="bi bi-x-lg"></i>' +
            '</button>';

        container.appendChild(toast);

        // Close button handler
        toast.querySelector('.ds-toast-close').addEventListener('click', function () {
            dismissToast(toast);
        });

        // Auto-dismiss after 4 seconds
        var timer = setTimeout(function () {
            dismissToast(toast);
        }, 4000);

        // If user hovers, pause timer
        toast.addEventListener('mouseenter', function () {
            clearTimeout(timer);
        });
        toast.addEventListener('mouseleave', function () {
            timer = setTimeout(function () {
                dismissToast(toast);
            }, 2000);
        });
    }

    function dismissToast(toast) {
        if (!toast || !toast.parentNode) return;
        toast.classList.add('toast-hiding');
        setTimeout(function () {
            if (toast.parentNode) toast.parentNode.removeChild(toast);
        }, 300);
    }

    // Prevent XSS: escape HTML special characters
    function escapeHtml(str) {
        var div = document.createElement('div');
        div.appendChild(document.createTextNode(String(str)));
        return div.innerHTML;
    }

    // -------------------------------------------------------
    // PASSWORD EYE TOGGLE
    // -------------------------------------------------------
    // Toggles an input between type="password" and type="text".
    // Call after DOM is ready, passing the input ID and button element.
    function initEyeToggle(inputId, buttonEl) {
        var input = document.getElementById(inputId);
        if (!input || !buttonEl) return;

        buttonEl.addEventListener('click', function (e) {
            e.preventDefault();
            var isPassword = input.type === 'password';
            input.type = isPassword ? 'text' : 'password';
            // Toggle the icon
            var icon = buttonEl.querySelector('i');
            if (icon) {
                icon.className = isPassword ? 'bi bi-eye-slash' : 'bi bi-eye';
            }
        });
    }

    // -------------------------------------------------------
    // PASSWORD STRENGTH CHECKER
    // -------------------------------------------------------
    // Call this once to set up live strength feedback on a password field.
    // strengthBarId = the div.ds-strength-fill element
    // labelId = the span showing "Weak / Fair / Good / Strong"
    // reqsId = the ul.ds-pw-reqs element with data-req attributes
    function initStrengthChecker(passwordInputId, strengthBarId, labelId, reqsId) {
        var input    = document.getElementById(passwordInputId);
        var bar      = document.getElementById(strengthBarId);
        var label    = document.getElementById(labelId);
        var reqsList = document.getElementById(reqsId);

        if (!input) return;

        input.addEventListener('input', function () {
            var val = input.value;
            var score = 0;

            var checks = {
                length:    val.length >= 8,
                uppercase: /[A-Z]/.test(val),
                lowercase: /[a-z]/.test(val),
                number:    /[0-9]/.test(val),
                special:   /[^A-Za-z0-9]/.test(val),
            };

            // Each requirement met adds 1 to score (max 5)
            Object.values(checks).forEach(function (met) {
                if (met) score++;
            });

            // Update requirement checkmarks
            if (reqsList) {
                var reqItems = reqsList.querySelectorAll('[data-req]');
                reqItems.forEach(function (li) {
                    var req = li.getAttribute('data-req');
                    var met = checks[req];
                    li.classList.toggle('met', met);
                    var icon = li.querySelector('i');
                    if (icon) {
                        icon.className = met ? 'bi bi-check-circle-fill' : 'bi bi-circle';
                    }
                });
            }

            // Update strength bar and label
            var levels = [
                { min: 0, cls: '',               text: '',       labelCls: '' },
                { min: 1, cls: 'ds-strength-weak',   text: 'Weak',   labelCls: 'weak'   },
                { min: 2, cls: 'ds-strength-fair',   text: 'Fair',   labelCls: 'fair'   },
                { min: 3, cls: 'ds-strength-good',   text: 'Good',   labelCls: 'good'   },
                { min: 5, cls: 'ds-strength-strong', text: 'Strong', labelCls: 'strong' },
            ];

            var level = levels[0];
            levels.forEach(function (l) {
                if (score >= l.min) level = l;
            });

            if (bar) {
                bar.className = 'ds-strength-fill ' + level.cls;
            }
            if (label) {
                label.textContent = level.text;
                label.className = 'ds-strength-label ' + level.labelCls;
            }
        });
    }

    // -------------------------------------------------------
    // PUBLIC API
    // -------------------------------------------------------
    return {
        toast:               toast,
        initEyeToggle:       initEyeToggle,
        initStrengthChecker: initStrengthChecker,
    };

})();
