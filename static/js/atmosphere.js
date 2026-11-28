/**
 * DeepShield — Reusable Cybersecurity Background & Atmosphere Engine
 * Supports configurable presets: 'subtle' (Auth/Login/Register) & 'rich' (Homepage)
 *
 * Capabilities:
 * - Architectural particle network with proximity lines.
 * - Interpolated (Lerp) cursor-responsive ambient lighting.
 * - Automatic pause via Page Visibility API & IntersectionObserver.
 * - Strict accessibility (prefers-reduced-motion renders static frame).
 * - Clamped DPR (1.5) for high performance on Retina screens.
 * - Touch-aware (disables pointer tracking on mobile/touch screens).
 */

(function () {
    'use strict';

    var PRESETS = {
        subtle: {
            // Tailored for Login, Register, Recovery (Zero distraction)
            desktopCount: 22,
            tabletCount: 14,
            mobileCount: 8,
            maxConnectionDist: 75,
            lineAlpha: 0.06,
            glowAlpha: 0.035,
            glowRadiusFactor: 0.38,
            maxSpeed: 0.18,
            particleAlphaBase: 0.16,
            particleAlphaRange: 0.12
        },
        rich: {
            // Tailored for Homepage Hero (Immersive showcase)
            desktopCount: 40,
            tabletCount: 24,
            mobileCount: 14,
            maxConnectionDist: 95,
            lineAlpha: 0.085,
            glowAlpha: 0.052,
            glowRadiusFactor: 0.45,
            maxSpeed: 0.24,
            particleAlphaBase: 0.20,
            particleAlphaRange: 0.16
        }
    };

    function setupAtmosphere(canvas) {
        if (!canvas || !canvas.getContext) return;

        var container = canvas.closest('.ds-atmosphere-container, .ds-auth-container, .ds-home-hero') || canvas.parentElement;
        var ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Determine preset
        var presetName = canvas.getAttribute('data-preset') ||
                         (canvas.id === 'dsHomeAtmosphere' ? 'rich' : 'subtle');
        var config = PRESETS[presetName] || PRESETS.subtle;

        // Accessibility & device checks
        var motionMediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
        var isReducedMotion = motionMediaQuery.matches;

        var touchMediaQuery = window.matchMedia('(hover: none), (pointer: coarse)');
        var isTouchDevice = touchMediaQuery.matches;

        var width = 0;
        var height = 0;
        var dpr = 1;
        var particles = [];
        var rafId = null;
        var isRunning = false;

        // Mouse lighting coordinates (Lerp)
        var targetMouseX = 0;
        var targetMouseY = 0;
        var currentMouseX = 0;
        var currentMouseY = 0;
        var isPointerInside = false;

        function getActiveCount(w) {
            if (w < 600) return config.mobileCount;
            if (w < 992) return config.tabletCount;
            return config.desktopCount;
        }

        function getConnectionThreshold(w) {
            return w < 768 ? Math.round(config.maxConnectionDist * 0.75) : config.maxConnectionDist;
        }

        // Particle Class
        function Particle() {
            this.x = Math.random() * (width || 800);
            this.y = Math.random() * (height || 600);

            var angle = Math.random() * Math.PI * 2;
            var speed = (0.08 + Math.random() * 0.12) * (config.maxSpeed / 0.18);
            this.vx = Math.cos(angle) * speed;
            this.vy = Math.sin(angle) * speed;

            this.radius = 1.0 + Math.random() * 0.6;
            this.isCyan = Math.random() > 0.82;
            this.color = this.isCyan ? '0, 229, 255' : '0, 255, 157';

            this.baseAlpha = config.particleAlphaBase + Math.random() * config.particleAlphaRange;
            this.pulsePhase = Math.random() * Math.PI * 2;
            this.pulseSpeed = 0.015 + Math.random() * 0.018;
        }

        Particle.prototype.update = function () {
            if (isReducedMotion) return;

            this.pulsePhase += this.pulseSpeed;
            this.x += this.vx;
            this.y += this.vy;

            var pad = 12;
            if (this.x < -pad) this.x = width + pad;
            else if (this.x > width + pad) this.x = -pad;
            if (this.y < -pad) this.y = height + pad;
            else if (this.y > height + pad) this.y = -pad;

            // Gentle drift towards smoothed pointer light
            if (!isTouchDevice && isPointerInside) {
                var dx = currentMouseX - this.x;
                var dy = currentMouseY - this.y;
                var dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 130 && dist > 12) {
                    var factor = (1 - dist / 130) * 0.012;
                    this.vx += (dx / dist) * factor;
                    this.vy += (dy / dist) * factor;

                    var curSpeed = Math.sqrt(this.vx * this.vx + this.vy * this.vy);
                    if (curSpeed > config.maxSpeed * 1.5) {
                        this.vx = (this.vx / curSpeed) * (config.maxSpeed * 1.5);
                        this.vy = (this.vy / curSpeed) * (config.maxSpeed * 1.5);
                    }
                }
            }
        };

        Particle.prototype.draw = function () {
            var alpha = this.baseAlpha + Math.sin(this.pulsePhase) * 0.06;
            if (alpha < 0.05) alpha = 0.05;
            if (alpha > 0.38) alpha = 0.38;

            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(' + this.color + ', ' + alpha + ')';
            ctx.fill();
        };

        function initParticles() {
            var count = getActiveCount(width);
            particles = [];
            for (var i = 0; i < count; i++) {
                particles.push(new Particle());
            }
        }

        function resize() {
            var rect = container.getBoundingClientRect();
            width = Math.max(rect.width, 320);
            height = Math.max(rect.height, 350);

            // DPR cap at 1.5 for performance
            dpr = Math.min(window.devicePixelRatio || 1, 1.5);
            canvas.width = Math.floor(width * dpr);
            canvas.height = Math.floor(height * dpr);

            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

            // Default resting mouse position
            if (!isPointerInside) {
                targetMouseX = width * 0.68;
                targetMouseY = height * 0.42;
                currentMouseX = targetMouseX;
                currentMouseY = targetMouseY;
            }

            initParticles();

            if (isReducedMotion) {
                renderFrame();
            }
        }

        function drawConnections() {
            var maxDist = getConnectionThreshold(width);
            var maxDistSq = maxDist * maxDist;
            var len = particles.length;

            ctx.lineWidth = 0.75;

            for (var i = 0; i < len; i++) {
                var p1 = particles[i];
                for (var j = i + 1; j < len; j++) {
                    var p2 = particles[j];
                    var dx = p1.x - p2.x;
                    var dy = p1.y - p2.y;
                    var distSq = dx * dx + dy * dy;

                    if (distSq < maxDistSq) {
                        var dist = Math.sqrt(distSq);
                        var alpha = (1 - dist / maxDist) * config.lineAlpha;

                        ctx.strokeStyle = p1.isCyan || p2.isCyan
                            ? 'rgba(0, 229, 255, ' + alpha + ')'
                            : 'rgba(0, 255, 157, ' + alpha + ')';

                        ctx.beginPath();
                        ctx.moveTo(p1.x, p1.y);
                        ctx.lineTo(p2.x, p2.y);
                        ctx.stroke();
                    }
                }
            }
        }

        function drawAmbientMouseLight() {
            currentMouseX += (targetMouseX - currentMouseX) * 0.045;
            currentMouseY += (targetMouseY - currentMouseY) * 0.045;

            var glowRadius = Math.min(420, width * config.glowRadiusFactor);
            var grad = ctx.createRadialGradient(
                currentMouseX, currentMouseY, 0,
                currentMouseX, currentMouseY, glowRadius
            );

            grad.addColorStop(0, 'rgba(0, 255, 157, ' + config.glowAlpha + ')');
            grad.addColorStop(0.45, 'rgba(0, 229, 255, ' + (config.glowAlpha * 0.45) + ')');
            grad.addColorStop(1, 'rgba(0, 0, 0, 0)');

            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.arc(currentMouseX, currentMouseY, glowRadius, 0, Math.PI * 2);
            ctx.fill();
        }

        function renderFrame() {
            ctx.clearRect(0, 0, width, height);
            drawAmbientMouseLight();
            drawConnections();
            for (var i = 0; i < particles.length; i++) {
                particles[i].draw();
            }
        }

        function loop() {
            if (!isRunning) return;
            for (var i = 0; i < particles.length; i++) {
                particles[i].update();
            }
            renderFrame();
            rafId = requestAnimationFrame(loop);
        }

        function startLoop() {
            if (isReducedMotion) {
                renderFrame();
                return;
            }
            if (!isRunning) {
                isRunning = true;
                rafId = requestAnimationFrame(loop);
            }
        }

        function stopLoop() {
            isRunning = false;
            if (rafId) {
                cancelAnimationFrame(rafId);
                rafId = null;
            }
        }

        // Pointer event listeners
        if (!isTouchDevice) {
            container.addEventListener('pointermove', function (e) {
                var rect = container.getBoundingClientRect();
                targetMouseX = e.clientX - rect.left;
                targetMouseY = e.clientY - rect.top;
                isPointerInside = true;
            }, { passive: true });

            container.addEventListener('pointerleave', function () {
                isPointerInside = false;
                targetMouseX = width * 0.68;
                targetMouseY = height * 0.42;
            }, { passive: true });
        }

        // Visibility & Intersection management
        document.addEventListener('visibilitychange', function () {
            if (document.hidden) stopLoop();
            else startLoop();
        });

        if ('IntersectionObserver' in window) {
            var observer = new IntersectionObserver(function (entries) {
                entries.forEach(function (entry) {
                    if (entry.isIntersecting) startLoop();
                    else stopLoop();
                });
            }, { threshold: 0.05 });
            observer.observe(container);
        } else {
            startLoop();
        }

        // Debounced resize
        var resizeTimeout;
        window.addEventListener('resize', function () {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(resize, 100);
        }, { passive: true });

        // Reduced motion listener
        motionMediaQuery.addEventListener('change', function (e) {
            isReducedMotion = e.matches;
            if (isReducedMotion) {
                stopLoop();
                renderFrame();
            } else {
                startLoop();
            }
        });

        resize();
        startLoop();
    }

    // Auto-discover all atmosphere canvases
    function initAll() {
        var canvases = document.querySelectorAll(
            'canvas.ds-atmosphere-canvas, #dsAuthAtmosphere, #dsHomeAtmosphere'
        );
        for (var i = 0; i < canvases.length; i++) {
            setupAtmosphere(canvases[i]);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initAll);
    } else {
        initAll();
    }

    // Expose for programmatic initialization if needed
    window.DS = window.DS || {};
    window.DS.initAtmosphere = setupAtmosphere;
})();

