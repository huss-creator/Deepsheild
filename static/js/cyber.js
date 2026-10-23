// Binary-code face — the animated hero visual on the landing page.
// Idea: scatter a bunch of points inside a face-shaped mask, draw a random
// 0/1 at each point, and run a soft brightness "scan" down the canvas so it
// reads as a face built out of code, not just random digit noise.
// Everything is drawn on a single 2D canvas, no external assets.

(function () {
    const canvas = document.getElementById('dsBinaryFaceCanvas');
    if (!canvas || !canvas.getContext) return;

    const ctx = canvas.getContext('2d');
    let w, h, dpr;
    let points = [];
    let scanY = 0;

    function resize() {
        dpr = window.devicePixelRatio || 1;
        const rect = canvas.parentElement.getBoundingClientRect();
        w = rect.width;
        h = rect.height;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        buildPoints();
    }

    // Is (x, y) inside a rough face silhouette? Normalized 0-1 coords.
    // Cheap approximation: an ellipse for the head + a narrower ellipse
    // chopped off for the jaw, good enough to read as "face" at this scale.
    function inFace(nx, ny) {
        const dx = (nx - 0.5) / 0.34;
        const dy = (ny - 0.46) / 0.42;
        const inHead = dx * dx + dy * dy <= 1;
        if (!inHead) return false;
        // taper the bottom third into a jaw instead of a flat oval
        if (ny > 0.72) {
            const jawX = (nx - 0.5) / (0.34 * (1 - (ny - 0.72) * 1.6));
            return Math.abs(jawX) < 1;
        }
        return true;
    }

    function buildPoints() {
        points = [];
        const spacing = Math.max(10, Math.floor(w / 26));
        for (let y = 0; y < h; y += spacing) {
            for (let x = 0; x < w; x += spacing) {
                const nx = x / w, ny = y / h;
                if (inFace(nx, ny)) {
                    points.push({
                        x: x + (Math.random() - 0.5) * 4,
                        y: y + (Math.random() - 0.5) * 4,
                        val: Math.random() < 0.5 ? '0' : '1',
                        flipEvery: 40 + Math.floor(Math.random() * 120),
                        frame: Math.floor(Math.random() * 100),
                    });
                }
            }
        }
        // a handful of "feature" points (eyes, nose, mouth) drawn brighter
        // so the shape reads as a face and not just a filled blob
        const features = [
            [0.38, 0.42], [0.62, 0.42], // eyes
            [0.5, 0.55],                 // nose
            [0.42, 0.68], [0.5, 0.7], [0.58, 0.68], // mouth
        ];
        features.forEach(([nx, ny]) => {
            points.push({
                x: nx * w, y: ny * h,
                val: Math.random() < 0.5 ? '0' : '1',
                flipEvery: 30, frame: 0, feature: true,
            });
        });
    }

    function draw() {
        ctx.clearRect(0, 0, w, h);
        ctx.font = '11px "Share Tech Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        scanY = (scanY + 1.4) % (h + 80);

        for (const p of points) {
            p.frame++;
            if (p.frame % p.flipEvery === 0) {
                p.val = p.val === '0' ? '1' : '0';
            }
            const distFromScan = Math.abs(p.y - scanY);
            const boost = Math.max(0, 1 - distFromScan / 70);
            const base = p.feature ? 0.9 : 0.35;
            const alpha = Math.min(1, base + boost * 0.6);

            ctx.fillStyle = p.feature
                ? `rgba(0, 229, 255, ${alpha})`
                : `rgba(0, 255, 157, ${alpha})`;
            if (boost > 0.05) {
                ctx.shadowColor = 'rgba(0, 255, 157, 0.8)';
                ctx.shadowBlur = 6 * boost;
            } else {
                ctx.shadowBlur = 0;
            }
            ctx.fillText(p.val, p.x, p.y);
        }

        requestAnimationFrame(draw);
    }

    window.addEventListener('resize', resize);
    resize();
    requestAnimationFrame(draw);
})();
