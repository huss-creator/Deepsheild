/**
 * Backwards compatibility proxy for auth_atmosphere.js -> atmosphere.js
 */
(function() {
    if (!window.DS || !window.DS.initAtmosphere) {
        var s = document.createElement('script');
        s.src = '/static/js/atmosphere.js?v=20261111';
        s.async = true;
        document.head.appendChild(s);
    }
})();
