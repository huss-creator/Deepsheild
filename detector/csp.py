"""
Content-Security-Policy with a per-request nonce for inline <script> blocks.

Policy source of truth is here (not Nginx) so the nonce and the policy always agree.
  * script-src: self + the CDN host used for Bootstrap + nonce. No 'unsafe-inline' for scripts, no inline handlers.
  * style-src keeps 'unsafe-inline' because templates use inline style="" attributes. Style injection is far
    lower risk than script injection; removing it needs a template refactor (documented in docs/security.md).
  * img-src: self + data: + CSP_IMG_SOURCES (set to the S3/MinIO host when STORAGE_BACKEND=s3, because
    /media/ answers with a redirect to a pre-signed URL on that host).
  * API/JSON responses get a locked-down policy (default-src 'none').
CSP_REPORT_ONLY=true switches to Content-Security-Policy-Report-Only for staged roll-out.
"""

import secrets

from django.conf import settings

CDN_HOSTS = 'https://cdn.jsdelivr.net'


def build_policy(nonce: str, api: bool = False) -> str:
    if api:
        return "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"
    img = ' '.join(['\'self\'', 'data:', *getattr(settings, 'CSP_IMG_SOURCES', [])])
    return '; '.join([
        "default-src 'self'",
        f"script-src 'self' {CDN_HOSTS} 'nonce-{nonce}'",
        f"style-src 'self' 'unsafe-inline' {CDN_HOSTS} https://fonts.googleapis.com",
        f"font-src 'self' {CDN_HOSTS} https://fonts.gstatic.com",
        f"img-src {img}",
        "connect-src 'self'",
        "object-src 'none'",
        "frame-ancestors 'none'",
        "base-uri 'self'",
        "form-action 'self'",
    ])


class CSPMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request.csp_nonce = secrets.token_urlsafe(16)
        response = self.get_response(request)
        if 'Content-Security-Policy' in response or 'Content-Security-Policy-Report-Only' in response:
            return response
        is_api = request.path.startswith('/api/')
        header = ('Content-Security-Policy-Report-Only' if getattr(settings, 'CSP_REPORT_ONLY', False)
                  else 'Content-Security-Policy')
        # The Django admin and DRF's browsable pages use inline script/style; only the app's own pages get the strict policy.
        if request.path.startswith('/admin/'):
            response[header] = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                                "img-src 'self' data:; frame-ancestors 'none'; object-src 'none'")
        else:
            response[header] = build_policy(request.csp_nonce, api=is_api)
        return response


def csp_nonce(request):
    """Template context processor: {{ csp_nonce }}."""
    return {'csp_nonce': getattr(request, 'csp_nonce', '')}
