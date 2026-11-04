"""Small security middleware."""

import contextvars
import re
import uuid

from django.conf import settings
from django.http import HttpResponse


class RequestSizeLimitMiddleware:
    """Reject oversized request bodies from the Content-Length header BEFORE Django
    parses the multipart body (stops memory/disk exhaustion by huge uploads).
    Nginx enforces the same limit in front of Django (client_max_body_size)."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.method in ('POST', 'PUT', 'PATCH'):
            try:
                length = int(request.META.get('CONTENT_LENGTH') or 0)
            except ValueError:
                return HttpResponse('Invalid Content-Length.', status=400, content_type='text/plain')
            if length > settings.MAX_REQUEST_BODY_SIZE:
                return HttpResponse('Request body too large.', status=413, content_type='text/plain')
        return self.get_response(request)


request_id_var = contextvars.ContextVar('request_id', default='')
_REQUEST_ID_RE = re.compile(r'^[A-Za-z0-9._-]{8,64}$')


class RequestIDMiddleware:
    """Attach a request ID (client-supplied X-Request-ID if well-formed, else generated) to the
    request, the response header, API envelopes, audit rows and logs."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        incoming = request.META.get('HTTP_X_REQUEST_ID', '')
        request.request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request.request_id)
        try:
            response = self.get_response(request)
        finally:
            request_id_var.reset(token)
        response['X-Request-ID'] = request.request_id
        return response
