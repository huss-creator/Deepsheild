"""Uniform JSON envelope + central exception handling for the API.

Success: {"success": true,  "data": ..., "error": null, "request_id": "..."}
Error:   {"success": false, "data": null, "error": {"code", "message", "details?"}, "request_id": "..."}
Unhandled exceptions are logged server-side with the request ID and answered with a generic
INTERNAL_ERROR: stack traces, paths and settings never reach the client."""

import logging

from django.core.exceptions import PermissionDenied as DjangoPermissionDenied
from django.http import Http404
from rest_framework import exceptions as drf
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response

logger = logging.getLogger('deepshield.api')


class ApiError(drf.APIException):
    """Domain error with a stable machine-readable code."""

    def __init__(self, code: str, message: str, status: int = 400, details=None):
        super().__init__(detail=message)
        self.status_code, self.error_code, self.message, self.details = status, code, message, details


_MAP = [
    (drf.ValidationError, 'VALIDATION_ERROR', 'The request was invalid.'),
    (drf.NotAuthenticated, 'UNAUTHENTICATED', 'Authentication credentials were missing or invalid.'),
    (drf.AuthenticationFailed, 'UNAUTHENTICATED', 'Authentication credentials were missing or invalid.'),
    (drf.PermissionDenied, 'FORBIDDEN', 'You do not have permission to perform this action.'),
    (drf.NotFound, 'NOT_FOUND', 'The requested resource was not found.'),
    (drf.MethodNotAllowed, 'METHOD_NOT_ALLOWED', 'This method is not allowed for this resource.'),
    (drf.Throttled, 'RATE_LIMITED', 'Too many requests. Please retry later.'),
    (drf.ParseError, 'MALFORMED_REQUEST', 'The request body could not be parsed.'),
    (drf.UnsupportedMediaType, 'UNSUPPORTED_MEDIA_TYPE', 'Unsupported request content type.'),
]


def _request_id(context):
    req = context.get('request')
    return getattr(req, 'request_id', '') if req is not None else ''


def exception_handler(exc, context):
    if isinstance(exc, Http404):
        exc = drf.NotFound()
    elif isinstance(exc, DjangoPermissionDenied):
        exc = drf.PermissionDenied()
    if isinstance(exc, ApiError):
        err = {'code': exc.error_code, 'message': exc.message}
        if exc.details:
            err['details'] = exc.details
        return Response({'__error__': err}, status=exc.status_code)
    if isinstance(exc, drf.APIException):
        code, message = next(((c, m) for cls, c, m in _MAP if isinstance(exc, cls)),
                             ('ERROR', 'The request could not be completed.'))
        err = {'code': code, 'message': message}
        if isinstance(exc, drf.ValidationError):
            err['details'] = exc.detail
        resp = Response({'__error__': err}, status=exc.status_code)
        if isinstance(exc, drf.Throttled) and exc.wait:
            resp['Retry-After'] = str(int(exc.wait))
        if isinstance(exc, (drf.NotAuthenticated, drf.AuthenticationFailed)):
            resp['WWW-Authenticate'] = 'Bearer'
        return resp
    logger.exception('unhandled API error', extra={'request_id': _request_id(context)})
    return Response({'__error__': {'code': 'INTERNAL_ERROR', 'message': 'An internal error occurred.'}},
                    status=500)


class EnvelopeJSONRenderer(JSONRenderer):
    def render(self, data, accepted_media_type=None, renderer_context=None):
        ctx = renderer_context or {}
        response, request = ctx.get('response'), ctx.get('request')
        rid = getattr(request, 'request_id', '') if request is not None else ''
        if response is not None and response.status_code == 204:
            return b''
        if isinstance(data, dict) and '__error__' in data:
            body = {'success': False, 'data': None, 'error': data['__error__'], 'request_id': rid}
        elif response is not None and response.status_code >= 400:
            body = {'success': False, 'data': None,
                    'error': {'code': 'ERROR', 'message': 'The request could not be completed.'},
                    'request_id': rid}
        else:
            body = {'success': True, 'data': data, 'error': None, 'request_id': rid}
        return super().render(body, accepted_media_type, renderer_context)
