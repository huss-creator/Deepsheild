"""JSON 404/500 for /api/ paths; the normal HTML pages are used elsewhere."""

from django.http import JsonResponse
from django.views.defaults import page_not_found, server_error as default_server_error


def _is_api(request):
    return request.path.startswith('/api/')


def _json(request, status, code, message):
    return JsonResponse({'success': False, 'data': None, 'error': {'code': code, 'message': message},
                         'request_id': getattr(request, 'request_id', '')}, status=status)


def not_found(request, exception=None, *a, **kw):
    if _is_api(request):
        return _json(request, 404, 'NOT_FOUND', 'The requested resource was not found.')
    return page_not_found(request, exception, *a, **kw)


def server_error(request, *a, **kw):
    if _is_api(request):
        return _json(request, 500, 'INTERNAL_ERROR', 'An internal error occurred.')
    return default_server_error(request, *a, **kw)
