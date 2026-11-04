from rest_framework.throttling import BaseThrottle

from detector import ratelimit


class UserWindowThrottle(BaseThrottle):
    """Per-user request ceiling shared by all authenticated endpoints (settings.RATE_LIMITS['api_user'])."""
    scope = 'api_user'

    def allow_request(self, request, view):
        user = getattr(request, 'user', None)
        if user is None or not user.is_authenticated:
            return True
        return ratelimit.allow(self.scope, f'user:{user.pk}')

    def wait(self):
        return ratelimit.retry_after(self.scope)
