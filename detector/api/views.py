"""REST API v1. Every resource is looked up through the requesting user (object-level
authorization): another user's id yields the same 404 as a non-existent id, and the attempt is
audit-logged."""

import logging
import uuid

from django.conf import settings
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.core.cache import cache
from django.core.exceptions import ValidationError as DjangoValidationError
from django.db import connection
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenRefreshView

from detector import audit, ratelimit, reports, retention
from detector.api import serializers as ser
from detector.api.envelope import ApiError
from detector.api.throttles import UserWindowThrottle
from detector.forms import ImageUploadForm
from detector.image_validation import validate_uploaded_image
from detector.media import serve_media_file
from detector.models import Face, ForensicAnalysis, ModelResult, Report, UploadedImage
from detector.jobs import JobLimitError
from detector.services import process_upload
from rest_framework.exceptions import AuthenticationFailed, NotFound, Throttled, ValidationError

logger = logging.getLogger('deepshield.api')
UUID_PARAM = OpenApiParameter('id', str, OpenApiParameter.PATH, description='Analysis UUID')


def _throttled(request, scope):
    audit.audit(audit.THROTTLED, request, success=False, scope=scope)
    return Throttled(wait=ratelimit.retry_after(scope))


def _tokens(user):
    refresh = RefreshToken.for_user(user)
    return {'access': str(refresh.access_token), 'refresh': str(refresh),
            'access_expires_in': int(settings.SIMPLE_JWT['ACCESS_TOKEN_LIFETIME'].total_seconds())}


# ------------------------------------------------------------------ auth
class RegisterView(APIView):
    authentication_classes, permission_classes = [], [AllowAny]

    @extend_schema(request=ser.RegisterSerializer, responses={201: ser.envelope(name='RegisterEnvelope')},
                   tags=['auth'], summary='Create an account')
    def post(self, request, version=None):
        if not ratelimit.allow('register', ratelimit.client_ip(request)):
            raise _throttled(request, 'register')
        s = ser.RegisterSerializer(data=request.data)
        s.is_valid(raise_exception=True)
        user = User.objects.create_user(s.validated_data['username'], s.validated_data['email'],
                                        s.validated_data['password'])
        audit.audit(audit.REGISTER, request, user=user, channel='api')
        return Response({'user': {'username': user.username, 'email': user.email}, **_tokens(user)},
                        status=status.HTTP_201_CREATED)


class LoginView(APIView):
    authentication_classes, permission_classes = [], [AllowAny]

    def get_authenticate_header(self, request):     # keeps AuthenticationFailed a 401 (DRF turns it into 403 otherwise)
        return 'Bearer'

    @extend_schema(request=ser.LoginSerializer, responses={200: ser.envelope(name='LoginEnvelope')},
                   tags=['auth'], summary='Obtain access + refresh tokens')
    def post(self, request, version=None):
        s = ser.LoginSerializer(data=request.data)
        s.is_valid(raise_exception=True)
        username, password = s.validated_data['username'].strip(), s.validated_data['password']
        ip = ratelimit.client_ip(request)
        pair_id = f'{ip}|{username.lower()}'
        if ratelimit.is_limited('login_fail', pair_id) or ratelimit.is_limited('login_fail_ip', ip):
            raise _throttled(request, 'login_fail')
        user = authenticate(request, username=username, password=password)
        if user is None:
            ratelimit.hit('login_fail', pair_id)
            ratelimit.hit('login_fail_ip', ip)
            audit.audit(audit.LOGIN_FAILED, request, success=False, channel='api',
                        username_hash=audit.hash_identifier(username))
            raise AuthenticationFailed('Invalid username or password.')
        ratelimit.reset('login_fail', pair_id)
        audit.audit(audit.LOGIN, request, user=user, channel='api')
        return Response(_tokens(user))


class RefreshView(TokenRefreshView):
    """Rotating refresh: the old refresh token is blacklisted and a new one returned."""
    authentication_classes, permission_classes = [], [AllowAny]

    @extend_schema(tags=['auth'], summary='Exchange a refresh token (rotates it)',
                   responses={200: ser.envelope(name='RefreshEnvelope')})
    def post(self, request, *args, **kwargs):
        return super().post(request, *args, **kwargs)


class LogoutView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_classes = [UserWindowThrottle]

    @extend_schema(request=ser.LogoutSerializer, responses={200: ser.envelope(name='LogoutEnvelope')},
                   tags=['auth'], summary='Revoke a refresh token')
    def post(self, request, version=None):
        s = ser.LogoutSerializer(data=request.data)
        s.is_valid(raise_exception=True)
        try:
            token = RefreshToken(s.validated_data['refresh'])
            if str(token.get('user_id')) != str(request.user.pk):
                raise ValidationError({'refresh': ['This token does not belong to the authenticated user.']})
            token.blacklist()
        except TokenError:
            raise ValidationError({'refresh': ['Invalid or already revoked token.']})
        audit.audit(audit.LOGOUT, request, channel='api')
        return Response({'logged_out': True})


class AccountView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_classes = [UserWindowThrottle]

    @extend_schema(tags=['auth'], request=ser.AccountDeleteSerializer, responses={204: None},
                   summary='Permanently delete the account and every analysis/file (requires password)')
    def delete(self, request, version=None):
        s = ser.AccountDeleteSerializer(data=request.data)
        s.is_valid(raise_exception=True)
        ident = f'acct|{request.user.pk}'
        if ratelimit.is_limited('login_fail', ident):
            raise _throttled(request, 'login_fail')
        if not request.user.check_password(s.validated_data['password']):
            ratelimit.hit('login_fail', ident)
            audit.audit(audit.ACCESS_DENIED, request, success=False, resource='account_delete')
            raise ValidationError({'password': ['Incorrect password.']})
        ratelimit.reset('login_fail', ident)
        retention.delete_account(request.user, request)
        return Response(status=status.HTTP_204_NO_CONTENT)


# -------------------------------------------------------------- analyses
class OwnedMixin:
    permission_classes = [IsAuthenticated]
    throttle_classes = [UserWindowThrottle]

    def owned_upload(self, request, pk, **qs):
        """Fetch strictly through the requesting user; 404 (and audit) otherwise."""
        base = UploadedImage.objects.select_related('job', 'analysis', 'analysis__report', 'analysis__forensic_analysis')
        obj = base.filter(user=request.user, public_id=pk).first()
        if obj is None:
            if UploadedImage.objects.filter(public_id=pk).exists():
                audit.audit(audit.ACCESS_DENIED, request, success=False, resource='analysis', requested=str(pk))
            raise NotFound()
        return obj


class AnalysisListCreateView(OwnedMixin, APIView):
    @extend_schema(tags=['analyses'], summary='List your analyses (paginated)', operation_id='analyses_list',
                   parameters=[OpenApiParameter('page', int), OpenApiParameter('status', str)],
                   responses={200: ser.envelope(ser.AnalysisSerializer, many=True)})
    def get(self, request, version=None):
        from rest_framework.pagination import PageNumberPagination
        qs = (UploadedImage.objects.filter(user=request.user).select_related('analysis')
              .order_by('-uploaded_at'))
        if request.query_params.get('status'):
            qs = qs.filter(processing_status=request.query_params['status'])
        pager = PageNumberPagination()
        page = pager.paginate_queryset(qs, request, view=self)
        return Response({'count': pager.page.paginator.count, 'next': pager.get_next_link(),
                         'previous': pager.get_previous_link(),
                         'results': ser.AnalysisSerializer(page, many=True).data})

    @extend_schema(tags=['analyses'], summary='Upload an image and analyse it',
                   description='multipart/form-data with field `image` (JPEG or PNG). Limits: 10 MB, '
                               '8000 px per side, 25 megapixels. Validation uses file CONTENT, not the '
                               'extension or Content-Type. Processing is ASYNCHRONOUS: the response is 202 with '
                               'the job state; poll `GET /analyses/{id}/status/` until `job.finished`. At most '
                               '3 analyses per user may be in progress (429 TOO_MANY_ACTIVE_JOBS); 503 QUEUE_FULL '
                               'when the queue is saturated.',
                   request={'multipart/form-data': {'type': 'object', 'properties': {
                       'image': {'type': 'string', 'format': 'binary'}}, 'required': ['image']}},
                   responses={202: ser.envelope(ser.AnalysisDetailSerializer), 400: ser.envelope(name='ErrorEnvelope'),
                              413: ser.envelope(name='TooLargeEnvelope'), 429: ser.envelope(name='RateLimitEnvelope')})
    def post(self, request, version=None):
        if not ratelimit.allow('upload', f'user:{request.user.pk}'):
            raise _throttled(request, 'upload')
        f = request.FILES.get('image')
        if f is None:
            raise ValidationError({'image': ['This field is required (multipart/form-data file).']})
        try:
            info = validate_uploaded_image(f)
        except DjangoValidationError as exc:
            if exc.code == 'too_large':
                raise ApiError('FILE_TOO_LARGE', exc.messages[0], 413)
            raise ApiError('INVALID_IMAGE', exc.messages[0] if exc.code != 'unsupported' else
                           'The uploaded file is not a supported image.', 400, {'reason': exc.code})
        original = ImageUploadForm._display_name(f.name)
        f.name = f'upload.{info.extension}'
        try:
            rec = process_upload(request.user, f, original)
        except JobLimitError as exc:
            audit.audit(audit.THROTTLED, request, success=False, scope=exc.code.lower())
            raise ApiError(exc.code, exc.message, exc.status)
        job = getattr(rec, 'job', None)
        if job is not None and job.error_code == 'QUEUE_UNAVAILABLE':
            raise ApiError('QUEUE_UNAVAILABLE', 'The analysis queue is unavailable. Please try again later.', 503)
        audit.audit(audit.ANALYSIS_CREATED, request, obj=rec, channel='api',
                    success=rec.processing_status != UploadedImage.STATUS_FAILED)
        rec = self.owned_upload(request, rec.public_id)
        return Response(ser.AnalysisDetailSerializer(rec).data, status=status.HTTP_202_ACCEPTED)


class AnalysisDetailView(OwnedMixin, APIView):
    @extend_schema(tags=['analyses'], parameters=[UUID_PARAM], summary='Analysis detail', operation_id='analyses_detail',
                   responses={200: ser.envelope(ser.AnalysisDetailSerializer)})
    def get(self, request, pk, version=None):
        rec = self.owned_upload(request, pk)
        if rec.processing_status in (UploadedImage.STATUS_COMPLETED, UploadedImage.STATUS_NO_FACE):
            rec = (UploadedImage.objects.select_related('job', 'analysis', 'analysis__report', 'analysis__forensic_analysis')
                   .prefetch_related('analysis__model_results__model_version').get(pk=rec.pk))
        return Response(ser.AnalysisDetailSerializer(rec).data)

    @extend_schema(tags=['analyses'], parameters=[UUID_PARAM], summary='Delete an analysis and its files',
                   responses={204: None})
    def delete(self, request, pk, version=None):
        rec = self.owned_upload(request, pk)
        audit.audit(audit.ANALYSIS_DELETED, request, obj=rec, channel='api')
        rec.delete()                                  # cascades; signals remove the stored files
        return Response(status=status.HTTP_204_NO_CONTENT)


class AnalysisStatusView(OwnedMixin, APIView):
    @extend_schema(tags=['analyses'], parameters=[UUID_PARAM], summary='Job status (cheap; use for polling)',
                   responses={200: ser.envelope(ser.JobSerializer)})
    def get(self, request, pk, version=None):
        rec = (UploadedImage.objects.select_related('job').filter(user=request.user, public_id=pk).first())
        if rec is None:
            if UploadedImage.objects.filter(public_id=pk).exists():
                audit.audit(audit.ACCESS_DENIED, request, success=False, resource='analysis', requested=str(pk))
            raise NotFound()
        return Response({'id': str(rec.public_id), 'status': rec.processing_status, 'job': ser.job_data(rec)})


class AnalysisCancelView(OwnedMixin, APIView):
    @extend_schema(tags=['analyses'], parameters=[UUID_PARAM], summary='Cancel a queued or running analysis',
                   description='A queued job is cancelled immediately; a running job stops at its next '
                               'checkpoint (status stays in progress until then). 409 if already finished.',
                   request=None, responses={200: ser.envelope(ser.JobSerializer), 409: ser.envelope(name='ConflictEnvelope')})
    def post(self, request, pk, version=None):
        from detector import jobs
        rec = self.owned_upload(request, pk)
        job = getattr(rec, 'job', None)
        if job is None or job.is_terminal:
            raise ApiError('JOB_FINISHED', 'This analysis has already finished.', 409)
        try:
            outcome = jobs.request_cancel(job)
        except ValueError:
            raise ApiError('JOB_FINISHED', 'This analysis has already finished.', 409)
        audit.audit(audit.ANALYSIS_CANCELLED, request, obj=rec, channel='api')
        rec = self.owned_upload(request, pk)
        return Response({'id': str(rec.public_id), 'outcome': outcome, 'status': rec.processing_status,
                         'job': ser.job_data(rec)})


class _AnalysisPart(OwnedMixin, APIView):
    def analysis_for(self, request, pk):
        rec = self.owned_upload(request, pk)
        a = getattr(rec, 'analysis', None)
        if a is None:
            raise ApiError('NOT_AVAILABLE', 'No analysis result is available for this upload '
                           f'(status: {rec.processing_status}).', 409)
        return rec, a


class AnalysisFacesView(_AnalysisPart):
    @extend_schema(tags=['analyses'], parameters=[UUID_PARAM], summary='Faces detected in the image',
                   responses={200: ser.envelope(ser.FaceSerializer, many=True)})
    def get(self, request, pk, version=None):
        _, a = self.analysis_for(request, pk)
        return Response(ser.FaceSerializer(Face.objects.filter(analysis=a), many=True).data)


class AnalysisModelsView(_AnalysisPart):
    @extend_schema(tags=['analyses'], parameters=[UUID_PARAM], summary='Per-face, per-model results',
                   responses={200: ser.envelope(name='ModelResultsEnvelope')})
    def get(self, request, pk, version=None):
        _, a = self.analysis_for(request, pk)
        by_face = {}
        for r in ModelResult.objects.filter(analysis=a).order_by('face_index', 'model_key'):
            by_face.setdefault(r.face_index, []).append(r)
        out = []
        for idx, rows in by_face.items():
            data = ser.ModelResultSerializer(rows, many=True).data
            live = [d for d in data if d['status'] == 'AVAILABLE']
            out.append({'face_index': idx, 'results': data,
                        'disagreement': {'manipulated': [d['key'] for d in live if d['label'] == 'MANIPULATED'],
                                         'authentic': [d['key'] for d in live if d['label'] == 'AUTHENTIC'],
                                         'unavailable': [d['key'] for d in data if d['status'] != 'AVAILABLE']}})
        return Response(out)


class AnalysisForensicsView(_AnalysisPart):
    @extend_schema(tags=['analyses'], parameters=[UUID_PARAM], summary='Forensic indicators, provenance, integrity',
                   responses={200: ser.envelope(name='ForensicsEnvelope')})
    def get(self, request, pk, version=None):
        _, a = self.analysis_for(request, pk)
        f = ForensicAnalysis.objects.filter(analysis=a).first()
        rep = reports._strip_artifacts((f.evidence_report if f else {}) or {})
        if not rep:
            raise ApiError('NOT_AVAILABLE', 'Forensic analysis is not available for this analysis.', 409)
        return Response({k: rep.get(k) for k in ('forensic_indicators', 'provenance', 'integrity',
                                                 'explainability', 'summary')})


class AnalysisReportView(_AnalysisPart):
    @extend_schema(tags=['analyses'], parameters=[UUID_PARAM], summary='Evidence report with integrity hash',
                   responses={200: ser.envelope(name='ReportEnvelope')})
    def get(self, request, pk, version=None):
        rec, a = self.analysis_for(request, pk)
        rep = Report.objects.filter(analysis=a).first()
        if rep is None:
            raise ApiError('NOT_AVAILABLE', 'No report exists for this analysis.', 409)
        audit.audit(audit.REPORT_VIEWED, request, obj=rec, channel='api')
        return Response({'report': rep.content, 'sha256': rep.content_sha256, 'signature': rep.signature,
                         'software_version': rep.software_version,
                         'verification': reports.verify_report(rep)})


class AnalysisImageView(_AnalysisPart):
    @extend_schema(tags=['artifacts'], parameters=[UUID_PARAM], summary='Original uploaded image (binary)',
                   responses={(200, 'image/*'): bytes})
    def get(self, request, pk, version=None):
        return serve_media_file(self.owned_upload(request, pk).image.name)


class FaceCropView(_AnalysisPart):
    @extend_schema(tags=['artifacts'], parameters=[UUID_PARAM, OpenApiParameter('index', int, OpenApiParameter.PATH)],
                   summary='Face crop (PNG)', responses={(200, 'image/png'): bytes})
    def get(self, request, pk, index, version=None):
        _, a = self.analysis_for(request, pk)
        face = Face.objects.filter(analysis=a, face_index=index).first()
        if not face or not face.crop_image:
            raise NotFound()
        return serve_media_file(face.crop_image.name)


class ForensicImageView(_AnalysisPart):
    FIELDS = {'ela': 'ela_image', 'frequency': 'freq_image', 'noise': 'noise_image',
              'copy_move': 'copy_move_image', 'face_boundary': 'face_boundary_image',
              'resampling': 'resampling_image'}

    @extend_schema(tags=['artifacts'], parameters=[UUID_PARAM, OpenApiParameter('test', str, OpenApiParameter.PATH)],
                   summary='Forensic visualisation (PNG)', responses={(200, 'image/png'): bytes})
    def get(self, request, pk, test, version=None):
        _, a = self.analysis_for(request, pk)
        field = self.FIELDS.get(test)
        f = ForensicAnalysis.objects.filter(analysis=a).first()
        img = getattr(f, field, None) if (f and field) else None
        if not img:
            raise NotFound()
        return serve_media_file(img.name)


# ---------------------------------------------------------------- system
class ModelsView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_classes = [UserWindowThrottle]

    @extend_schema(tags=['system'], summary='Configured AI models, versions and status',
                   responses={200: ser.envelope(name='ModelsEnvelope')})
    def get(self, request, version=None):
        from ai_models.registry import get_registry
        reg = get_registry()
        out = []
        for d in reg.ordered():
            h = d.health_check()
            m = d.spec.metadata()
            out.append({**h, 'preprocessing_version': m['preprocessing_version'], 'threshold': m['threshold'],
                        'threshold_version': m['threshold_version'], 'weights_checksum': m['weights_checksum'],
                        'training_dataset': m['training_dataset'],
                        'training_dataset_version': m['training_dataset_version'],
                        'evaluation_metrics': m['evaluation_metrics']})
        return Response(out)


class HealthView(APIView):
    authentication_classes, permission_classes = [], [AllowAny]

    @extend_schema(tags=['system'], summary='Liveness: is the service up?', responses={200: ser.envelope(name='HealthEnvelope')})
    def get(self, request, version=None):
        return Response({'status': 'ok', 'version': settings.DEEPSHIELD_VERSION})


class ReadinessView(APIView):
    authentication_classes, permission_classes = [], [AllowAny]

    @extend_schema(tags=['system'], summary='Readiness: can the service process analyses?',
                   responses={200: ser.envelope(name='ReadyEnvelope'), 503: ser.envelope(name='NotReadyEnvelope')})
    def get(self, request, version=None):
        checks = {}

        def run(name, fn):
            try:
                ok, info = fn()
            except Exception:
                logger.exception('readiness check %s failed', name)
                ok, info = False, 'check failed'
            checks[name] = {'ok': bool(ok), 'info': info}

        def db():
            with connection.cursor() as c:
                c.execute('SELECT 1')
            return True, 'reachable'

        def cache_check():
            k = f'ready:{uuid.uuid4().hex}'
            cache.set(k, '1', 5)
            return cache.get(k) == '1', 'read/write'

        def storage():
            import os
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            p = os.path.join(settings.MEDIA_ROOT, f'.ready_{uuid.uuid4().hex}')
            with open(p, 'w') as fh:
                fh.write('x')
            os.remove(p)
            return True, 'writable'

        def queue():
            if settings.CELERY_TASK_ALWAYS_EAGER:
                return True, 'eager mode (tasks run in-process; no broker)'
            from config.celery import app
            with app.connection_for_read() as conn:
                conn.ensure_connection(max_retries=1, timeout=2)
            from detector.models import AnalysisJob
            return True, f'broker reachable; {AnalysisJob.objects.filter(status="QUEUED").count()} queued'

        def models():
            from ai_models.detectors import resolve_device
            from ai_models.registry import get_registry
            reg = get_registry()
            n, need = reg.available_count(), settings.READINESS_MIN_MODELS
            if settings.DEVICE == 'cuda' and resolve_device('cuda') != 'cuda':
                return False, 'GPU required but unavailable'
            return n >= need, f'{n} of {len(reg.ordered())} models available (required: {need})'

        for name, fn in (('database', db), ('cache', cache_check), ('storage', storage), ('queue', queue), ('models', models)):
            run(name, fn)
        if all(c['ok'] for c in checks.values()):
            return Response({'status': 'ready', 'checks': checks})
        return Response({'__error__': {'code': 'NOT_READY', 'message': 'Service is not ready to process analyses.',
                                       'details': checks}}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
