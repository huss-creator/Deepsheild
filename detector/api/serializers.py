"""API serializers. Public identifiers are UUIDs; server file paths are never exposed."""

from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import extend_schema_field, inline_serializer
from rest_framework import serializers

from detector.models import Face, ModelResult, UploadedImage

_LEGACY_TO_LABEL = {'LIKELY_AUTHENTIC': 'AUTHENTIC', 'LIKELY_MANIPULATED': 'POTENTIALLY_MANIPULATED',
                    'UNCERTAIN': 'UNCERTAIN', 'NO_FACE': 'NO_FACE'}
OBJ = extend_schema_field(OpenApiTypes.OBJECT)
ANY = extend_schema_field(OpenApiTypes.ANY)


_ENVELOPES = {}


def envelope(serializer=None, many=False, name=None):
    """OpenAPI wrapper describing the {success, data, error, request_id} envelope (memoised so each
    component name maps to exactly one schema component)."""
    name = name or f'{(serializer.__name__ if serializer else "Object")}{"List" if many else ""}Envelope'
    if name not in _ENVELOPES:
        data = (serializer(many=many) if serializer else serializers.DictField())
        _ENVELOPES[name] = inline_serializer(
            name=name, fields={'success': serializers.BooleanField(), 'data': data,
                               'error': serializers.DictField(allow_null=True),
                               'request_id': serializers.CharField()})
    return _ENVELOPES[name]


def assessment_label(upload: UploadedImage):
    if upload.processing_status == UploadedImage.STATUS_FAILED:
        return 'ANALYSIS_FAILED'
    analysis = getattr(upload, 'analysis', None)
    return _LEGACY_TO_LABEL.get(analysis.final_prediction) if analysis else None


class AnalysisSerializer(serializers.ModelSerializer):
    id = serializers.UUIDField(source='public_id', read_only=True)
    status = serializers.CharField(source='processing_status', read_only=True)
    created_at = serializers.DateTimeField(source='uploaded_at', read_only=True)
    sha256 = serializers.CharField(source='sha256_hash', read_only=True)
    assessment = serializers.SerializerMethodField()

    class Meta:
        model = UploadedImage
        fields = ['id', 'status', 'created_at', 'original_filename', 'file_size', 'sha256',
                  'face_count', 'assessment']

    @extend_schema_field(OpenApiTypes.STR)
    def get_assessment(self, obj):
        return assessment_label(obj)


def job_data(upload):
    """Real job state. `stage` is the stage the worker is in; there is deliberately no percentage."""
    job = getattr(upload, 'job', None)
    if job is None:
        return None
    from detector.models import AnalysisJob
    return {'stage': job.status, 'stage_index': job.stage_index, 'stage_total': len(AnalysisJob.STAGES),
            'stages': AnalysisJob.STAGES, 'finished': job.is_terminal, 'attempts': job.attempts,
            'partial': job.partial, 'warnings': job.warnings, 'cancel_requested': job.cancel_requested,
            'error': ({'code': job.error_code, 'message': job.error_message} if job.error_code else None),
            'last_stage': job.last_stage or None, 'started_at': job.started_at, 'finished_at': job.finished_at}


class JobSerializer(serializers.Serializer):
    id = serializers.UUIDField()
    status = serializers.CharField()
    job = serializers.DictField(allow_null=True)


class AnalysisDetailSerializer(AnalysisSerializer):
    image = serializers.SerializerMethodField()
    ai_detection = serializers.SerializerMethodField()
    summary = serializers.SerializerMethodField()
    model_versions = serializers.SerializerMethodField()
    warnings = serializers.SerializerMethodField()
    report = serializers.SerializerMethodField()
    job = serializers.SerializerMethodField()
    timing_ms = serializers.SerializerMethodField()

    class Meta(AnalysisSerializer.Meta):
        fields = AnalysisSerializer.Meta.fields + ['image', 'ai_detection', 'summary', 'model_versions',
                                                    'warnings', 'report', 'job', 'timing_ms']

    def _a(self, obj):
        return getattr(obj, 'analysis', None)

    @OBJ
    def get_image(self, obj):
        a = self._a(obj)
        img = ((a.pipeline_info or {}).get('image') or {}) if a else {}
        return {'width': img.get('width'), 'height': img.get('height'),
                'processing_changes': img.get('processing_changes', [])}

    @OBJ
    def get_ai_detection(self, obj):
        a = self._a(obj)
        if not a:
            return None
        info = a.pipeline_info or {}
        level = info.get('image_level') or {}
        rep = level.get('representative_face')
        faces = info.get('faces') or []
        fe = (faces[rep]['ensemble'] if rep is not None and rep < len(faces) else {})
        return {'label': level.get('label'), 'score': level.get('score'),
                'score_kind': fe.get('score_kind'), 'ensemble_method': fe.get('method'),
                'ensemble_validated': fe.get('validated'), 'models_available': fe.get('models_available'),
                'models_expected': fe.get('models_expected'), 'disagreement': fe.get('disagreement'),
                'reasons': fe.get('reasons', []), 'explanation': level.get('explanation', ''),
                'faces_analysed': level.get('faces_analysed', len(faces))}

    @OBJ
    def get_summary(self, obj):
        a = self._a(obj)
        f = getattr(a, 'forensic_analysis', None) if a else None
        s = ((f.evidence_report or {}).get('summary') if f else None) or {}
        return {'assessment': s.get('assessment'), 'label': s.get('label'), 'reasoning': s.get('reasoning', []),
                'limitations': s.get('limitations', [])}

    @OBJ
    def get_model_versions(self, obj):
        a = self._a(obj)
        if not a:
            return []
        seen = {}
        for r in a.model_results.all():                       # prefetched: no extra queries
            mv = r.model_version
            if mv:
                seen[mv.pk] = {'key': mv.key, 'name': mv.name, 'version': mv.version,
                               'weights_sha256': mv.weights_sha256, 'framework': mv.framework}
        return sorted(seen.values(), key=lambda v: v['key'])

    @OBJ
    def get_warnings(self, obj):
        w = []
        if obj.processing_status == UploadedImage.STATUS_FAILED:
            w.append('Processing failed; no result is available.')
        job = getattr(obj, 'job', None)
        if job is not None:
            w.extend(job.warnings or [])
        a = self._a(obj)
        if a:
            d = self.get_ai_detection(obj) or {}
            if d.get('models_available') == 0:
                w.append('No AI model produced a result for this image (models not configured or unavailable).')
            elif d.get('ensemble_validated') is False:
                w.append('The ensemble has not been validated on your data; treat scores as rough indications.')
            if d.get('disagreement'):
                w.append('The AI models disagree with each other.')
        return w

    @OBJ
    def get_job(self, obj):
        return job_data(obj)

    @OBJ
    def get_report(self, obj):
        a = self._a(obj)
        rep = getattr(a, 'report', None) if a else None
        return {'sha256': rep.content_sha256, 'created_at': rep.created_at} if rep else None

    @ANY
    def get_timing_ms(self, obj):
        a = self._a(obj)
        return ((a.pipeline_info or {}).get('timings_ms') or {}).get('total_ms') if a else None


class FaceSerializer(serializers.ModelSerializer):
    id = serializers.UUIDField(source='face_id', read_only=True)
    index = serializers.IntegerField(source='face_index', read_only=True)
    bbox = serializers.SerializerMethodField()
    ensemble = serializers.SerializerMethodField()
    crop_available = serializers.SerializerMethodField()

    class Meta:
        model = Face
        fields = ['id', 'index', 'bbox', 'detector_confidence', 'aligned', 'quality_flags', 'ensemble',
                  'crop_available']
        validators = []        # read-only: avoids DRF unique-constraint introspection

    @OBJ
    def get_bbox(self, obj):
        return {'x': obj.x, 'y': obj.y, 'width': obj.w, 'height': obj.h}

    @OBJ
    def get_ensemble(self, obj):
        e = obj.ensemble or {}
        keep = ('label', 'score', 'score_kind', 'method', 'ensemble_version', 'validated', 'models_available',
                'models_expected', 'manipulated_votes', 'authentic_votes', 'score_std', 'disagreement',
                'reasons', 'explanation')
        return {k: e.get(k) for k in keep}

    @OBJ
    def get_crop_available(self, obj):
        return bool(obj.crop_image)


class ModelResultSerializer(serializers.ModelSerializer):
    key = serializers.CharField(source='model_key')
    name = serializers.CharField(source='model_name')
    version = serializers.SerializerMethodField()
    status = serializers.SerializerMethodField()
    label = serializers.SerializerMethodField()
    score = serializers.SerializerMethodField()
    threshold = serializers.SerializerMethodField()
    score_type = serializers.SerializerMethodField()
    inference_ms = serializers.SerializerMethodField()
    explanation = serializers.SerializerMethodField()

    class Meta:
        model = ModelResult
        fields = ['key', 'name', 'version', 'status', 'label', 'score', 'threshold', 'score_type',
                  'inference_ms', 'message', 'explanation']
        validators = []        # read-only: avoids DRF unique-constraint introspection

    d = lambda self, o: o.details or {}

    @ANY
    def get_version(self, o): return self.d(o).get('version', '')
    @ANY
    def get_status(self, o): return self.d(o).get('status', o.status)
    @ANY
    def get_label(self, o): return self.d(o).get('label')
    @ANY
    def get_score(self, o): return round(o.fake_probability, 4) if o.status == 'success' else None
    @ANY
    def get_threshold(self, o): return self.d(o).get('threshold')
    @ANY
    def get_score_type(self, o): return self.d(o).get('score_type', '')
    @ANY
    def get_inference_ms(self, o): return self.d(o).get('inference_ms')

    @OBJ
    def get_explanation(self, o):
        e = dict(self.d(o).get('explanation') or {})
        e['image_available'] = bool(o.gradcam_image)
        return e


class RegisterSerializer(serializers.Serializer):
    username = serializers.RegexField(r'^[\w.@+-]{3,150}$', max_length=150)
    email = serializers.EmailField(max_length=254)
    password = serializers.CharField(write_only=True, trim_whitespace=False, max_length=128)

    def validate_username(self, value):
        if User.objects.filter(username__iexact=value).exists():
            raise serializers.ValidationError('This username is already taken.')
        return value

    def validate_email(self, value):
        if User.objects.filter(email__iexact=value).exists():
            raise serializers.ValidationError('An account with this email already exists.')
        return value.lower()

    def validate(self, attrs):
        validate_password(attrs['password'], User(username=attrs['username'], email=attrs['email']))
        return attrs


class LoginSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    password = serializers.CharField(write_only=True, trim_whitespace=False, max_length=128)


class AccountDeleteSerializer(serializers.Serializer):
    password = serializers.CharField(write_only=True, trim_whitespace=False)


class LogoutSerializer(serializers.Serializer):
    refresh = serializers.CharField()
