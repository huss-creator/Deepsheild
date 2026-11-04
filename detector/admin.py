"""
Django admin. Administrators can inspect users, analyses, jobs, model versions, reports and the
audit trail, but raw private images are never rendered or downloadable here: file fields are
excluded and derived records are read-only. The audit log is append-only.
"""

from django.contrib import admin

from detector.models import (Analysis, AuditLog, Face, ForensicAnalysis, ModelResult, ModelVersion,
                             Report, UploadedImage)


class ReadOnlyAdmin(admin.ModelAdmin):
    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False


@admin.register(UploadedImage)
class UploadedImageAdmin(ReadOnlyAdmin):
    list_display = ('public_id', 'user', 'processing_status', 'face_count', 'file_size', 'uploaded_at')
    list_filter = ('processing_status', 'uploaded_at')
    search_fields = ('public_id', 'user__username', 'sha256_hash')
    list_select_related = ('user',)
    date_hierarchy = 'uploaded_at'
    exclude = ('image', 'cropped_face', 'original_filename', 'error_message')   # no raw images / filenames


@admin.register(Analysis)
class AnalysisAdmin(ReadOnlyAdmin):
    list_display = ('id', 'uploaded_image', 'final_prediction', 'executed_models', 'unavailable_models', 'created_at')
    list_filter = ('final_prediction', 'created_at')
    list_select_related = ('uploaded_image',)
    readonly_fields = ('pipeline_info',)


@admin.register(ModelResult)
class ModelResultAdmin(ReadOnlyAdmin):
    list_display = ('analysis', 'face_index', 'model_key', 'status', 'prediction', 'model_version')
    list_filter = ('model_key', 'status')
    list_select_related = ('analysis', 'model_version')
    exclude = ('gradcam_image',)


@admin.register(Face)
class FaceAdmin(ReadOnlyAdmin):
    list_display = ('analysis', 'face_index', 'ensemble_label', 'ensemble_score', 'aligned')
    list_filter = ('ensemble_label', 'aligned')
    exclude = ('crop_image',)


@admin.register(ForensicAnalysis)
class ForensicAnalysisAdmin(ReadOnlyAdmin):
    list_display = ('analysis', 'overall_risk', 'c2pa_status', 'created_at')
    list_filter = ('overall_risk', 'c2pa_status')
    fields = ('analysis', 'overall_risk', 'overall_risk_score', 'c2pa_status', 'evidence_report')


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    """Model registry view: which exact weights produced results."""
    list_display = ('key', 'name', 'version', 'framework', 'weights_sha256', 'first_seen')
    list_filter = ('key', 'framework')
    search_fields = ('key', 'version', 'weights_sha256')

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(Report)
class ReportAdmin(ReadOnlyAdmin):
    list_display = ('analysis', 'content_sha256', 'software_version', 'created_at')
    readonly_fields = ('content', 'content_sha256', 'signature')


@admin.register(AuditLog)
class AuditLogAdmin(ReadOnlyAdmin):
    list_display = ('created_at', 'action', 'username', 'success', 'object_type', 'object_id', 'ip_address')
    list_filter = ('action', 'success', 'created_at')
    search_fields = ('username', 'object_id', 'request_id')
    date_hierarchy = 'created_at'

    def has_delete_permission(self, request, obj=None):
        return False
