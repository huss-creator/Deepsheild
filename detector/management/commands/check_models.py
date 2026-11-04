"""Load every configured model and print its status (use before/after deploying weights)."""

from django.core.management.base import BaseCommand

from ai_models.registry import get_registry


class Command(BaseCommand):
    help = 'Verify model files, checksums, loading and output semantics.'

    def handle(self, *args, **options):
        registry = get_registry()
        ok = 0
        for d in registry.ordered():
            h = d.health_check()
            ok += h['status'] == 'AVAILABLE'
            self.stdout.write(f"{h['name']:<16} {h['status']:<15} v={h['version'] or '-':<8} "
                              f"{h['framework'] or '-':<8} {h['message']}")
        total = len(registry.ordered())
        self.stdout.write(f'{ok}/{total} models available')
