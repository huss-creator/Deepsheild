
from django.core.management.base import BaseCommand

from detector import model_monitoring


class Command(BaseCommand):
    help = ('Per-model-version score distribution, failure rate and window-over-window drift signals. '
            'A drift alert is a prompt to investigate, not proof of degradation (no labels here).')

    def add_arguments(self, parser):
        parser.add_argument('--days', type=int, default=7)
        parser.add_argument('--json', action='store_true')
        parser.add_argument('--fail-on-alert', action='store_true', help='exit 1 if any alert (for cron/CI)')

    def handle(self, *args, days=7, json=False, fail_on_alert=False, **opts):
        import json as _json
        rep = model_monitoring.summarise(days)
        if json:
            self.stdout.write(_json.dumps(rep, indent=2))
        else:
            if not rep:
                self.stdout.write('No model results in the last %d days.' % (2 * days))
            for r in rep:
                c, p = r['current'], r['previous']
                self.stdout.write(f"{r['model']} v{r['version']} [{r['weights_sha256']}]  "
                                  f"n={c['n']} (prev {p['n']})  fail={c['failure_rate']}  "
                                  f"flagged={c.get('flagged_rate')} (prev {p.get('flagged_rate')})  "
                                  f"mean={c.get('mean_score')} (prev {p.get('mean_score')})"
                                  + ('' if r['enough_data'] else '  [too little data for drift]'))
                for a in r['alerts']:
                    self.stdout.write(self.style.WARNING(f'   ALERT: {a}'))
        if fail_on_alert and any(r['alerts'] for r in rep):
            raise SystemExit(1)
