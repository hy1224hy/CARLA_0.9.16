"""Refresh a file-only panel from completed, SHA-bound frozen evaluations."""
import hashlib
import html
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1] / 'experiments/adversarial_u1000_R2_budget_20261104_v1'


def refresh(root=ROOT):
    read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
    output = []
    for node in read(root/'control/milestones.json'):
        name = 'frozen_'+str(node['target_train_ticks']).zfill(6)
        marker = root/'control'/(name+'_completed.json')
        if not marker.exists():
            continue
        path = root/'evaluations'/name/'analysis/critical_report.json'
        if hashlib.sha256(path.read_bytes()).hexdigest() not in read(marker)['artifact_hashes'].values():
            raise ValueError('completed report SHA mismatch')
        report = read(path)
        for row in report['interaction_quality_team_rows']:
            legacy = next(r for r in report['team_rows'] if r['team'] == row['team'])
            output.append(dict(target=node['target_train_ticks'],actual_train_ticks=node['actual_train_ticks'],
                **row,blue_eligible=legacy['blue_eligible_rate_C2']))
    payload = dict(updated_at=datetime.now(timezone.utc).isoformat(),
        scope='Completed frozen nodes only; 24 paired cases/team. A/B are not independent seeds. Unfinished nodes are absent.', rows=output)
    (root/'control/frozen_metrics.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    cols = ('target','team','predicted_conflict_C2','observed_contact_free_resolved_C2',
            'contact_free_censored_C2','blue_eligible','driving_quality_rate_C2','joint_normal_resolved_C2')
    def cell(row, col):
        value = row.get(col)
        return str(value) if col in ('target','team') else ('unmeasured' if value is None else str(round(24*value))+'/24')
    page = '<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=10><title>R2 frozen outcomes</title><h1>R2 frozen outcomes</h1><p>'+html.escape(payload['scope'])+'</p><table border=1 cellpadding=7><tr>'+''.join('<th>'+c+'</th>' for c in cols)+'</tr>'
    page += ''.join('<tr>'+''.join('<td>'+html.escape(cell(r,c))+'</td>' for c in cols)+'</tr>' for r in output)+'</table>'
    (root/'control/frozen_metrics.html').write_text(page,encoding='utf-8')
    return len(output)


if __name__ == '__main__':
    print('FROZEN_PANEL_ROWS',refresh())
