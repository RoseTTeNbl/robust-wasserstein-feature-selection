from pathlib import Path
import csv
import hashlib
import json

HERE=Path(__file__).resolve().parent
summary=json.loads((HERE/'development/summary.json').read_text(encoding='utf-8'))
with (HERE/'development/metrics.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(summary['rows'][0]))
    writer.writeheader()
    writer.writerows(summary['rows'])
secondary={}
reuse={}
for path in sorted((HERE/'development/raw').glob('*.json')):
    record=json.loads(path.read_text(encoding='utf-8'))
    result=record['methods']['core']
    for info in result['measured']['second_stage_info']:
        status=info['status']
        secondary[status]=secondary.get(status,0)+1
    reuse[record['case']['id']]=result['diagnostics'].get('model_reuse')
seal={'development_case_count':26,'confirmation_evaluations':0,
      'second_stage_status_counts':secondary,'core_model_reuse_by_case':reuse,
      'runtime_scope':'Single paired development run with rotated method order; not final timing evidence.',
      'preflight_note':'One initial harness run reached JSON serialization failure because a secondary core-point solve reported an infinite objective. No case record was published from it. The serializer now writes nonfinite solver diagnostic objectives as null, keeps all solver status information and reruns the complete fixed protocol. No method or case was excluded.',
      'hashes':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [HERE/'prepare.py',HERE/'study.py',HERE/'protocol.json',
                           HERE/'development/summary.json',HERE/'development/metrics.csv',HERE/'run.log']}}
(HERE/'development/seal.json').write_text(json.dumps(seal,indent=2),encoding='utf-8')
print(secondary)
