from pathlib import Path
import csv
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
summary=json.loads((HERE/'development/summary.json').read_text(encoding='utf-8'))
with (HERE/'development/metrics.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(summary['rows'][0]))
    writer.writeheader()
    writer.writerows(summary['rows'])
checks=[]
for path in sorted((HERE/'development/raw').glob('*.json')):
    record=json.loads(path.read_text(encoding='utf-8'))
    dimension=record['case']['dimension']
    reference=record['reference']['trace']
    masks=np.zeros((len(reference),dimension))
    for index,row in enumerate(reference):
        masks[index,row['support']]=1
    uppers=np.array([row['upper'] for row in reference])
    worst=-np.inf
    for owner in record['methods']['curvature']['measured']['owned_evaluations']:
        tau=owner['tau']
        numerator=max(0.,(owner['entropy'] or 0.)-(owner['margin'] or 0.))
        if tau is not None and tau>0 and numerator>0:
            denominators=tau+masks@np.asarray(owner['mu'])
            assert np.all(denominators>0)
            worst=max(worst,float(np.max(numerator/denominators-uppers)))
    checks.append(dict(case=record['case']['id'],maximum_fraction_violation=worst,
                       passed=worst<=1e-7))
assert all(row['passed'] for row in checks)
failure=json.loads((HERE/'development/raw/primary_overlap_d6_n4_s11_b2.json').read_text())
result=failure['methods']['curvature']
incumbent=np.zeros(6)
incumbent[result['support']]=1
values=[]
for owner in result['measured']['owned_evaluations']:
    fraction=(owner['entropy']-owner['margin'])/(owner['tau']+np.asarray(owner['mu'])@incumbent)
    tangent=owner['intercept']+np.asarray(owner['gradient'])@incumbent
    values.append(dict(owner=owner['support'],fraction=fraction,tangent=tangent,
                       feasible=bool(fraction<=result['upper'] and tangent<=result['upper'])))
seal={'development_case_count':26,'confirmation_evaluations':0,
      'cross_support_original_risk_checks':checks,
      'failure_not_excluded':dict(case=failure['case']['id'],status=result['status'],
                                 true_incumbent_support=result['support'],actual_upper=result['upper'],
                                 incumbent_checks=values),
      'interpretation':'25 optimal plus one numerical master infeasibility; the failed incumbent satisfies every stored lower constraint. No all-success query/runtime improvement claim. This was the final development ranking/model combination, and confirmation remains unseen.',
      'hashes':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [HERE/'prepare.py',HERE/'study.py',HERE/'protocol.json',
                           HERE/'development/summary.json',HERE/'development/metrics.csv',HERE/'run.log']}}
(HERE/'development/seal.json').write_text(json.dumps(seal,indent=2),encoding='utf-8')
print('All cross-support fraction checks pass; 1 master failure retained; confirmation evaluations 0')
