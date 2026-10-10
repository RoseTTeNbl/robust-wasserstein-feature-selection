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
cross_checks=[]
for path in sorted((HERE/'development/raw').glob('*.json')):
    record=json.loads(path.read_text(encoding='utf-8'))
    dimension=record['case']['dimension']
    reference=record['reference']['trace']
    masks=np.zeros((len(reference),dimension))
    for index,row in enumerate(reference):
        masks[index,row['support']]=1
    uppers=np.array([row['upper'] for row in reference])
    worst=-np.inf
    fractions=0
    tangent_worst=-np.inf
    for owner in record['methods']['ratio']['measured']['owned_evaluations']:
        tangent=owner['intercept']+masks @ np.asarray(owner['gradient'])
        tangent_worst=max(tangent_worst,float(np.max(tangent-uppers)))
        tau=owner['tau']
        numerator=max(0.,(owner['entropy'] or 0.)-(owner['margin'] or 0.))
        if tau is not None and tau>0 and numerator>0:
            denominators=tau+masks@np.asarray(owner['mu'])
            assert np.all(denominators>0)
            worst=max(worst,float(np.max(numerator/denominators-uppers)))
            fractions+=1
    cross_checks.append(dict(case=record['case']['id'],fraction_count=fractions,
                              supports=len(reference),maximum_fraction_violation=worst,
                              maximum_tangent_violation=tangent_worst,
                              passed=worst<=1e-7 and tangent_worst<=1e-7))
assert all(row['passed'] for row in cross_checks)
seal={'development_case_count':26,'confirmation_evaluations':0,
      'cross_support_original_risk_checks':cross_checks,
      'runtime_scope':'Single paired development run with rotated method order; final timing will use repeated paired runs.',
      'interpretation':'Full fractions retain curvature of the same repaired dual certificate; fewer complete support solves are not credited to GP scoring. Shared auxiliary variables and added MILP constraints carry their measured cost.',
      'hashes':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [HERE/'prepare.py',HERE/'study.py',HERE/'protocol.json',
                           HERE/'development/summary.json',HERE/'development/metrics.csv',HERE/'run.log']}}
(HERE/'development/seal.json').write_text(json.dumps(seal,indent=2),encoding='utf-8')
print('All 26 cross-support fraction/tangent checks passed; confirmation evaluations: 0')
