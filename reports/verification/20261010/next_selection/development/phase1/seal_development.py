from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
summary=json.loads((HERE/'development/summary.json').read_text(encoding='utf-8'))
seal={'development_case_count':26,'confirmation_evaluations':0,
      'interpretation':'All four acquisition variants are negative for aggregate query reduction: 475/475/477/475 completed evaluations. The provisional EI timing tie winner was not adopted and no confirmation was run.',
      'runtime_scope':'Single paired development run; method order rotated. Cold-start effects mean timing differences alone are insufficient evidence.',
      'harness_preflight_note':'Before formal records were written the harness needed the archived generated_data field and conversion of numpy booleans to JSON. The complete predeclared case list was rerun without excluding any method or optimization failure.',
      'hashes':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [HERE/'prepare.py',HERE/'study.py',HERE/'protocol.json',
                           HERE/'development/summary.json',HERE/'development/metrics.csv',HERE/'winner.json']}}
(HERE/'development/seal.json').write_text(json.dumps(seal,indent=2),encoding='utf-8')
print('First-stage complete negative results sealed; confirmation evaluations 0')
