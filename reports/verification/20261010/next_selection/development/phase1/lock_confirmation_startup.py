from pathlib import Path
import csv
import hashlib
import json

HERE=Path(__file__).resolve().parent
assert not (HERE/'confirmation').exists()
protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
protocol['confirmation']['startup']='One explicit common data-screening support: top-d absolute class-mean differences divided by the sum of class standard deviations plus 1e-12. Stable ties. No retuning to tail startup after observing confirmation.'
protocol['comparison']='Same explicit data/seed and tolerances. Development uses the 26 known difficult tail-coordinate starts and rotates all five methods. Confirmation uses normal data-screening starts, compares the locked winner with LCB/master, and rotates order. All failures retained.'
protocol['runner_sha256_before_confirmation']=hashlib.sha256((HERE/'study.py').read_bytes()).hexdigest()
(HERE/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
summary=json.loads((HERE/'development/summary.json').read_text(encoding='utf-8'))
with (HERE/'development/metrics.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(summary['rows'][0]))
    writer.writeheader()
    writer.writerows(summary['rows'])
print('Confirmation startup locked before any confirmation evaluation')
