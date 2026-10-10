from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path('D:/OptExp/RubustHW')
HERE=Path(__file__).resolve().parent
PACKAGE=HERE/'source/robusthw'
assert not PACKAGE.exists()
shutil.copytree(ROOT/'code/robusthw',PACKAGE,ignore=shutil.ignore_patterns('__pycache__'))
expected={'oracle':'c056a4fdfa6fae866d027ae5a3575b02e7cafc515f58f22323d699a149f4ef9a',
          'solver':'7a3a436724b831ddce5ca29396f9cdf28043f082cb97fecf62ea25f9132f4b48',
          'history':'8a6a349f2f4164f670fd5ded20bc1cd40c69675843b64ece8d0edd07375e1899',
          'adaptive':'1947ca4f37e566d9901ecdd48f99117860e19039bad34b42c633750cad1564f1'}
for name,digest in expected.items():
    assert hashlib.sha256((PACKAGE/f'{name}.py').read_bytes()).hexdigest()==digest
base=(PACKAGE/'adaptive.py').read_text(encoding='utf-8')
for name in ('baseline','screened','core'):
    variant=base
    if name=='screened':
        variant=variant.replace('from .history import EvaluationHistory, proposal_pool',
                                'from .history import EvaluationHistory\nfrom .screened_pool import proposal_pool')
    (PACKAGE/f'adaptive_{name}.py').write_text(variant,encoding='utf-8',newline='\n')
(PACKAGE/'screened_pool.py').write_text('''"""Two fixed data-informed candidates; the remaining pool/ranking is v3."""
import numpy as np
from .history import proposal_pool as original_pool

def proposal_pool(problem, history, incumbent, *, count, rng, guard=None, remaining=None):
    means=[q @ x for q,x in zip(problem.weights,problem.samples)]
    scales=[np.sqrt(q @ ((x-mean)**2))
            for q,x,mean in zip(problem.weights,problem.samples,means)]
    difference=np.abs(means[0]-means[1])
    denominator=scales[0]+scales[1]+1e-12
    ordinary=difference/denominator
    robust=np.maximum(0.,difference-problem.theta.sum())/denominator
    mandatory=[]
    for values in (ordinary,robust):
        support=tuple(sorted(np.argsort(-values,kind="stable")[:problem.budget]))
        if support not in mandatory:
            mandatory.append(support)
    original=original_pool(problem,history,incumbent,count=count,rng=rng,
                           guard=guard,remaining=remaining)
    supports=mandatory+[problem.support(z) for z in original
                         if problem.support(z) not in mandatory]
    return [problem.mask(s).astype(float) for s in supports[:count]]
''',encoding='utf-8')
old=Path('C:/Users/27512/AppData/Local/Temp/risk-acquisition-study-20261010-6f4d731a')
protocol=json.loads((old/'protocol.json').read_text(encoding='utf-8'))
protocol['scope']='Second-stage development only: independent pool enrichment and existing core-point cut selection; frozen simplified core'
protocol['rules']={'baseline':'Original v3 with simplified oracle/solver; no pool/cut-selection change',
                   'screened':'Original v3 ranking/pool with two reserved candidate supports: ordinary class mean/std and robust mean separation; no cut-selection change',
                   'core':'Original v3 pool/ranking and core_point=True; this changes dual-face cut selection, not the acquisition score',
                   'master':'Same simplified-core master-only baseline; extra contextual comparison'}
protocol['selection']='Report independent development effects first. No automatic winner and no confirmation before root selects one final simple rule. Do not mix pool enrichment and core-point yet.'
protocol['cost_accounting']='Count all actual LP/cone solver invocations and dual-face second-stage attempts/statuses, alongside full binary evaluations, master calls and total elapsed time. core_point=True currently disables persistent logistic models; include construction cost and do not attribute it solely to the additional solve.'
protocol['sources']={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(PACKAGE.rglob('*.py'))}
protocol['first_stage']='All four ranking-rule negative results remain in '+str(old)
(HERE/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
runner=(old/'study.py').read_text(encoding='utf-8')
runner=runner.replace('import robusthw.solver as outer','import robusthw.solver as outer\nimport robusthw.oracle as inner')
runner=runner.replace('        self.binary_only = True','''        self.binary_only = True
        self.continuous_calls = self.second_stage_calls = 0
        self.second_stage_info = []''')
runner=runner.replace('        return super().solve(z, **kwargs)', '''        result=super().solve(z, **kwargs)
        second=result.certificate.get("dual",{}).get("core_point")
        if second is not None:
            self.second_stage_calls += 1
            self.second_stage_info.append(second)
        return result''')
runner=runner.replace('    def reprice(self, *args, **kwargs):', '''    def _cvx_solve(self, *args, **kwargs):
        deadline=args[1] if len(args)>1 else kwargs.get("deadline")
        if deadline is None or deadline>perf_counter():
            self.continuous_calls += 1
        return super()._cvx_solve(*args, **kwargs)

    def reprice(self, *args, **kwargs):''')
runner=runner.replace('outer.GateOracle = MeasuredOracle','''actual_lp=inner._lp
def counted_lp(*args,**kwargs):
    deadline=kwargs.get("deadline")
    if deadline is None or deadline>perf_counter():
        MeasuredOracle.last.continuous_calls += 1
    return actual_lp(*args,**kwargs)
inner._lp=counted_lp
outer.GateOracle = MeasuredOracle''')
runner=runner.replace('warm_support=warm, random_state=seed, **SETTINGS)',
                      'warm_support=warm, random_state=seed, **{**SETTINGS,"core_point":rule=="core"})')
runner=runner.replace('external_seconds=perf_counter()-started)', '''external_seconds=perf_counter()-started,
                                continuous_calls=oracle.continuous_calls,
                                second_stage_calls=oracle.second_stage_calls,
                                second_stage_info=oracle.second_stage_info)''')
runner=runner.replace("    diagnostics = result.get('diagnostics',{})", "    diagnostics = result.get('diagnostics',{})")
runner=runner.replace("                all_checks=all(record['checks'][rule].values()))", '''                continuous_calls=result.get("measured",{}).get("continuous_calls"),
                second_stage_calls=result.get("measured",{}).get("second_stage_calls"),
                all_checks=all(record['checks'][rule].values()))''')
runner=runner.replace("'seconds','masters')", "'seconds','masters','continuous_calls','second_stage_calls')")
runner=runner.replace("method_metrics(record,'lcb')", "method_metrics(record,'baseline')")
runner=runner.replace("if rule!='lcb':", "if rule!='baseline':")
runner=runner.replace("'pairs_vs_lcb'", "'pairs_vs_baseline'")
runner=runner.replace("methods=['lcb','ei','guard','ei_guard','master']", "methods=['baseline','screened','core','master']")
runner=runner.replace("    if stage=='development':\n        eligible=", "    if False:\n        eligible=")
(HERE/'study.py').write_text(runner,encoding='utf-8',newline='\n')
print(HERE)
