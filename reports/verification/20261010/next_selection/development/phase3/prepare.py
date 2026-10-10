from pathlib import Path
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
PREVIOUS=Path('C:/Users/27512/AppData/Local/Temp/risk-acquisition-study-phase2-20261010-913c5e72')
PACKAGE=HERE/'source/robusthw'
assert not PACKAGE.exists()
shutil.copytree(PREVIOUS/'source/robusthw',PACKAGE,ignore=shutil.ignore_patterns('__pycache__'))
protocol=json.loads((PREVIOUS/'protocol.json').read_text(encoding='utf-8'))
protocol['scope']='Third-stage development only: retain the same completed dual fraction in the existing shared-product binary master'
protocol['rules']={'baseline':'Original v3 on the frozen simplified oracle/solver',
                   'ratio':'Existing risk/ratio/master search, root_rounds=0/core=False; one owned tangent plus the same certificate fraction; D shared eta*z variables',
                   'master':'Same-core tangent/master search, root_rounds=0/core=False'}
protocol['selection']='Only one new mathematical rule. Report complete development effects before root chooses a final rule; confirmation remains unseen.'
protocol['cost_accounting']='Count full binary evaluations, continuous solver calls, MILP master invocations/times/nodes and shared auxiliary variable count. Fractions use the existing repaired certificate; no new continuous solve or off-point tangent is introduced.'
protocol['sources']={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(PACKAGE.rglob('*.py'))}
protocol['previous_development']=str(PREVIOUS)
(HERE/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
runner=(PREVIOUS/'study.py').read_text(encoding='utf-8')
runner=runner.replace('        self.second_stage_info = []','''        self.second_stage_info = []
        self.owned_evaluations=[]''')
runner=runner.replace('        second=result.certificate', '''        self.owned_evaluations.append(dict(point=np.asarray(z).tolist(),
            support=list(self.problem.support(z)),lower=result.value,upper=result.upper,
            intercept=result.intercept,gradient=result.gradient.tolist(),
            tau=result.certificate.get("tau"),mu=result.certificate.get("mu"),
            entropy=result.certificate.get("entropy"),margin=result.certificate.get("margin")))
        second=result.certificate''')
runner=runner.replace('        if rule == \'master\':','        if rule in (\'master\',\'ratio\'):')
runner=runner.replace('random_state=seed, **SETTINGS)',
                      'random_state=seed, **{**SETTINGS,"cut_mode":"ratio" if rule=="ratio" else "tangent"})')
runner=runner.replace('        started = perf_counter()', '        started = perf_counter()')
runner=runner.replace('    started = perf_counter()\n    try:', '    started = perf_counter()\n    master_measurements.clear()\n    try:')
runner=runner.replace('second_stage_info=oracle.second_stage_info)',
                      'second_stage_info=oracle.second_stage_info, owned_evaluations=oracle.owned_evaluations, master_measurements=list(master_measurements))')
runner=runner.replace("if rule != 'master' and 'diagnostics' in result:",
                      "if rule not in ('master','ratio') and 'diagnostics' in result:")
runner=runner.replace('    return checks\n', '''    if rule=="ratio":
        owned=result.get("measured",{}).get("owned_evaluations",[])
        evaluated=[r for r in result.get("trace",[]) if r["phase"] in ("initial","integer")]
        # The existing legacy trace uses integer for every queried master point.
        if not evaluated:
            evaluated=[r for r in result.get("trace",[]) if r["phase"]!="master"]
        checks["unique_supports"]=len({tuple(r["support"]) for r in owned})==len(owned)
        checks["one_owned_tangent"]=len(owned)==result.get("visited")==len(evaluated)
        checks["anchor_accuracy"]=all(r["upper"]-r["lower"]<=tolerance/4 for r in owned)
        checks["owned_anchor"]=all(abs(r["intercept"]+np.asarray(r["gradient"])@r["point"]-r["lower"])<=1e-10 for r in owned)
    return checks
''')
runner=runner.replace('outer.GateOracle = MeasuredOracle','''master_measurements=[]
def measured_master(function,kind):
    def wrapper(problem,*args,**kwargs):
        started=perf_counter()
        result=function(problem,*args,**kwargs)
        master_measurements.append(dict(kind=kind,seconds=perf_counter()-started,
            status=int(result.status),nodes=getattr(result,"mip_node_count",None),
            shared_auxiliary_variables=problem.p if kind=="ratio" else 0,
            tangent_count=len(args[0]),fraction_count=len(args[1]) if kind=="ratio" else 0))
        return result
    return wrapper
outer._master=measured_master(outer._master,"tangent")
outer._ratio_master=measured_master(outer._ratio_master,"ratio")
outer.GateOracle = MeasuredOracle''')
runner=runner.replace("                continuous_calls=result.get", '''                master_seconds=sum(r["seconds"] for r in result.get("measured",{}).get("master_measurements",[])),
                master_nodes=sum(r["nodes"] or 0 for r in result.get("measured",{}).get("master_measurements",[])),
                continuous_calls=result.get''')
runner=runner.replace("'masters','continuous_calls'", "'masters','master_seconds','master_nodes','continuous_calls'")
runner=runner.replace("methods=['baseline','screened','core','master']", "methods=['baseline','ratio','master']")
(HERE/'study.py').write_text(runner,encoding='utf-8',newline='\n')
print(HERE)
