from pathlib import Path
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
PREVIOUS=Path('C:/Users/27512/AppData/Local/Temp/risk-acquisition-study-phase3-20261010-e47a618d')
PACKAGE=HERE/'source/robusthw'
assert not PACKAGE.exists()
shutil.copytree(PREVIOUS/'source/robusthw',PACKAGE,ignore=shutil.ignore_patterns('__pycache__'))
history=(PACKAGE/'history.py').read_text(encoding='utf-8')
history=history.replace('        self.cuts, self.points, self.anchor_gaps = [], [], []',
                        '        self.cuts, self.points, self.anchor_gaps = [], [], []\n        self.fractions = []')
history=history.replace('def add(self, z, *, test, intercept, gradient, accurate):',
                        'def add(self, z, *, test, intercept, gradient, accurate, fraction=None):')
history=history.replace('        self.cuts.append((float(intercept), gradient.copy()))', '''        if fraction is not None:
            numerator,tau,mu=fraction
            if not (numerator>=0 and tau>0 and np.isfinite(numerator) and np.isfinite(tau)):
                raise ArithmeticError("invalid owned safe fraction")
            mu=np.asarray(mu,float)
            if mu.shape!=(self.problem.p,) or not np.isfinite(mu).all() or np.any(mu<0):
                raise ArithmeticError("invalid fraction correction")
            anchor=numerator/(tau+mu@z)
            if abs(anchor-own_lower)>1e-7*max(1.,self.problem.B):
                raise ArithmeticError("fraction and owned tangent do not share the anchor certificate")
            self.fractions.append((float(numerator),float(tau),mu.copy()))
        self.cuts.append((float(intercept), gradient.copy()))''')
old='''        return float(max([0., *[intercept + float(gradient @ z)
                               for intercept, gradient in self.cuts]]))'''
new='''        return float(max([0., *[intercept + float(gradient @ z)
                               for intercept, gradient in self.cuts],
                          *[numerator/(tau+float(mu@z))
                            for numerator,tau,mu in self.fractions]]))'''
assert old in history
history=history.replace(old,new)
(PACKAGE/'fraction_history.py').write_text(history,encoding='utf-8',newline='\n')
adaptive=(PACKAGE/'adaptive.py').read_text(encoding='utf-8')
adaptive=adaptive.replace('from .history import EvaluationHistory, proposal_pool',
                          'from .fraction_history import EvaluationHistory, proposal_pool')
adaptive=adaptive.replace('from .solver import SelectionResult, _master',
                          'from .solver import SelectionResult, _ratio_master')
adaptive=adaptive.replace('        best_cut = None', '        best_cut = best_fraction = None')
adaptive=adaptive.replace('                best_cut, best_cut_value = (float(result.intercept), gradient.copy()), owned_lower', '''                best_cut, best_cut_value = (float(result.intercept), gradient.copy()), owned_lower
                cert=result.certificate
                tau=float(cert.get("tau",0.))
                numerator=max(0.,float(cert.get("entropy",0.))-float(cert.get("margin",0.)))
                best_fraction=(numerator,tau,np.asarray(cert["mu"],float).copy()) if tau>0 and numerator>0 else None''')
adaptive=adaptive.replace('gradient=best_cut[1], accurate=accurate)',
                          'gradient=best_cut[1], accurate=accurate, fraction=best_fraction)')
adaptive=adaptive.replace('result = _master(p, history.cuts, remaining(), integer=True)',
                          'result = _ratio_master(p, history.cuts, history.fractions, remaining(), upper)')
adaptive=adaptive.replace('cuts_added=len(history.cuts), cut_count=len(history.cuts), evaluations=history.records,',
                          'cuts_added=len(history.cuts), cut_count=len(history.cuts), fraction_count=len(history.fractions), evaluations=history.records,')
(PACKAGE/'adaptive_curvature.py').write_text(adaptive,encoding='utf-8',newline='\n')
protocol=json.loads((PREVIOUS/'protocol.json').read_text(encoding='utf-8'))
protocol['scope']='Last development variant: retain completed dual fractions in the existing period-3 v3 adaptive loop; no new score, pool, hyperparameter or extra continuous solve'
protocol['rules']={'baseline':'Original v3 on frozen simplified core',
                   'curvature':'Original v3 acquisition/pool/period3; certified lower=max(affines,safe owned fractions); shared-product ratio master only periodic or no eligible proposal',
                   'ratio':'Same-core ratio/master search with every-step master refresh',
                   'master':'Same-core tangent/master search'}
protocol['selection']='This is the final development variant. Lock one simple scheme after complete development comparison; confirmation remains unseen and is never used to retune.'
protocol['fraction_ownership']='The final retained safe fraction comes from the exact same oracle result as the final best owned tangent. Refinement may retain a different actually evaluated primal at that same binary support. No inherited solution or off-point tangent.'
protocol['sources']={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(PACKAGE.rglob('*.py'))}
protocol['previous_development']=str(PREVIOUS)
(HERE/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
runner=(PREVIOUS/'study.py').read_text(encoding='utf-8')
runner=runner.replace("methods=['baseline','ratio','master']", "methods=['baseline','curvature','ratio','master']")
runner=runner.replace("    return checks\n", '''    if rule=="curvature":
        owned=result.get("measured",{}).get("owned_evaluations",[])
        rows=result["diagnostics"]["evaluations"]
        checks["fraction_same_result_as_cut"]=all(any(
            r["support"]==row["support"] and r["intercept"]==row["intercept"]
            and r["gradient"]==row["gradient"] for r in owned) for row in rows)
    return checks
''')
(HERE/'study.py').write_text(runner,encoding='utf-8',newline='\n')
print(HERE)
