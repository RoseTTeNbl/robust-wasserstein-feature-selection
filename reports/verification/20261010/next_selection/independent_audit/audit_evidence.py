from pathlib import Path
import hashlib,itertools,json,math,sys
import numpy as np
ROOT=Path('D:/OptExp/RubustHW')
TEMP=Path('C:/Users/27512/AppData/Local/Temp/robusthw-final-independent-audit-20261010-eh47j6a8')
BASE=ROOT/'reports/verification/20261010/next_selection/current'
errors=[]; stats={}; checks_count=0; hashes_at_start={}

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path): return json.loads(path.read_text(encoding='utf-8'))
def need(condition,message):
 global checks_count
 checks_count+=1
 if not condition: errors.append(message)
def close(a,b,tol=1e-8): return bool(np.allclose(a,b,atol=tol,rtol=0))

def entropy(plans,loss):
 a,b=[g.sum(axis=0) for g in plans]
 if loss=='overlap': return float(np.minimum(a,b).sum())
 total=a+b; h=0.
 for v in (a,b):
  positive=v>0
  h-=float(np.sum(v[positive]*np.log(v[positive]/total[positive])))
 return h

for phase in ('development','confirmation'):
 folder=BASE/phase; proto=load(folder/'protocol.json'); seal=load(folder/'seal.json'); summary=load(folder/'summary.json')
 expected_cases=26 if phase=='development' else 16
 repeats=1 if phase=='development' else 3
 tolerance=proto['settings']['tolerance']; methods=proto['methods']
 need(methods==['master','baseline','curvature'],phase+' methods')
 need(len(proto['cases'])==expected_cases,phase+' expected case count')
 need(proto['timing_repeats']==repeats,phase+' timing repeats')
 need(proto['sources']==seal['sources'],phase+' protocol/seal source set')
 files={p.relative_to(folder).as_posix():sha(p) for p in [folder/'protocol.json',folder/'summary.json',*sorted((folder/'raw').glob('*.json'))]}
 need(files==seal['files'],phase+' sealed raw/protocol/summary hashes')
 for name,digest in proto['sources'].items():
  need(sha(ROOT/name)==digest,phase+' canonical freeze '+name)
  need(sha(folder/'source'/name)==digest,phase+' snapshot hash '+name)
  hashes_at_start[name]=sha(ROOT/name)
 ids={case['id']:i for i,case in enumerate(proto['cases'])}
 need({p.stem for p in (folder/'raw').glob('*.json')}==set(ids),phase+' all and only declared raw cases')
 summary_map={(r['case'],r['repeat'],r['method']):r for r in summary['rows']}
 need(len(summary_map)==expected_cases*repeats*3==len(summary['rows']),phase+' summary unique rows')
 counts={m:dict(outcomes=0,complete_evaluations=0,physical_calls=0,continuous_calls=0,masters=0) for m in methods}
 extrema=dict(cross_certificate_violation=-np.inf,dual_row_residual=0.,dual_budget_residual=0.,dual_coordinate_residual=0.,dual_normalization_residual=0.,final_primal_violation=0.,reference_gap=0.,result_gap=0.,cut_gradient_delta=0.,fraction_anchor_delta=0.)
 case_comparisons=[]; data_hashes=set(); same_outcome_repeats=0
 for path in sorted((folder/'raw').glob('*.json')):
  raw=load(path); case=raw['case']; identifier=case['id']; index=ids[identifier]
  need(case==proto['cases'][index],identifier+' declared case data')
  data=raw['data']; x=[np.asarray(data[k],float) for k in ('x0','x1')]; q=[np.asarray(data[k],float) for k in ('q0','q1')]
  data_bytes=np.concatenate([np.asarray(data[k],float).ravel() for k in ('x0','x1','q0','q1')]).tobytes()
  need(hashlib.sha256(data_bytes).hexdigest()==case['data_sha256'],identifier+' data digest'); data_hashes.add(case['data_sha256'])
  D,budget=case['dimension'],case['budget']; theta=np.asarray(case['theta']); B=1. if case['loss']=='overlap' else 2*math.log(2)
  pool=np.vstack(x); delta=[np.abs(samples[:,None]-pool[None]) for samples in x]
  M=np.max(np.array([mass@d.max(axis=1) for mass,d in zip(q,delta)])/theta[:,None],axis=0)
  support_set=set(itertools.combinations(range(D),budget)); ref=raw['reference']
  ref_map={tuple(t['support']):t for t in ref['trace']}
  need(ref['status']=='optimal' and len(ref_map)==len(support_set)==len(ref['trace'])==ref['visited'],identifier+' complete original enumeration')
  need(set(ref_map)==support_set,identifier+' enumeration coverage')
  need(close(ref['lower'],max(0,min(t['lower'] for t in ref['trace'])-1e-9)),identifier+' reference lower aggregation')
  need(close(ref['upper'],min(B,min(t['upper'] for t in ref['trace']))),identifier+' reference upper aggregation')
  extrema['reference_gap']=max(extrema['reference_gap'],ref['upper']-ref['lower'])
  masks=np.array([np.isin(np.arange(D),s).astype(float) for s in sorted(support_set)])
  support_uppers=np.array([ref_map[s]['upper'] for s in sorted(support_set)])
  need(len(raw['repeats'])==repeats,identifier+' repeats present')
  repeat_signature=[]
  for rep in raw['repeats']:
   r=rep['repeat']; offset=(index+r)%3; expected_order=methods[offset:]+methods[:offset]
   need(rep['method_order']==expected_order,identifier+f' rotated execution order r{r}')
   need(set(rep['methods'])==set(methods),identifier+f' method completeness r{r}')
   signatures=[]
   for method,result in rep['methods'].items():
    label=identifier+f' r{r} {method}'; measured=result['measured']; diag=result['diagnostics']; own=measured['owned_evaluations']
    c=counts[method]; c['outcomes']+=1; c['complete_evaluations']+=result['visited']; c['physical_calls']+=result['oracle_calls']; c['continuous_calls']+=measured['continuous_calls']; c['masters']+=len(measured['master_measurements'])
    need(result['status']=='optimal' and result['upper']-result['lower']<=tolerance,label+' numerical convergence')
    extrema['result_gap']=max(extrema['result_gap'],result['upper']-result['lower'])
    need(result['upper']-ref['lower']<=tolerance and result['lower']<=ref['upper']+tolerance and result['upper']>=ref['lower']-tolerance,label+' exhaustive consistency')
    need(measured['repricings']==0 and measured['binary_only'],label+' no transfer / binary only')
    need(result['oracle_calls']==measured['full_calls']==len(own),label+' physical-call accounting')
    need(measured['continuous_calls']==2*measured['full_calls'],label+' each full solve primal and dual')
    need(all(np.isin(o['point'],(0,1)).all() and np.sum(o['point'])==budget for o in own),label+' full-budget binary measurements')
    # Check every measured repaired dual directly against the actual data.
    for o in own:
     cert=o['certificate']; z=np.asarray(o['point']); gamma=[np.asarray(g,float) for g in cert['gamma']]; beta=np.asarray(cert['beta']); mu=np.asarray(cert['mu']); tau=cert['tau']; h=entropy(gamma,case['loss'])
     cost=np.array([np.einsum('im,imj->j',g,d) for g,d in zip(gamma,delta)])
     row_res=max(float(np.max(np.abs(g.sum(axis=1)-tau*mass))) for g,mass in zip(gamma,q))
     budget_res=max(0.,float(np.max(beta.sum(axis=1)-tau*theta)))
     coord_res=max(0.,float(np.max(cost-beta-theta[:,None]*mu)))
     norm_res=abs(tau+mu@z-1.)
     extrema['dual_row_residual']=max(extrema['dual_row_residual'],row_res); extrema['dual_budget_residual']=max(extrema['dual_budget_residual'],budget_res); extrema['dual_coordinate_residual']=max(extrema['dual_coordinate_residual'],coord_res); extrema['dual_normalization_residual']=max(extrema['dual_normalization_residual'],norm_res)
     need(min(g.min() for g in gamma)>=0 and tau>0 and mu.min()>=0 and beta.min()>=0,label+' nonnegative dual data')
     need(max(row_res,budget_res,coord_res,norm_res)<=1e-7,label+' independently recomputed feasible dual residuals')
     need(close(cost,cert['costs']) and close(h,cert['entropy']),label+' own repaired entropy and costs')
     gradient=np.asarray(o['gradient']); rawgrad=-h*mu
     extrema['cut_gradient_delta']=max(extrema['cut_gradient_delta'],float(np.max(np.abs(rawgrad-gradient))))
     need(close(rawgrad,gradient) and close(o['lower'],h-cert['margin']) and close(o['intercept'],o['lower']-gradient@z),label+' raw gradient / shifted tangent')
     need(np.all(mu<=tau*M+1e-7),label+' source-weighted coordinate price bounds')
    # Recompute the returned test's risk and finite loss allowances.
    test=result['test_certificate']; a=[np.asarray(v) for v in test['potentials']]; lam=np.asarray(test['multipliers']); selected=tuple(test['support']); mask=np.isin(np.arange(D),selected)
    risk=sum(float(mass@allow) for mass,allow in zip(q,a))+float(theta@lam)
    need(selected==tuple(result['support']) and close(risk,result['upper']) and close(risk,test['primal_objective']),label+' genuine final test and risk')
    E=[np.min(allow[:,None]+price*(d*mask).sum(axis=2),axis=0) for allow,price,d in zip(a,lam,delta)]
    violation=max(0.,float(np.max(1-E[0]-E[1]))) if case['loss']=='overlap' else max(0.,float(np.max(np.exp(-E[0])+np.exp(-E[1])-1)))
    extrema['final_primal_violation']=max(extrema['final_primal_violation'],violation)
    need(violation<=1e-7 and lam.min()>=0 and min(v.min() for v in a)>=0,label+' returned finite whole-space allowance')
    if method!='master':
     rows=diag['evaluations']; trace=result['trace']
     need(diag['cut_count']==diag['accurate_support_evaluations']==result['visited']==len(rows),label+' one accurate owned tangent per completion')
     need(len({tuple(v['support']) for v in rows})==len(rows),label+' no repeat completed subset')
     need(diag['fraction_count']==(len(rows) if method=='curvature' else 0),label+' one optional owned fraction per completion')
     need(diag['transferred_tests']==diag['predicted_upper_updates']==diag['fractional_root_calls']==0,label+' actual-only bounds')
     need(diag['search']==('full_inner_fraction_v4' if method=='curvature' else 'full_inner_uncertainty_v3'),label+' version identity')
     need(diag['proposal_count']==8 and diag['certify_every']==3,label+' frozen simple search controls')
     need(close(diag['metric_weights'],M),label+' same source-weighted M')
     need(close(result['upper'],min(B,*[v['actual_upper'] for v in rows])),label+' upper only genuine evaluated risks')
     for row in rows:
      z=np.asarray(row['point']); g=np.asarray(row['gradient']); intercept=row['intercept']; matches=[o for o in own if o['point']==row['point'] and o['intercept']==intercept and o['gradient']==row['gradient']]
      need(bool(matches),label+' tangent from same actual complete solve')
      need(row['accurate'] and row['anchor_gap']<=tolerance/4 and close(row['anchor_lower'],intercept+g@z) and close(row['anchor_gap'],row['actual_upper']-row['anchor_lower']),label+' accurate anchor accounting')
      L=intercept+masks@g; worst=float(np.max(L-support_uppers)); extrema['cross_certificate_violation']=max(extrema['cross_certificate_violation'],worst)
      need(worst<=1e-7,label+' affine global validity against all enumerated support upper bounds')
      fraction=row['owned_fraction']
      if method=='curvature':
       need(fraction is not None,label+' saved complete fraction')
       cert=matches[0]['certificate']; n,tau,mu=fraction['numerator'],fraction['tau'],np.asarray(fraction['mu'])
       need(tau==cert['tau'] and close(mu,cert['mu'],0) and n==max(0,cert['entropy']-cert['margin']),label+' fraction and tangent same retained certificate including margin')
       anchor=n/(tau+mu@z); extrema['fraction_anchor_delta']=max(extrema['fraction_anchor_delta'],abs(anchor-row['anchor_lower']))
       need(abs(anchor-row['anchor_lower'])<=1e-7,label+' fraction anchor agrees')
       F=n/(tau+masks@mu); worst=float(np.max(F-support_uppers)); extrema['cross_certificate_violation']=max(extrema['cross_certificate_violation'],worst)
       need(worst<=1e-7,label+' complete fraction global validity against all enumerated support upper bounds')
      else: need(fraction is None,label+' frozen tangent control')
     # Reconstruct model values and incumbent in actual chronological trace.
     previous=[]; incumbent=B; lower=0.
     rows_by_support={tuple(v['support']):v for v in rows}
     for t in trace:
      if t['phase']=='acquisition':
       z=np.isin(np.arange(D),t['support']).astype(float); vals=[0.]
       for v in previous:
        vals.append(v['intercept']+np.asarray(v['gradient'])@z)
        if v['owned_fraction']:
         f=v['owned_fraction']; vals.append(f['numerator']/(f['tau']+np.asarray(f['mu'])@z))
       need(close(max(vals),t['model_lower']),label+' actual historical model at selection')
       need(close(t['upper'],incumbent) and t['model_lower']<=incumbent-tolerance/2 and close(t['score'],t['mean']-t['std']),label+' eligibility and prediction ranking only')
      if t['phase'] in ('initial','adaptive'):
       v=rows_by_support[tuple(t['support'])]; previous.append(v); incumbent=min(incumbent,v['actual_upper'])
       need(t['cut_added'] and close(t['upper'],incumbent),label+' one post-solve update')
      need(t['lower']>=lower-1e-12,label+' running global lower monotonicity'); lower=t['lower']
     need(len(previous)==len(rows),label+' all completed evaluations in chronological trace')
    # Check summary values independently rather than trusting precomputed booleans.
    sr=summary_map[(identifier,r,method)]; master_seconds=sum(v['seconds'] for v in measured['master_measurements'])
    values=dict(status=result['status'],completed_evaluations=result['visited'],physical_calls=result['oracle_calls'],seconds=result['seconds'],masters=len(measured['master_measurements']),master_seconds=master_seconds,continuous_calls=measured['continuous_calls'],lower=result['lower'],upper=result['upper'],gap=result['gap'])
    need(all(sr[k]==value for k,value in values.items()),label+' raw/summary numerical identity')
    need(sr['all_checks'] and all(rep['checks'][method].values()),label+' producer check reports retained')
    signatures.append((method,result['visited'],result['oracle_calls'],tuple(result['support'])))
   repeat_signature.append(sorted(signatures))
  if repeats==3:
   need(all(v==repeat_signature[0] for v in repeat_signature),identifier+' deterministic counts/support across timing repeats'); same_outcome_repeats+=1
  case_comparisons.append(dict(case=identifier,evaluations={m:[rep['methods'][m]['visited'] for rep in raw['repeats']] for m in methods},median_seconds={m:float(np.median([rep['methods'][m]['seconds'] for rep in raw['repeats']])) for m in methods}))
 need(summary['case_count']==expected_cases and summary['timing_repeats']==repeats,phase+' summary declared counts')
 stats[phase]=dict(case_count=expected_cases,timing_repeats=repeats,outcome_count=expected_cases*repeats*3,sealed_files=len(files),sources=len(proto['sources']),unique_datasets=len(data_hashes),counts=counts,extrema=extrema,all_confirmation_repeats_same_outcomes=same_outcome_repeats,paired_comparisons=case_comparisons)
for name,digest in hashes_at_start.items(): need(sha(ROOT/name)==digest,'source remained frozen through independent audit '+name)
record=dict(audit_kind='independent read-only saved-evidence reconstruction; no optimization reruns',checks_count=checks_count,failures=errors,all_pass=not errors,stages=stats,canonical_sources=hashes_at_start)
(TEMP/'independent_saved_evidence_audit.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(dict(all_pass=not errors,checks_count=checks_count,failures=errors[:20],stage_totals={p:{k:v for k,v in s.items() if k in ('case_count','timing_repeats','outcome_count','sealed_files','sources','unique_datasets','counts','extrema')} for p,s in stats.items()},audit_path=str(TEMP/'independent_saved_evidence_audit.json')),indent=2))
sys.exit(bool(errors))
