from pathlib import Path
import csv,json,math,re,hashlib,collections
import numpy as np
ROOT=Path('D:/OptExp/RubustHW'); TEMP=Path('C:/Users/27512/AppData/Local/Temp/robusthw-report-content-audit-20261010-g3kxkirq'); author=Path('C:/Users/27512/AppData/Local/Temp/robusthw-report-review-5200fc63b46d47c59eaf4689ce669928/authoring')
source={language:(TEMP/name).read_text(encoding='utf-8') for language,name in [('en','experiment_report.tex'),('zh','experiment_report_zh.tex')]}
errors=[]; checks=0

def need(condition,message):
 global checks
 checks+=1
 if not condition: errors.append(message)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def table(label,lang='en'):
 pattern=re.escape('\\begin{table}')+r'.*?'+re.escape('\\end{table}')
 for t in re.findall(pattern,source[lang],re.S):
  if '\\label{'+label+'}' in t: return t
 raise AssertionError('missing table '+label)
def numbers(table_text):
 body=table_text.split('\\midrule',1)[1].split('\\bottomrule',1)[0]
 return re.findall(r'(?<![A-Za-z])-?\d+(?:\.\d+)?',body)
def needrow(label,prefix,numeric):
 t=table(label); need(prefix+numeric in t,label+': '+prefix+numeric)
def boot(values):
 values=np.array(values,float); rng=np.random.default_rng(20261003)
 draws=values[rng.integers(len(values),size=(2000,len(values)))].mean(axis=1)
 lo,hi=np.quantile(draws,[.025,.975]); return values.mean(),lo,hi
def interval(values,digits=3):
 m,lo,hi=boot(values)
 return '\\shortstack{'+f'{m:.{digits}f}'+'\\\\{['+f'{lo:.{digits}f}, {hi:.{digits}f}'+']}}'
def filt(suite,data,method,dimension=None,theta=.25):
 return [r for r in rows if r['suite']==suite and r['dataset']==data and r['method']==method and (dimension is None or int(r['dimension'])==dimension) and abs(float(r['theta'])-theta)<1e-12]
def vals(group,field): return [float(r[field]) for r in group]
with (ROOT/'results/research_20261003/per_trial.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
# Check preserved input hashes and all raw records, independently of the builder.
frozen=json.loads((ROOT/'reports/verification/20261010/next_selection/frozen_inputs.json').read_text())
for name,digest in frozen.items(): need(sha(ROOT/name)==digest,'frozen study input '+name)
record_manifest=json.loads((ROOT/'_archive/report_revision_20261007/record_hashes.json').read_text())
if isinstance(record_manifest,dict) and 'files' in record_manifest: record_manifest=record_manifest['files']
raw_records={}; fields_checked=0
for row in rows:
 path=ROOT/'results/research_20261003'/row['record_file'].replace('\\','/')
 record=json.loads(path.read_text()); raw_records[row['record_file']]=record
 need(record['selection']['status']==row['selection_status'] and tuple(record['selection']['support'])==tuple(json.loads(row['support'])),'raw identity '+row['record_file'])
 for name,value in record.get('metrics',{}).items():
  if name in row and row[name]!='':
   need(math.isclose(float(row[name]),float(value),abs_tol=1e-10,rel_tol=1e-9),'raw metric '+row['record_file']+' '+name); fields_checked+=1
 for name in ('lower','upper','gap','seconds','visited','oracle_calls'):
  if name in record['selection']:
   need(math.isclose(float(row['selection_'+name]),float(record['selection'][name]),abs_tol=1e-10,rel_tol=1e-9),'raw selection '+row['record_file']+' '+name); fields_checked+=1
# Check all old raw hashes regardless of the manifest path convention.
checked_hashes=0
for name,digest in record_manifest.items():
 path=Path(name)
 candidates=[ROOT/path,ROOT/'results/research_20261003'/path,ROOT/'results/research_20261003'/str(path).replace('\\','/')]
 candidate=next((p for p in candidates if p.is_file()),None)
 need(candidate is not None,'old hash path '+name)
 if candidate is not None: need(sha(candidate)==digest,'old raw unchanged '+name); checked_hashes+=1
# Every numerical table is bilingual, excluding procedure prose in selector table.
labels=re.findall(r'\\label\{(tab:[^}]+)\}',source['en'])
need(set(labels)==set(re.findall(r'\\label\{(tab:[^}]+)\}',source['zh'])),'bilingual table labels')
for label in labels:
 if label!='tab:selectors': need(numbers(table(label,'en'))==numbers(table(label,'zh')),'bilingual table numeric identity '+label)
need('\\appendix' not in source['en'] and '\\appendix' not in source['zh'],'no report appendix')
# Historical computation and paired supplement table entries.
for data,label in [('mean','Mean'),('mixture','Mixture')]:
 for method,title in [('box','Box'),('triangle','Triangle'),('risk_no_root','Risk, no root'),('risk','Global risk')]:
  g=filt('computational',data,method,12); solved=sum(r['selection_status']=='optimal' for r in g)
  numeric=f'{solved}/{len(g)} & '+interval(vals(g,'selection_seconds'),2)+f" & {np.mean(vals(g,'selection_visited')):.1f}"
  needrow('tab:reference-computation',f'{label} & {title} & ',numeric)
 for dim in (8,12,20):
  g0=filt('ratio',data,'risk',dim); g1=filt('ratio',data,'risk_ratio',dim)
  byseed0={int(r['seed']):r for r in g0}; byseed1={int(r['seed']):r for r in g1}
  dif=[float(byseed1[s]['selection_seconds'])-float(byseed0[s]['selection_seconds']) for s in sorted(byseed0)]
  numeric=f'{dim} & {sum(r["selection_status"]=="optimal" for r in g0)}/{len(g0)} & {sum(r["selection_status"]=="optimal" for r in g1)}/{len(g1)} & '+interval(dif,2)
  needrow('tab:reference-ratio',f'{label} & ',numeric)
# Native/common power, independent head and radius table entries.
for data,title in [('mean','Mean'),('variance','Variance'),('dependence','Dependence'),('mixture','Mixture')]:
 g=filt('statistical',data,'wang'); numeric='30 & '+' & '.join(interval(vals(g,m)) for m in ('power','native_additive_power','native_additive_null_rejection'))
 needrow('tab:native',title+' & ',numeric)
 g=filt('statistical',data,'risk'); count=sum(float(r['robust_randomized_balanced_error'])>float(r['head_robust_upper'])/2+1e-8 for r in g)
 means=[np.mean(vals(g,'head_robust_upper'))/2,np.mean(vals(g,'robust_randomized_balanced_error')),np.mean(vals(g,'robust_worst_stress_randomized_balanced_error'))]
 needrow('tab:head',title+' & ',f'{len(g)} & '+' & '.join(f'{v:.3f}' for v in means)+f' & {count}/{len(g)}')
 if data in ('mean','mixture'):
  for theta in (.1,.25,.5):
   g=filt('sensitivity',data,'risk',theta=theta)
   needrow('tab:radius',title+' & ',f'{theta:.2f} & {len(g)} & '+' & '.join(interval(vals(g,m)) for m in ('power','robust_randomized_balanced_error','robust_worst_stress_randomized_balanced_error')))
for data,title in [('wdbc','WDBC'),('ionosphere','Ionosphere')]:
 for method,mt in [('risk','Global risk'),('greedy','Robust greedy'),('mmd','MMD screening'),('wang','Additive MMD'),('l1','L1 logistic'),('random','Random')]:
  g=filt('real',data,method)
  cells=[]
  for metric in ('robust_randomized_balanced_error','svm_balanced_error','robust_auc'):
   v=np.array(vals(g,metric)); cells.append(f'{v.mean():.3f}'+'\\(\\pm\\)'+f'{v.std(ddof=1):.3f}')
  needrow('tab:real',f'{title} & {mt} & ',' & '.join(cells))
for suite,name in [('statistical','Synthetic'),('computational','Optimization'),('real','Real data'),('sensitivity','Radius'),('ratio','Fraction supplement')]:
 g=[r for r in rows if r['suite']==suite]; counts=collections.Counter(r['selection_status'] for r in g); failures=sum(counts[k] for k in ('master_failure','oracle_failure','failure','exception','numerical_inconsistency'))
 needrow('tab:outcomes',name+' & ',f'{len(g)} & {counts["optimal"]} & {counts["time_limit"]} & {failures} & {len(g)-counts["optimal"]-counts["time_limit"]-failures}')
# Development comparisons are recomputed from every original raw method outcome.
phase_names={1:{"lcb":"Mean minus SD","ei":"Expected improvement","guard":"Mean minus SD, master priority","ei_guard":"Improvement, master priority","master":"Tangent master at every step"},2:{"baseline":"Original GP search","screened":"Ranked proposals","core":"Dual-face tangent","master":"Tangent master at every step"},3:{"baseline":"Original GP search","ratio":"Fractions at every step","master":"Tangent master at every step"},4:{"baseline":"Original GP search","curvature":"Stored fractions","ratio":"Fractions at every step","master":"Tangent master at every step"}}
for phase,names in phase_names.items():
 folder=ROOT/f"reports/verification/20261010/next_selection/development/phase{phase}/development"
 stage_records=[json.loads(p.read_text()) for p in sorted((folder/"raw").glob("*.json"))]
 need(len(stage_records)==26,"phase development count "+str(phase))
 for method,title in names.items():
  result_list=[r["methods"][method] for r in stage_records]
  T=sum(r["visited"] for r in result_list); masters=sum(r["diagnostics"].get("master_calls",sum(t["phase"]=="master" for t in r["trace"])) for r in result_list)
  continuous="---" if phase==1 else str(sum(r["measured"]["continuous_calls"] for r in result_list))
  total=sum(r["seconds"] for r in result_list); optimal=sum(r["status"]=="optimal" for r in result_list)
  needrow("tab:search-development",f"{phase} & {title} & ",f"{T} & {continuous} & {masters} & {total:.3f} & {optimal}/26")

# New confirmed tables: sum within-case medians, including master cost separately.
for stage,label in [('development','tab:search-integration'),('confirmation','tab:search-confirmation')]:
 folder=ROOT/'reports/verification/20261010/next_selection/current'/stage
 raw=[json.loads(p.read_text()) for p in sorted((folder/'raw').glob('*.json'))]
 for method,title in [('master','Tangent master at every step'),('baseline','Original GP search'),('curvature','Stored fractions')]:
  totals={k:0. for k in ('T','continuous','masters','mastertime','time')}; optimum=runs=0
  for record in raw:
   results=[rep['methods'][method] for rep in record['repeats']]; runs+=len(results); optimum+=sum(v['status']=='optimal' for v in results)
   totals['T']+=np.median([v['visited'] for v in results]); totals['continuous']+=np.median([v['measured']['continuous_calls'] for v in results]); totals['masters']+=np.median([len(v['measured']['master_measurements']) for v in results]); totals['mastertime']+=np.median([sum(m['seconds'] for m in v['measured']['master_measurements']) for v in results]); totals['time']+=np.median([v['seconds'] for v in results])
  needrow(label,title+' & ',f'{totals["T"]:g} & {totals["continuous"]:g} & {totals["masters"]:g} & {totals["mastertime"]:.3f} & {totals["time"]:.3f} & {optimum}/{runs}')
# Language-consistent equations and explicitly correct units/definitions.
for environment in ['equation']:
 pattern=r'\\begin\{'+environment+r'\}.*?\\end\{'+environment+r'\}'
 b=chr(92); translated_text_pattern=b+b+"text"+b+"{[^}]*"+b+"}"
 math_only=lambda v:"".join(re.sub(translated_text_pattern,"",v).split())
 need([math_only(v) for v in re.findall(pattern,source["en"],re.S)]==[math_only(v) for v in re.findall(pattern,source["zh"],re.S)],"bilingual equation identity after translated text")
selection_hours=sum(float(r['selection_seconds']) for r in rows)/3600
total_hours=sum(float(r.get('total_seconds',0)) for r in raw_records.values())/3600
need(f'{selection_hours:.2f} and {total_hours:.2f} hours' in source['en'],'charged historical times')
record={'report_sha256':{name:sha(TEMP/name) for name in ['experiment_report.tex','experiment_report_zh.tex']},'checks_count':checks,'all_pass':not errors,'failures':errors,'historical_raw_records':len(raw_records),'historical_old_hashes_checked':checked_hashes,'raw_numeric_fields_checked':fields_checked,'labels':labels,'charged_selection_hours':selection_hours,'charged_end_to_end_hours':total_hours,'current_reports_changed_since_snapshot':{name:sha(author/name)!=sha(TEMP/name) for name in ['experiment_report.tex','experiment_report_zh.tex']},'scope':'read-only numerical/table/code/config audit of report snapshot; current authority is report_design TEMP, not canonical writer'}
(TEMP/'report_numerical_content_audit.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
