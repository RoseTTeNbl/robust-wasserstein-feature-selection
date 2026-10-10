"""Predeclared paired development/confirmation study on an immutable snapshot."""
import argparse
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
HERE = Path(__file__).resolve().parent
ROOT = Path('D:/OptExp/RubustHW')
sys.path.insert(0, str(HERE/'source'))
import numpy as np
from threadpoolctl import threadpool_limits
from robusthw.oracle import GateOracle
from robusthw.problem import Problem, json_value
import robusthw.solver as outer
import robusthw.oracle as inner


class MeasuredOracle(GateOracle):
    last = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        MeasuredOracle.last = self
        self.full_calls = self.repricings = 0
        self.binary_only = True
        self.continuous_calls = self.second_stage_calls = 0
        self.second_stage_info = []

    def solve(self, z, **kwargs):
        self.full_calls += 1
        self.binary_only &= bool(np.all(np.isin(z, [0, 1])))
        result=super().solve(z, **kwargs)
        second=result.certificate.get("dual",{}).get("core_point")
        if second is not None:
            self.second_stage_calls += 1
            self.second_stage_info.append(second)
        return result

    def _cvx_solve(self, *args, **kwargs):
        deadline=args[1] if len(args)>1 else kwargs.get("deadline")
        if deadline is None or deadline>perf_counter():
            self.continuous_calls += 1
        return super()._cvx_solve(*args, **kwargs)

    def reprice(self, *args, **kwargs):
        self.repricings += 1
        raise AssertionError('solution transfer is forbidden')


actual_lp=inner._lp
def counted_lp(*args,**kwargs):
    deadline=kwargs.get("deadline")
    if deadline is None or deadline>perf_counter():
        MeasuredOracle.last.continuous_calls += 1
    return actual_lp(*args,**kwargs)
inner._lp=counted_lp
outer.GateOracle = MeasuredOracle
PROTOCOL = json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
SETTINGS = PROTOCOL['settings']


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    def finite(value):
        if isinstance(value,dict):
            return {key:finite(item) for key,item in value.items()}
        if isinstance(value,list):
            return [finite(item) for item in value]
        if isinstance(value,float) and not np.isfinite(value):
            return None
        return value
    path.write_text(json.dumps(finite(json_value(value)), indent=2, allow_nan=False), encoding='utf-8')


def run(problem, rule, warm, seed):
    started = perf_counter()
    try:
        if rule == 'master':
            result = outer.solve(problem, search='master', warm_support=warm,
                                 random_state=seed, **SETTINGS)
        else:
            module = importlib.import_module('robusthw.adaptive_'+rule)
            result = module.solve_adaptive(problem, oracle_class=MeasuredOracle,
                        warm_support=warm, random_state=seed, **{**SETTINGS,"core_point":rule=="core"})
        answer = result.to_dict()
        oracle = MeasuredOracle.last
        answer['measured'] = dict(binary_only=oracle.binary_only,
                                full_calls=oracle.full_calls, repricings=oracle.repricings,
                                external_seconds=perf_counter()-started,
                                continuous_calls=oracle.continuous_calls,
                                second_stage_calls=oracle.second_stage_calls,
                                second_stage_info=oracle.second_stage_info)
        return answer
    except Exception as error:
        return dict(status='exception', message=str(error), seconds=perf_counter()-started)


def invariant_check(result, reference, rule):
    tolerance = SETTINGS['tolerance']
    checks = dict(optimal=result['status']=='optimal',
                  reference_consistent=result.get('lower', np.inf) <= reference['upper']+tolerance
                   and result.get('upper', -np.inf) >= reference['lower']-tolerance,
                  reference_accuracy=result.get('upper', np.inf)-reference['lower'] <= tolerance,
                  binary_only=result.get('measured',{}).get('binary_only',False),
                  no_repricing=result.get('measured',{}).get('repricings',1)==0)
    if rule != 'master' and 'diagnostics' in result:
        diagnostics = result['diagnostics']
        rows = diagnostics['evaluations']
        acquisitions = [r for r in result['trace'] if r['phase']=='acquisition']
        checks['one_owned_cut'] = diagnostics['cut_count']==result['visited']==len(rows)
        checks['unique_supports'] = len({tuple(r['support']) for r in rows})==len(rows)
        checks['anchor_accuracy'] = all(r['anchor_gap'] <= tolerance/4 for r in rows)
        checks['cut_filter'] = all(r['model_lower']<=r['upper']-tolerance/2 for r in acquisitions)
        running_upper = problem_bound = 1. if reference['loss']=='overlap' else 2*np.log(2)
        actual_only = True
        for row in result['trace']:
            if row['phase'] in ('initial','adaptive'):
                running_upper=min(running_upper,row['oracle_upper'])
            actual_only &= abs(row['upper']-running_upper)<=1e-12
        checks['actual_only_upper'] = actual_only and result['upper']<=problem_bound
    return checks


def method_metrics(record, rule):
    result = record['methods'][rule]
    diagnostics = result.get('diagnostics',{})
    return dict(case=record['case']['id'], method=rule, status=result['status'],
                completed_evaluations=result.get('visited'), physical_calls=result.get('oracle_calls'),
                seconds=result.get('seconds'),
                masters=diagnostics.get('master_calls',sum(r['phase']=='master' for r in result.get('trace',[]))),
                continuous_calls=result.get("measured",{}).get("continuous_calls"),
                second_stage_calls=result.get("measured",{}).get("second_stage_calls"),
                all_checks=all(record['checks'][rule].values()))


def summarize(records, methods):
    rows=[method_metrics(record,rule) for record in records for rule in methods]
    totals={}
    for rule in methods:
        subset=[r for r in rows if r['method']==rule]
        totals[rule]=dict(cases=len(subset), all_pass=all(r['all_checks'] for r in subset),
                         statuses={s:sum(r['status']==s for r in subset)
                                   for s in sorted({r['status'] for r in subset})})
        for field in ('completed_evaluations','physical_calls','seconds','masters','continuous_calls','second_stage_calls'):
            finite=[r[field] for r in subset if r[field] is not None]
            totals[rule][field]=dict(total=float(sum(finite)), median=float(np.median(finite)))
        baseline=[method_metrics(record,'baseline') for record in records]
        if rule!='baseline':
            pairs=list(zip(subset,baseline))
            totals[rule]['pairs_vs_baseline']=dict(
                fewer=sum(a['completed_evaluations']<b['completed_evaluations'] for a,b in pairs),
                equal=sum(a['completed_evaluations']==b['completed_evaluations'] for a,b in pairs),
                more=sum(a['completed_evaluations']>b['completed_evaluations'] for a,b in pairs),
                faster=sum(a['seconds']<b['seconds'] for a,b in pairs))
    return dict(totals=totals, rows=rows)


def development_cases():
    for path in sorted((ROOT/'reports/verification/20261010/full_inner_uncertainty/raw').glob('*.json')):
        original=json.loads(path.read_text(encoding='utf-8'))
        case=original['case']
        data=original['generated_data']
        reference={**original['methods']['enumeration'],'loss':case['loss']}
        yield case,data,reference


def confirmation_cases():
    for loss,(dimension,budget),seed,theta in itertools.product(
            ('overlap','logistic'),((7,2),(10,3)),(19,23),((.12,.20),(.30,.45))):
        rng=np.random.default_rng(seed)
        scales=np.r_[1.,1.,np.geomspace(.3,3.,dimension-2)]
        x0=rng.normal(size=(6,dimension))*scales
        x1=rng.normal(size=(6,dimension))*scales
        x1[:,:2]+=[1.4,.9]
        screening=np.abs(x0.mean(axis=0)-x1.mean(axis=0))/(x0.std(axis=0)+x1.std(axis=0)+1e-12)
        warm=tuple(sorted(np.argsort(-screening,kind='stable')[:budget]))
        case=dict(id=f'confirm_{loss}_d{dimension}_n6_s{seed}_b{budget}_theta{theta[0]}_{theta[1]}',
                  loss=loss,dimension=dimension,budget=budget,seed=seed,theta=list(theta),
                  warm_support=list(warm), startup='data_screening',
                  screening_scores=screening.tolist())
        data=dict(x0=x0.tolist(),x1=x1.tolist(),q0=[1/6]*6,q1=[1/6]*6)
        case['data_sha256']=hashlib.sha256(np.r_[x0.ravel(),x1.ravel()].tobytes()).hexdigest()
        yield case,data,None


def main(stage):
    for name,expected in PROTOCOL['sources'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected
    if stage=='development':
        methods=['baseline','screened','core','master']
        cases=development_cases()
    else:
        winner=json.loads((HERE/'winner.json').read_text(encoding='utf-8'))['winner']
        methods=list(dict.fromkeys(['lcb',winner,'master']))
        cases=confirmation_cases()
    records=[]
    with threadpool_limits(limits=1):
        for index,(case,data,reference) in enumerate(cases):
            problem=Problem(data['x0'],data['x1'],case['budget'],*case['theta'],case['loss'],
                            q0=data['q0'],q1=data['q1'])
            raw_path=HERE/stage/'raw'/f"{case['id']}.json"
            if raw_path.exists():
                record=json.loads(raw_path.read_text(encoding='utf-8'))
            else:
                results={}
                order=methods[index%len(methods):]+methods[:index%len(methods)]
                for rule in order:
                    results[rule]=run(problem,rule,tuple(case['warm_support']),case['seed'])
                if reference is None:
                    reference=outer.enumerate_supports(problem,time_limit=60.,tolerance=2e-5).to_dict()
                    reference['loss']=case['loss']
                record=dict(case=case,data=data,method_order=order,reference=reference,
                            methods=results,checks={rule:invariant_check(results[rule],reference,rule)
                                                   for rule in methods})
                save(raw_path,record)
            records.append(record)
            print(stage,index+1,case['id'],
                  {r:(record['methods'][r]['visited'],record['methods'][r]['status']) for r in methods},flush=True)
    summary=summarize(records,methods)
    save(HERE/stage/'summary.json',summary)
    if False:
        eligible=[r for r in methods if r!='master' and summary['totals'][r]['all_pass']]
        winner=min(eligible,key=lambda r:(summary['totals'][r]['completed_evaluations']['total'],
                                        summary['totals'][r]['seconds']['total'],
                                        summary['totals'][r]['masters']['total']))
        save(HERE/'winner.json',dict(winner=winner,criterion=PROTOCOL['selection'],
                                    development_summary_sha256=hashlib.sha256(
                                        (HERE/stage/'summary.json').read_bytes()).hexdigest()))
    print(json.dumps(summary['totals'],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['development','confirmation'])
    main(parser.parse_args().stage)
