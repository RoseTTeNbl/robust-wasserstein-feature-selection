from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path('D:/OptExp/RubustHW')
HERE = Path(__file__).resolve().parent
SOURCE = ROOT/'reports/verification/20261010/full_inner_uncertainty/source_snapshot'
PACKAGE = HERE/'source/robusthw'
if PACKAGE.exists():
    raise SystemExit('The immutable source snapshot already exists')
shutil.copytree(ROOT/'code/robusthw', PACKAGE, ignore=shutil.ignore_patterns('__pycache__'))
for name in ('problem', 'oracle', 'solver', 'history', 'adaptive'):
    shutil.copy2(SOURCE/f'{name}.py', PACKAGE/f'{name}.py')

base = (PACKAGE/'adaptive.py').read_text(encoding='utf-8')
selection = '''            chosen, model, mean, sigma = min(choices, key=lambda item:
                                            (item[2]-item[3], p.support(item[0])))'''
assert selection in base
replacement = '''            def acquisition(item):
                if RULE in ("ei", "ei_guard"):
                    improvement = upper-tolerance/2-item[2]
                    if item[3] <= 0:
                        return -max(0., improvement)
                    standardized = improvement/item[3]
                    return -(improvement*ndtr(standardized)
                             + item[3]*np.exp(-.5*standardized**2)/np.sqrt(2*np.pi))
                return item[2]-item[3]

            if refreshed_guard and RULE in ("guard", "ei_guard"):
                chosen, model, mean, sigma = next(
                    item for item in choices if p.support(item[0]) == guard)
            else:
                chosen, model, mean, sigma = min(choices, key=lambda item:
                                                (acquisition(item), p.support(item[0])))'''
for rule in ('lcb', 'ei', 'guard', 'ei_guard'):
    variant = base.replace('import numpy as np',
                           f'import numpy as np\nfrom scipy.special import ndtr\nRULE = {rule!r}')
    variant = variant.replace('            scheduled = ',
                               '            refreshed_guard = False\n            scheduled = ')
    variant = variant.replace('                guard = p.support(z_guard)',
                               '                guard = p.support(z_guard)\n                refreshed_guard = True')
    variant = variant.replace(selection, replacement)
    variant = variant.replace('score=mean-sigma, candidates=candidate_count,',
                               'score=acquisition((chosen, model, mean, sigma)), '
                               'acquisition_rule=RULE, guard_priority=refreshed_guard '
                               'and RULE in ("guard", "ei_guard"), candidates=candidate_count,')
    (PACKAGE/f'adaptive_{rule}.py').write_text(variant, encoding='utf-8', newline='\n')

protocol = {
    'scope': 'TEMP-only ranking comparison, frozen v3 inner solver and certificate machinery',
    'rules': {
        'lcb': 'minimize m-sigma among cut-eligible candidates; original v3',
        'ei': 'maximize E[(T-F)+] with T=actual U-epsilon/2; zero sigma -> positive deterministic improvement',
        'guard': 'on a refreshed master prioritize its certified binary candidate; otherwise original LCB',
        'ei_guard': 'on a refreshed master prioritize its certified binary candidate; otherwise EI',
        'master': 'unchanged master-only baseline, not an acquisition-rule selection candidate'
    },
    'development': 'All 26 existing cases: two losses, D=6/8, n=4/8, seeds3/7/11, budget2; two D12 n4 seed3 budget3 cases',
    'confirmation': {'seeds':[19,23], 'dimension_budget_pairs':[[7,2],[10,3]],
                     'per_class_n':6, 'radius_pairs':[[.12,.20],[.30,.45]],
                     'losses':['overlap','logistic'], 'case_count':16},
    'selection': 'Among the four rules with every development case certified optimal and reference-consistent, minimize total completed support evaluations, then total elapsed solve time, then master count. Freeze exactly one winner before running confirmation; no retuning from confirmation.',
    'comparison': 'Same tail-coordinate initial support, explicit data/seed and solver tolerances. Five development methods rotate order by case index. Confirmation compares locked winner, original LCB and master only; all failures retained.',
    'settings': {'time_limit':60., 'root_rounds':0, 'tolerance':2e-5,
                 'relative_tolerance':0., 'proposal_count':8, 'certify_every':3,
                 'core_point':False, 'gate':'risk', 'cut_mode':'tangent',
                 'max_iterations':10000, 'max_refinements':2},
    'invariants': ['Every queried support binary and fully optimized',
                   'One final owned tangent per completed support',
                   'Only actual tests update U; valid global master bounds update LB',
                   'No repricing, inherited tests, new-point cuts, fractional inner calls',
                   'All acquisitions satisfy a <= U-epsilon/2 and accurate evaluations meet delta=epsilon/4'],
    'sources': {str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(PACKAGE.rglob('*.py'))}
}
(HERE/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
print(HERE)
