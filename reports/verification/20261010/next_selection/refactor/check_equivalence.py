"""Independent before/after checks; temporary audit, not a benchmark."""
import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('D:/OptExp/RubustHW')
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'code'))
from robusthw import GateOracle, Problem
import robusthw.oracle as current_oracle
import robusthw.solver as current_solver


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


before_oracle = load('robusthw.oracle_before', HERE / 'oracle_before.py')
before_solver = load('robusthw.solver_before', HERE / 'solver_before.py')
problem = Problem([[0, 0, 1], [1, .4, 0]], [[2, .1, 0], [3, 1, .5], [1, 2, 1]],
                  1, .13, .21, 'logistic', q0=[.3, .7], q1=[.2, .3, .5])


def canonical(oracle, method, point, fixed=None):
    captured = []

    def capture(model, *args, **kwargs):
        data = model.get_problem_data('CLARABEL')[0]
        captured.append(data)
        return {'status': 'accuracy_limit'}

    oracle._cvx_solve = capture
    if fixed is None:
        getattr(oracle, method)(np.asarray(point), None)
    else:
        getattr(oracle, method)(np.asarray(point), None, fixed)
    assert len(captured) == 1
    return captured[0]


model_checks = []
for gate, core in [('risk', False), ('risk', True), ('triangle', False), ('box', False)]:
    for point in ([1., 0, 0], [.2, .7, .3]):
        for fixed in (False, True):
            old = canonical(before_oracle.GateOracle(problem, gate, core), '_primal_cone', point, fixed)
            new = canonical(GateOracle(problem, gate, core), '_primal_cone', point, fixed)
            assert str(old['dims']) == str(new['dims'])
            for key in ('c', 'b'):
                np.testing.assert_array_equal(old[key], new[key])
            assert old['A'].shape == new['A'].shape
            assert (old['A'] != new['A']).nnz == 0
            model_checks.append(dict(side='primal', gate=gate, core=core, fixed=fixed,
                                     binary=bool(point == [1., 0, 0]), exact=True))
        old = canonical(before_oracle.GateOracle(problem, gate, core), '_dual_cone', point)
        new = canonical(GateOracle(problem, gate, core), '_dual_cone', point)
        assert str(old['dims']) == str(new['dims'])
        for key in ('c', 'b'):
            np.testing.assert_array_equal(old[key], new[key])
        assert old['A'].shape == new['A'].shape
        assert (old['A'] != new['A']).nnz == 0
        model_checks.append(dict(side='dual', gate=gate, core=core,
                                 binary=bool(point == [1., 0, 0]), exact=True))


def functions(path):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


unchanged = {}
for filename, names in {
    'oracle': ['_lp', '_primal_lp', '_dual_lp', '_repair_primal', '_repair_dual', '_cvx_solve',
               'reprice', 'solve', 'fixed_support', 'gate_primal'],
    'solver': ['_master', '_master_impl', '_ratio_master', '_milp_with_retry', 'enumerate_supports'],
}.items():
    old = functions(HERE / f'{filename}_before.py')
    new = functions(ROOT / f'code/robusthw/{filename}.py')
    for name in names:
        assert old[name] == new[name], (filename, name)
        unchanged[f'{filename}.{name}'] = True

assert (HERE / 'cli_before.py').read_bytes() == (ROOT / 'code/robusthw/cli.py').read_bytes()

bad_options = [dict(time_limit=0), dict(tolerance=0), dict(relative_tolerance=-1),
               dict(root_rounds=-1), dict(max_iterations=0), dict(cut_mode='bad'),
               dict(cut_mode='ratio', gate='box'), dict(search='invalid')]
validation_checks = []
for options in bad_options:
    outcomes = []
    for module in (before_solver, current_solver):
        try:
            module.solve(problem, **options)
        except ValueError as error:
            outcomes.append(str(error))
        else:
            raise AssertionError('Invalid options accepted')
    assert outcomes[0] == outcomes[1]
    validation_checks.append(dict(options=options, identical_error=outcomes[0]))

record = dict(canonical_cone_models=model_checks, canonical_checks=len(model_checks),
              critical_functions_ast_unchanged=unchanged, cli_bytes_unchanged=True,
              validation_checks=validation_checks)
(HERE / 'equivalence.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(dict(canonical_checks=len(model_checks), all_exact=True,
                     critical_functions_unchanged=len(unchanged), invalid_options=len(bad_options))))
