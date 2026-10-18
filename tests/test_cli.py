"""The core demo must remain usable without optional experiment dependencies."""

import subprocess
import sys

from robusthw.cli import parser


def test_run_worker_count_defaults_to_serial_and_can_be_declared():
    assert parser().parse_args(["run", "protocol.json"]).workers == 1
    assert parser().parse_args(["run", "protocol.json", "--workers", "2"]).workers == 2


def test_demo_does_not_import_optional_experiment_dependencies(tmp_path):
    code = """
import importlib.abc
import sys
class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'sklearn', 'matplotlib'}:
            raise ModuleNotFoundError('Optional dependency blocked: ' + fullname)
sys.meta_path.insert(0, CoreOnly())
from robusthw.cli import main
main(['demo', '--method', 'enumeration', '--output', sys.argv[1]])
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "demo.json")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert '"status":"optimal"' in result.stdout
