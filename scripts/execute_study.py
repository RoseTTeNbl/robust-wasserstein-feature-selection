"""Run the frozen study in sequence; keep raw outcomes and refresh a report.

Example: python scripts/execute_study.py --study-root outputs/study-20261018
The immutable checkout is STUDY_ROOT/frozen-source. Scaling runs last, alone,
with one worker. Stop other research jobs before starting this pipeline.
"""

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import BadZipFile, ZipFile

DEFAULT_SUITES = ("main", "public", "shift", "ablation", "scaling")
SUITES = ("small", "anchor", *DEFAULT_SUITES)


class IntegrityError(RuntimeError):
    """A frozen input or completed output cannot be trusted."""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hash(value):
    content = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, value):
    value["updated_at_utc"] = _now()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_protocol(root, suite):
    """Check the manifest, its ZIP and every live file in the frozen checkout."""
    path = root / suite / "protocol.json"
    try:
        manifest = _read(path)
        payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
        if _hash(payload) != manifest["manifest_hash"] or manifest["suite"] != suite:
            raise IntegrityError(f"Frozen protocol changed: {path}")
        sources = manifest["source_hashes"]
        with ZipFile(path.with_suffix(".sources.zip")) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != set(sources):
                raise IntegrityError(f"Source snapshot members changed: {suite}")
            for name, expected in sources.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                    raise IntegrityError(f"Source snapshot changed: {suite}/{name}")
                if _file_hash(root / "frozen-source" / name) != expected:
                    raise IntegrityError(f"Frozen checkout changed: {name}")
        return manifest
    except (OSError, KeyError, TypeError, ValueError, BadZipFile) as error:
        raise IntegrityError(f"Cannot verify {suite}: {error}") from error


def verify_execution(root, suite, manifest):
    """Accept success/failure outcomes only after every planned ID is recorded."""
    try:
        execution = _read(root / suite / "execution.json")
        expected = {spec["independent_data_id"]: spec for spec in manifest["specs"]}
        observed = execution["records"]
        if (
            execution["manifest_hash"] != manifest["manifest_hash"]
            or execution["planned"] != len(expected)
            or execution["attempted"] != len(expected)
            or execution["partial_execution"]
            or len(observed) != len(expected)
            or {row["data_id"] for row in observed} != set(expected)
        ):
            raise IntegrityError(f"Incomplete or conflicting execution: {suite}")
        counts = {"completed": 0, "failed": 0}
        for row in observed:
            path = root / suite / "records" / (row["data_id"] + ".json")
            record = _read(path)
            payload = {key: value for key, value in record.items() if key != "record_hash"}
            if (
                _file_hash(path) != row["sha256"]
                or _hash(payload) != record["record_hash"]
                or record["manifest_hash"] != manifest["manifest_hash"]
                or _hash(record["spec"]) != _hash(expected[row["data_id"]])
                or record["status"] != row["status"]
                or row["status"] not in counts
            ):
                raise IntegrityError(f"Raw outcome changed: {suite}/{row['data_id']}")
            counts[row["status"]] += 1
        if any(execution[key] != count for key, count in counts.items()):
            raise IntegrityError(f"Execution counts disagree with records: {suite}")
        return execution
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise IntegrityError(f"Cannot verify {suite} execution: {error}") from error


def _alive(pid):
    if os.name == "nt":
        from ctypes import wintypes

        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return kernel.GetLastError() == 5
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and (
                code.value == 259
            )
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def run_pipeline(study_root, data_root, *, workers=2, suites=DEFAULT_SUITES, report_script=None):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("Workers must be a positive integer")
    suites = tuple(suites)
    if (
        not suites
        or len(set(suites)) != len(suites)
        or any(suite not in SUITES for suite in suites)
    ):
        raise ValueError("Choose distinct known suites")
    if "scaling" in suites and suites[-1] != "scaling":
        raise ValueError("Scaling must run last, without other experimental jobs")
    if "main" in suites and "shift" in suites and suites.index("shift") < suites.index("main"):
        raise ValueError("Shift must follow main")
    root, data_root = Path(study_root).resolve(), Path(data_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "pipeline.lock"
    if lock.exists():
        if _alive(int(lock.read_text(encoding="ascii"))):
            raise RuntimeError("A study pipeline is already running")
        lock.unlink()
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(str(os.getpid()))
    state_path = root / "pipeline.json"
    previous = _read(state_path) if state_path.exists() else {}
    state = {
        "schema_version": 1,
        "study_root": str(root),
        "started_at_utc": _now(),
        "pid": os.getpid(),
        "python": sys.executable,
        "suites": suites,
        "workers": workers,
        "status": "running",
        "phase": None,
        "phases": [],
        "history": previous.get("history", [])
        + (
            [{key: value for key, value in previous.items() if key != "history"}]
            if previous
            else []
        ),
    }
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "frozen-source" / "src")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    temporary = root / "runtime-temp"
    temporary.mkdir(exist_ok=True)
    environment["TEMP"] = environment["TMP"] = str(temporary)
    environment["MPLCONFIGDIR"] = str(temporary / "matplotlib")
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    report_script = Path(report_script).resolve() if report_script else None

    def launch(command, phase):
        token = f"pipeline-{os.getpid()}-{len(state['phases']):02d}-{phase['suite']}"
        suffix = "-report" if phase.get("kind") == "report" else ""
        phase.update(
            command=command,
            started_at_utc=_now(),
            status="running",
            stdout=str(logs / (token + suffix + ".stdout.log")),
            stderr=str(logs / (token + suffix + ".stderr.log")),
        )
        with Path(phase["stdout"]).open("wb") as stdout, Path(phase["stderr"]).open("wb") as stderr:
            process = subprocess.Popen(
                command,
                cwd=root / "frozen-source",
                env=environment,
                stdout=stdout,
                stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            phase["pid"] = process.pid
            _save(state_path, state)
            try:
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        phase.update(exit_code=code, ended_at_utc=_now())
        _save(state_path, state)
        return code

    exit_code = 0
    try:
        _save(state_path, state)
        for suite in suites:
            state["phase"] = suite
            phase = {"suite": suite, "kind": "experiment", "status": "verifying"}
            state["phases"].append(phase)
            _save(state_path, state)
            manifest = verify_protocol(root, suite)
            if suite == "shift":
                verify_execution(root, "main", verify_protocol(root, "main"))
            command = [
                sys.executable,
                "-m",
                "robusthw",
                "run",
                str(root / suite / "protocol.json"),
                "--data-root",
                str(data_root),
                "--workers",
                str(1 if suite == "scaling" else workers),
            ]
            if suite == "shift":
                command.extend(["--reuse-root", str(root / "main")])
            code = launch(command, phase)
            if code not in (0, 1):
                raise RuntimeError(f"{suite} exited unexpectedly with code {code}")
            execution = verify_execution(root, suite, manifest)
            if code != int(execution["failed"] > 0):
                raise RuntimeError(f"{suite} exit code disagrees with completed outcomes")
            phase.update(
                status="completed_with_failures" if execution["failed"] else "completed",
                planned=execution["planned"],
                completed=execution["completed"],
                failed=execution["failed"],
            )
            exit_code = max(exit_code, code)
            _save(state_path, state)
            if report_script and report_script.is_file():
                report = {"suite": suite, "kind": "report", "status": "pending"}
                state["phases"].append(report)
                report_code = launch(
                    [sys.executable, str(report_script), "--study-root", str(root)], report
                )
                report["status"] = "completed" if report_code == 0 else "failed"
                exit_code = max(exit_code, int(report_code != 0))
            else:
                phase["report_status"] = "skipped"
        state.update(
            status="completed_with_failures" if exit_code else "completed",
            phase=None,
            ended_at_utc=_now(),
        )
    except IntegrityError as error:
        state.update(status="stopped_integrity_error", error=str(error), ended_at_utc=_now())
        if state["phases"]:
            state["phases"][-1].update(status="integrity_error", error=str(error))
        exit_code = 2
    except KeyboardInterrupt:
        state.update(status="interrupted", ended_at_utc=_now())
        if state["phases"]:
            state["phases"][-1]["status"] = "interrupted"
        exit_code = 130
    except Exception as error:
        state.update(status="stopped_unexpected_error", error=str(error), ended_at_utc=_now())
        if state["phases"]:
            state["phases"][-1].update(status="unexpected_error", error=str(error))
        exit_code = 3
    finally:
        _save(state_path, state)
        lock.unlink(missing_ok=True)
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/uci"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--suites", choices=SUITES, nargs="+", default=DEFAULT_SUITES)
    parser.add_argument("--report-script", type=Path)
    args = parser.parse_args(argv)
    code = run_pipeline(
        args.study_root,
        args.data_root,
        workers=args.workers,
        suites=args.suites,
        report_script=args.report_script,
    )
    print(json.dumps({"pipeline": str(args.study_root / "pipeline.json"), "exit_code": code}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
