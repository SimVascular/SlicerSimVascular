"""Running svZeroDSolver on a case, and reading back what it wrote.

svZeroDSolver is `svzerod` on PyPI (`pip install svzerod`), as wheels for every platform and
every CPython from 3.9, so nothing has to be built or kept anywhere of its own. It is still run
as a process -- the `svzerodsolver <input.json> <output.csv>` command the package installs --
rather than imported, because it is C++ and a model it cannot handle is not always an exception.
The released wheels compile its assertions out, so an inconsistent model reaches Eigen
unchecked; and its command line ends in C's `exit()` when it is called wrongly. Either, in
process, takes the host with it -- under Slicer, the whole application and an unsaved scene. A
0D model of a few dozen vessels solves in well under a second, so nothing is lost by the round
trip through files, and the host never imports the pandas `svzerod` brings.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sysconfig
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The package, as pip is asked for it, and the command it installs.
SOLVER_PACKAGE = "svzerod"
SOLVER_EXECUTABLE_NAME = "svzerodsolver"

# The columns svZeroDSolver writes by default: one row per vessel per time point.
RESULT_COLUMNS = ("name", "time", "flow_in", "flow_out", "pressure_in", "pressure_out")


class SolverError(RuntimeError):
    """Raised when the solver cannot be found, fails, or writes something unreadable."""


def installed_solver() -> str | None:
    """`svzerodsolver` as `pip install svzerod` put it beside this Python, or None.

    pip puts the command in the scripts folder of the interpreter it installs into, which is on
    the PATH only while that environment is activated -- and, under Slicer, never: Slicer's own
    Python is not on anyone's PATH. So it is looked for where pip put it, not where a shell would.
    """
    return shutil.which(SOLVER_EXECUTABLE_NAME, path=sysconfig.get_path("scripts"))


def find_solver(configured: str | None = None) -> str | None:
    """The executable to run: the one configured, else svzerod's beside this Python, else
    `svzerodsolver` on the PATH, else None.

    A host whose environment is not the solver's -- Slicer is one -- should ask
    `installed_solver` instead: a command found on the PATH belongs to some other Python, and run
    with the host's environment it starts on the host's.
    """
    if configured:
        configured = os.path.expanduser(configured)
        return configured if os.path.isfile(configured) and os.access(configured, os.X_OK) else None
    return installed_solver() or shutil.which(SOLVER_EXECUTABLE_NAME)


def run_solver(config_path, results_path, executable: str, timeout: float | None = 600, env=None):
    """Solve the case and write the results, raising with the solver's own last words if it fails.

    The results file is removed first: the solver writes it only on success, so an old one left
    in place would be read as this run's.

    :param env: the environment to run it in: by default this one's, which is the right one for
      svzerod installed into this Python. A host running another Python's build passes that
      Python's -- under Slicer, `slicer.util.startupEnvironment()`, because Slicer's launcher
      points PYTHONHOME and the library path at Slicer's Python, and another Python run on those
      finds Slicer's packages and none of its own.
    """
    results_path = Path(results_path)
    if results_path.is_file():
        results_path.unlink()
    try:
        completed = subprocess.run(
            [executable, str(config_path), str(results_path)],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(Path(config_path).parent), env=env,
        )
    except FileNotFoundError:
        raise SolverError(f"No solver at {executable}.") from None
    except subprocess.TimeoutExpired:
        raise SolverError(f"svZeroDSolver did not finish within {timeout:g} s.") from None
    if completed.returncode != 0 or not results_path.is_file():
        said = (completed.stderr or completed.stdout or "").strip().splitlines()[-8:]
        raise SolverError("svZeroDSolver failed"
                          + (f" (exit {completed.returncode})" if completed.returncode else "")
                          + (": " + " ".join(said) if said else "."))
    return completed


@dataclass(frozen=True)
class VesselResult:
    """One vessel's flows and pressures at its two ends, over the cycle the solver wrote."""

    time: np.ndarray
    flow_in: np.ndarray
    flow_out: np.ndarray
    pressure_in: np.ndarray
    pressure_out: np.ndarray


def read_results(path) -> dict:
    """`{vessel name: VesselResult}` from svZeroDSolver's CSV.

    By default the solver writes the last cardiac cycle only, which is the one past the
    transient, so a mean over what is here is a cycle average.
    """
    rows = {}
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or tuple(header[:len(RESULT_COLUMNS)]) != RESULT_COLUMNS:
            raise SolverError(f"{path} does not start with svZeroDSolver's columns "
                              f"{', '.join(RESULT_COLUMNS)}; it starts with {header}.")
        for row in reader:
            if row:
                rows.setdefault(row[0], []).append([float(value) for value in row[1:6]])
    results = {}
    for name, values in rows.items():
        array = np.asarray(values, dtype=float)
        results[name] = VesselResult(*(array[:, column] for column in range(5)))
    if not results:
        raise SolverError(f"{path} holds no results.")
    return results
