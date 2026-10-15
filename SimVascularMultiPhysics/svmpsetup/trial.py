"""A few steps of a written case on this machine, before a cluster is given days of it.

A `solver.xml` that the writer's checks pass can still fail in the solver: a library missing, an
element the mesher left inverted, an outlet that draws flow in, a step the flow will not converge
at. On a cluster each of those is found after a wait in the queue, and some after hours of a run
going nowhere. Five steps here find them in minutes, and time the steps, which says what the full
run will cost before it is asked for.

**The test is a copy of the written file, not the case written again.** It is the case's own
`solver.xml` with the step count and the saving changed and nothing else, and its paths pointed
back at the case's mesh and waveforms -- so what passes is the file that goes to the cluster, hand
edits included. Only its last step is saved, the one there is to look at, so that the steps are
timed as the full run's are, without a 40 MB result written at every one. It is written beside the
case, in `<case>-test`, rather than in it, so that the case folder is still the one to copy,
without test results in it.

**svMultiPhysics' report is read here so that nobody has to know how.** Its iteration lines say
less than they look as though they do, and some of what goes wrong says nothing at all:

- the `s` after `<step>-<iteration>` marks a step whose results were *saved*, not one that
  converged;
- a step that uses up its nonlinear iterations is accepted as converged without a word, so a step
  that did not converge shows only as one whose last iteration is the limit and whose residual is
  still above the tolerance;
- nothing checks for NaN but an RCR outlet's own integration, so a run gone NaN otherwise carries
  on, writing NaN, for as long as it is allowed to;
- an error is printed by the first process only, so one raised on another shows only as MPI's
  abort, with nothing of the solver's own.

**Where the solver is.** The installers put it at `/usr/local/sv/svMultiPhysics/<date>/bin`, on
Ubuntu and macOS alike; a build of one's own is wherever its CMake put it, and is given as a path,
or at `SVMULTIPHYSICS`, as the tests take it. Windows has no build: the Ubuntu one is run inside
WSL2, which this finds and runs through `wsl.exe`. With no `mpiexec` it is run as one process,
which svMultiPhysics allows.

Nothing here imports Slicer. The panel runs the command `command` builds as a process of its own
and feeds what it prints, line by line, to `Progress`.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import vtk

from svromsetup import boundary_conditions as bcs

from svmpsetup.case import SOLVER_XML_NAME, CaseError

DEFAULT_STEPS = 5
DEFAULT_PROCESSES = 8

# The test's folder is the case's, with this after its name: beside the case, never in it.
TRIAL_SUFFIX = "-test"

SOLVER_NAME = "svmultiphysics"
SOLVER_ENVIRONMENT_VARIABLE = "SVMULTIPHYSICS"
INSTALL_ROOT = "/usr/local/sv/svMultiPhysics"
# Where `mpiexec` is when the PATH an application is started with leaves it out -- as a macOS
# application started from the Finder is, with only the system's folders on it: Homebrew's on
# Apple silicon, Homebrew's on Intel and most source installs, the system's own.
MPIEXEC_NAMES = ("mpiexec", "mpirun")
MPIEXEC_FOLDERS = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin")
WSL = "wsl.exe"

# What the case's file says when it says nothing: the stop file a run is stopped with, if the
# writer did not name one, so that the test can always be stopped.
DEFAULT_STOP_FILE = "STOP_SIM"
# The installers' instructions for what svMultiPhysics needs beside it.
DEPENDENCIES = ("Install OpenMPI, LAPACK and BLAS: on Ubuntu, sudo apt-get install "
                "libopenblas-dev libopenmpi-dev; on macOS, brew install openblas lapack openmpi.")


# -- finding the solver ------------------------------------------------------------------------
@dataclass(frozen=True)
class Solver:
    """svMultiPhysics as it is to be run: its path, and `mpiexec`'s, both as the process that runs
    them sees them -- inside WSL, for one run there."""

    executable: str
    mpiexec: str | None = None
    wsl: bool = False

    def describe(self) -> str:
        where = f"{self.executable} (in WSL)" if self.wsl else self.executable
        return where if self.mpiexec else f"{where}, without mpiexec"


def _runnable(path) -> bool:
    return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def installed_solvers(root=INSTALL_ROOT) -> list:
    """The installers' svmultiphysics, oldest build first: their folders are dated YYYY-MM-DD."""
    found = sorted(Path(root).glob(f"*/bin/{SOLVER_NAME}"), key=lambda path: path.parent.parent.name)
    return [str(path) for path in found if _runnable(str(path))]


def find_mpiexec(executable, search_path=None) -> str | None:
    """`mpiexec` beside the solver, on the PATH, or where Homebrew or the system put it."""
    folders = [str(Path(executable).parent)]
    folders += [folder for folder in (search_path or os.environ.get("PATH", "")).split(os.pathsep) if folder]
    folders += list(MPIEXEC_FOLDERS)
    for name in MPIEXEC_NAMES:
        found = shutil.which(name, path=os.pathsep.join(folders))
        if found:
            return found
    return None


def find_solver(configured=None, environ=None, *, install_root=INSTALL_ROOT, platform=None,
                wsl_runner=None) -> Solver | None:
    """The svMultiPhysics to run, or None: the one configured, else `SVMULTIPHYSICS`, else
    `svmultiphysics` on the PATH, else the newest the installers put in place; on Windows, else
    the same inside WSL.

    A path that is configured and is not there is None, not a reason to look elsewhere: a test
    passed by some other build than the one asked for is a test of the wrong thing. On Windows a
    configured path that begins with `/` is one inside WSL.

    `environ` should be the one the solver will be run with -- under Slicer, the environment
    Slicer was started with, not the one its launcher made for its own Python and libraries.
    """
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    search_path = environ.get("PATH", "")
    windows = platform.startswith("win")
    if configured:
        configured = os.path.expanduser(str(configured))
        if windows and configured.startswith("/"):
            found = find_solver_in_wsl(runner=wsl_runner)
            return Solver(configured, found.mpiexec if found else None, wsl=True)
        return Solver(configured, find_mpiexec(configured, search_path)) if _runnable(configured) else None
    candidates = [environ.get(SOLVER_ENVIRONMENT_VARIABLE), shutil.which(SOLVER_NAME, path=search_path)]
    candidates += list(reversed(installed_solvers(install_root)))
    for candidate in candidates:
        if _runnable(candidate):
            return Solver(candidate, find_mpiexec(candidate, search_path))
    if windows:
        return find_solver_in_wsl(runner=wsl_runner)
    return None


# What WSL is asked: every installed build, the one on its PATH, and its mpiexec, each on a line
# of its own that says which it is. One call, because each call to wsl.exe starts a shell in it.
WSL_PROBE = (
    f'for f in {INSTALL_ROOT}/*/bin/{SOLVER_NAME}; do [ -x "$f" ] && echo "installed:$f"; done; '
    f'p=$(command -v {SOLVER_NAME}) && echo "path:$p"; '
    'p=$(command -v mpiexec || command -v mpirun) && echo "mpiexec:$p"; true')


def find_solver_in_wsl(runner=None) -> Solver | None:
    """svMultiPhysics inside WSL, found as on Linux: on its PATH, else the newest installed."""
    runner = runner or subprocess.run
    if runner is subprocess.run and shutil.which(WSL) is None:
        return None
    try:
        done = runner([WSL, "--exec", "sh", "-c", WSL_PROBE], capture_output=True, text=True,
                      timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_wsl_probe(done.stdout or "")


def parse_wsl_probe(text) -> Solver | None:
    found = {"installed": [], "path": [], "mpiexec": []}
    for line in text.splitlines():
        kind, _, value = line.strip().partition(":")
        if kind in found and value:
            found[kind].append(value)
    installed = sorted(found["installed"], key=lambda path: path.split("/")[-3])
    chosen = (found["path"] or installed[-1:] or [None])[0]
    if chosen is None:
        return None
    return Solver(chosen, (found["mpiexec"] or [None])[0], wsl=True)


# -- the test case -----------------------------------------------------------------------------
def trial_folder(case_dir) -> Path:
    case_dir = Path(case_dir)
    return case_dir.with_name(case_dir.name + TRIAL_SUFFIX)


def _set(parent, tag, value):
    element = parent.find(tag)
    if element is None:
        element = ET.SubElement(parent, tag)
    element.text = str(value)


def _general(root) -> ET.Element:
    general = root.find("GeneralSimulationParameters")
    if general is None:
        raise CaseError("The solver.xml has no GeneralSimulationParameters.")
    return general


def read_case(case_dir) -> ET.Element:
    path = Path(case_dir) / SOLVER_XML_NAME
    if not path.is_file():
        raise CaseError(f"There is no {SOLVER_XML_NAME} in {case_dir}: create the solver files first.")
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as error:
        raise CaseError(f"{path} cannot be read: {error}.") from None


@dataclass(frozen=True)
class Schedule:
    """What the full run is asked to do, read off its `solver.xml`: the test is timed against it."""

    steps: int
    save_increment: int
    start_saving: int
    min_iterations: int
    max_iterations: int
    tolerance: float

    @property
    def saves(self) -> int:
        """How many result files the full run writes: a step is saved when it is a multiple of
        the increment, from the start step on."""
        if self.save_increment <= 0 or self.steps < self.start_saving:
            return 0
        first = math.ceil(max(self.start_saving, 1) / self.save_increment) * self.save_increment
        return 0 if first > self.steps else (self.steps - first) // self.save_increment + 1


def _number(element, tag, kind, default):
    text = (element.findtext(tag) if element is not None else None) or ""
    try:
        return kind(text.strip())
    except ValueError:
        return default


def schedule(case_dir) -> Schedule:
    root = read_case(case_dir)
    general = _general(root)
    fluid = root.find("Add_equation[@type='fluid']")
    return Schedule(_number(general, "Number_of_time_steps", int, 0),
                    _number(general, "Increment_in_saving_VTK_files", int, 0),
                    _number(general, "Start_saving_after_time_step", int, 1),
                    _number(fluid, "Min_iterations", int, 1),
                    _number(fluid, "Max_iterations", int, 0),
                    _number(fluid, "Tolerance", float, 0.0))


def write_trial(case_dir, steps=DEFAULT_STEPS, trial_dir=None) -> Path:
    """The case's `solver.xml`, cut to `steps` steps with the last one saved, in `<case>-test`.

    Every path in it is pointed back at the case's own files, so the test runs on them rather than
    on copies -- a 1.3-million-element mesh is 20 MB. A previous test's results are cleared first:
    svMultiPhysics appends to a face results file that is already there rather than starting it
    again, and a test read against the last one's rows reads wrong.
    """
    case_dir = Path(case_dir).resolve()
    trial_dir = Path(trial_dir).resolve() if trial_dir else trial_folder(case_dir)
    if steps < 1:
        raise CaseError("A test needs at least one time step.")
    root = read_case(case_dir)
    general = _general(root)
    _set(general, "Number_of_time_steps", steps)
    _set(general, "Continue_previous_simulation", "false")
    _set(general, "Save_results_to_VTK_format", "true")
    # The last step, and only it: the result shown, and a step timed as the full run's are -- which
    # save one in a hundred -- rather than with a result written at each. The faces' flow and
    # pressure are written every step regardless.
    _set(general, "Start_saving_after_time_step", 1)
    _set(general, "Increment_in_saving_VTK_files", steps)
    # Once, at the end: the test is not restarted.
    _set(general, "Increment_in_saving_restart_files", steps)
    if not (general.findtext("Searched_file_name_to_trigger_stop") or "").strip():
        _set(general, "Searched_file_name_to_trigger_stop", DEFAULT_STOP_FILE)
    # Results where the test looks for them, <N>-procs in its folder, whatever the case says.
    for folder in general.findall("Save_results_in_folder"):
        general.remove(folder)
    for element in root.iter():
        text = (element.text or "").strip()
        if element.tag.lower().endswith("file_path") and text and not os.path.isabs(text):
            # Forward slashes whatever this machine uses: the solver reads them on Linux, in WSL
            # from a file written on Windows.
            element.text = os.path.relpath(case_dir / text, trial_dir).replace(os.sep, "/")

    trial_dir.mkdir(parents=True, exist_ok=True)
    for previous in trial_dir.glob("*-procs"):
        if previous.is_dir():
            shutil.rmtree(previous)
    stop = trial_dir / stop_file_name(root)
    if stop.exists():
        stop.unlink()
    path = trial_dir / SOLVER_XML_NAME
    ET.indent(root, space="  ")
    path.write_text('<?xml version="1.0" encoding="UTF-8" ?>\n'
                    f"<!-- A {steps}-step test of {case_dir / SOLVER_XML_NAME}, written by SimVascular "
                    "MultiPhysics: only the steps and the saving differ from it. -->\n"
                    + ET.tostring(root, encoding="unicode") + "\n")
    return path


def stop_file_name(root) -> str:
    general = root.find("GeneralSimulationParameters")
    name = (general.findtext("Searched_file_name_to_trigger_stop") if general is not None else "") or ""
    return name.strip() or DEFAULT_STOP_FILE


def stop(trial_dir) -> Path:
    """Ask a running test to stop: svMultiPhysics looks for its stop file after every step, and
    stops cleanly, writing what it has, when it is there. An empty file stops it at once."""
    path = Path(trial_dir) / stop_file_name(read_case(trial_dir))
    path.write_text("")
    return path


def command(solver: Solver, processes, trial_dir) -> list:
    """The command line of a test: `mpiexec -np N svmultiphysics solver.xml`, run in the test's
    folder; svmultiphysics alone with one process or no mpiexec; through wsl.exe in WSL, which
    is told the folder because it translates a Windows one itself."""
    run = [solver.executable, SOLVER_XML_NAME]
    if processes > 1 and solver.mpiexec:
        run = [solver.mpiexec, "-np", str(int(processes))] + run
    if solver.wsl:
        return [WSL, "--cd", str(trial_dir), "--exec"] + run
    return run


# -- reading what it prints --------------------------------------------------------------------
# ` NS 3-5s  6.557e-02  [-28 3.571e-02 6.935e-03 2.235e-01]  [1 -9 0]`: the equation, the step and
# the nonlinear iteration (`s` when that step's results were saved), wall-clock seconds since the
# solver finished setting up; then, in brackets, the residual in dB against the step's first
# iteration, the residual against the step's first and against the run's first, and how far the
# linear solve brought it down; then the linear solve's iterations, its last reduction in dB, and
# its share of the time. `!` for `[` marks a residual grown tenfold in the step, or a linear solve
# that did not converge.
ITERATION_LINE = re.compile(
    r"^\s*(?P<equation>[A-Za-z]+)\s+(?P<step>\d+)-(?P<iteration>\d+)(?P<saved>s?)\s+(?P<time>\S+)"
    r"\s+(?P<open>[\[!])\s*(?P<db>\S+)\s+(?P<ri_r1>\S+)\s+(?P<ri_r0>\S+)\s+(?P<r_ri>\S+?)\s*[\]!]"
    r"\s+(?P<ls_open>[\[!])\s*(?P<ls_iterations>\S+)\s+(?P<ls_db>\S+)\s+(?P<ls_share>\S+?)\s*[\]!]")


def _float(text) -> float:
    try:
        return float(text)
    except ValueError:
        return math.nan


@dataclass(frozen=True)
class Iteration:
    step: int
    iteration: int
    saved: bool
    time: float
    ri_r1: float
    ri_r0: float
    residual_grew: bool
    linear_failed: bool

    @property
    def nan(self) -> bool:
        return any(math.isnan(value) for value in (self.time, self.ri_r1, self.ri_r0))


def parse_iteration(line) -> Iteration | None:
    match = ITERATION_LINE.match(line)
    if match is None:
        return None
    return Iteration(int(match["step"]), int(match["iteration"]), bool(match["saved"]),
                     _float(match["time"]), _float(match["ri_r1"]), _float(match["ri_r0"]),
                     match["open"] == "!", match["ls_open"] == "!")


# What is recognised in what the solver and MPI print, and what it is said to mean. The solver's
# own error line is followed by the exception's text, which is what says what went wrong.
SOLVER_ERROR = "[svMultiPhysics] ERROR"
KNOWN_FAILURES = (
    (re.compile(r"There are not enough slots available"),
     "This machine has fewer cores than the processors asked for: lower Number of Processor(s)."),
    (re.compile(r"error while loading shared libraries: (\S+?):"),
     "A library svMultiPhysics needs is missing ({0}). " + DEPENDENCIES),
    (re.compile(r"Library not loaded: (\S+)"),
     "A library svMultiPhysics needs is missing ({0}). " + DEPENDENCIES),
    (re.compile(r"MPI_ABORT was invoked on rank (\d+)"),
     "Process {0} aborted the run. Only the first process prints the solver's reason, so for "
     "another process there may be none above."),
    (re.compile(r"attempt to run as root"),
     "mpiexec will not run as root. Run Slicer as an ordinary user."),
)
KNOWN_WARNINGS = (
    (re.compile(r"ill-conditioned LHS matrix"),
     "The solver reported an ill-conditioned matrix, which usually means a badly shaped element."),
)


@dataclass
class Report:
    """What a test came to. `errors` stop a case going to the cluster; `notes` are worth reading."""

    steps_asked: int
    steps_done: int
    seconds_per_step: float | None
    exit_code: int | None
    stopped: bool
    errors: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and self.steps_done == self.steps_asked


class Progress:
    """svMultiPhysics' output, read a line at a time as it arrives.

    `limits` is the case's nonlinear iterations and tolerance, which is what a step's last
    iteration is held to: reaching the limit is how the solver accepts a step it could not
    converge, and says nothing about it.
    """

    def __init__(self, steps, limits: Schedule | None = None):
        self.steps = steps
        self.limits = limits
        self.last = {}            # step -> the step's last iteration seen
        self.errors = []
        self.warnings = []
        self.tail = deque(maxlen=200)
        self._error_context = 0   # lines of the exception's text still to come after the error line
        self._nan_step = None

    @property
    def current(self) -> Iteration | None:
        return self.last[max(self.last)] if self.last else None

    @property
    def nan_step(self) -> int | None:
        """The first step whose residual is NaN: the run has diverged, and will carry on."""
        return self._nan_step

    def feed(self, line) -> Iteration | None:
        line = line.rstrip("\r\n")
        self.tail.append(line)
        if self._error_context and line.strip():
            # The exception's text: one line, or a framework exception's header and its two lines
            # of where and what. What follows it is MPI's account of the abort, which adds nothing.
            text = line.strip()
            self.errors[-1] += " " + text
            self._error_context -= 1
            if text.startswith("[Core Exception]"):
                self._error_context = 2
            return None
        if SOLVER_ERROR in line:
            text = line.split(SOLVER_ERROR, 1)[1].lstrip(": ").strip()
            generic = "failed due to" in text
            self.errors.append("svMultiPhysics stopped with an error:" if generic else
                               f"svMultiPhysics stopped with an error: {text}")
            self._error_context = 1 if generic else 0
            return None
        for pattern, explanation in KNOWN_FAILURES:
            match = pattern.search(line)
            if match:
                message = explanation.format(*match.groups())
                # MPI's abort is the reason only when the solver gave none of its own.
                solver_said = any(error.startswith("svMultiPhysics stopped") for error in self.errors)
                if message not in self.errors and not (solver_said and "aborted the run" in message):
                    self.errors.append(message)
                return None
        for pattern, explanation in KNOWN_WARNINGS:
            if pattern.search(line) and explanation not in self.warnings:
                self.warnings.append(explanation)
        iteration = parse_iteration(line)
        if iteration is not None:
            self.last[iteration.step] = iteration
            if iteration.nan and self._nan_step is None:
                self._nan_step = iteration.step
        return iteration

    def step_seconds(self) -> list:
        """Wall-clock seconds of each finished step: the clock starts once setup is done."""
        times, previous = [], 0.0
        for step in sorted(self.last):
            times.append(self.last[step].time - previous)
            previous = self.last[step].time
        return times

    def seconds_per_step(self) -> float | None:
        """The steps after the first, which pays for what the first assembly sets up."""
        times = [t for t in self.step_seconds() if not math.isnan(t)]
        if not times:
            return None
        return float(np.mean(times[1:] if len(times) > 1 else times))

    def unconverged(self) -> list:
        """(step, iterations, residual) for each step accepted at the iteration limit with its
        residual still above the tolerance -- by both of the solver's measures of it."""
        if self.limits is None or not self.limits.max_iterations:
            return []
        found = []
        for step, last in sorted(self.last.items()):
            residual = min(last.ri_r1, last.ri_r0)
            if last.iteration >= self.limits.max_iterations and not residual <= self.limits.tolerance:
                found.append((step, last.iteration, residual))
        return found

    def finish(self, exit_code, crashed=False, stopped=False) -> Report:
        done = len(self.last)
        report = Report(self.steps, done, self.seconds_per_step(), exit_code, stopped,
                        list(self.errors), [])
        if self._nan_step is not None:
            report.errors.append(f"The solution became NaN at step {self._nan_step}: the run "
                                 "diverged, and svMultiPhysics carries on regardless.")
        if crashed and not report.errors:
            report.errors.append("svMultiPhysics crashed. Its last lines are in the output above.")
        elif exit_code not in (0, None) and not report.errors:
            report.errors.append(f"svMultiPhysics ended with exit code {exit_code}, without saying "
                                 "why. Its last lines are in the output above.")
        if not stopped and not report.errors and done < self.steps:
            report.errors.append(f"svMultiPhysics ended after {done} of {self.steps} steps.")
        unconverged = self.unconverged()
        if unconverged:
            steps = ", ".join(str(step) for step, _i, _r in unconverged)
            worst = max(residual for _s, _i, residual in unconverged)
            report.notes.append(
                f"Step{'s' if len(unconverged) > 1 else ''} {steps} used all "
                f"{self.limits.max_iterations} nonlinear iterations without reaching the tolerance "
                f"({self.limits.tolerance:g}); the worst ended at {worst:.1e}. The solver accepts such "
                "a step without saying so. Usual in the first steps from rest; if it goes on through "
                "the run, those steps are not converged.")
        grew = sorted(step for step, last in self.last.items() if last.residual_grew)
        if grew:
            report.notes.append("The residual grew more than tenfold within step"
                                f"{'s' if len(grew) > 1 else ''} {', '.join(map(str, grew))}.")
        linear = sorted(step for step, last in self.last.items() if last.linear_failed)
        if linear:
            report.notes.append("The linear solve did not converge at step"
                                f"{'s' if len(linear) > 1 else ''} {', '.join(map(str, linear))}.")
        report.notes += self.warnings
        return report


# -- what it says -----------------------------------------------------------------------------
def duration(seconds) -> str:
    """A time a person reads: `40 s`, `3 min`, `2.5 h`, `3.6 days`."""
    if seconds is None or math.isnan(seconds):
        return "an unknown time"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} min"
    if seconds < 36 * 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def size(nbytes) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if nbytes >= scale:
            return f"{nbytes / scale:.1f} {unit}"
    return f"{nbytes:.0f} bytes"


def results_folder(trial_dir, processes) -> Path:
    """Where svMultiPhysics writes: a folder named after the number of processes it ran on."""
    return Path(trial_dir) / f"{int(processes)}-procs"


def result_files(folder) -> list:
    """`result_NNN.vtu`, in step order: three digits to 999, unpadded from 1000."""
    def step(path):
        digits = path.stem.rsplit("_", 1)[-1]
        return int(digits) if digits.isdigit() else -1
    return sorted(Path(folder).glob("*_[0-9]*.vtu"), key=step)


def summary(report: Report, full: Schedule | None, processes, result_bytes=None) -> list:
    """The test in sentences: how far it got and how fast, what the full run will cost at that
    rate, and anything that went wrong."""
    lines = []
    per_step = report.seconds_per_step
    if report.steps_done:
        stopped = " before it was stopped" if report.stopped and report.steps_done < report.steps_asked else ""
        lines.append(f"{report.steps_done} of {report.steps_asked} steps{stopped}, "
                     f"{duration(per_step)} per step on {processes} processor{'s' if processes != 1 else ''}.")
    if per_step and full and full.steps:
        estimate = f"At that rate the full run, {full.steps} steps, takes about {duration(per_step * full.steps)} here"
        if result_bytes and full.saves:
            estimate += f", and its {full.saves} result files come to about {size(result_bytes * full.saves)}"
        lines.append(estimate + ". A cluster with more processors is faster; other work on this "
                     "machine makes the test, and the estimate with it, slower.")
    lines += report.errors
    if report.ok:
        lines.append("No errors.")
    lines += report.notes
    return lines


# -- which way the flow goes ---------------------------------------------------------------------
@dataclass(frozen=True)
class FaceIntegrals:
    """A `B_<eq>_<quantity>_flux.txt` or `_average.txt`: a value per face per step."""

    names: tuple
    areas: tuple
    rows: tuple        # (step, time, (value per face))

    def at(self, step=None):
        """The row of `step`, or the last."""
        if not self.rows:
            return None
        if step is None:
            return self.rows[-1]
        return next((row for row in self.rows if row[0] == step), None)


def read_face_integrals(path) -> FaceIntegrals:
    names, areas, rows = (), (), []
    for line in Path(path).read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("face areas:"):
            areas = tuple(float(value) for value in text.split(":", 1)[1].split())
        elif text.startswith("step"):
            names = tuple(text.split()[2:])
        else:
            values = text.split()
            rows.append((int(values[0]), float(values[1]), tuple(float(v) for v in values[2:])))
    return FaceIntegrals(names, areas, tuple(rows))


FLUX_FILE = "B_NS_Velocity_flux.txt"


def _inflow_at(condition: bcs.Inflow, time) -> float:
    if condition.is_steady:
        return float(condition.flow[0])
    times = np.asarray(condition.time, dtype=float)
    period = times[-1] - times[0]
    return float(np.interp(times[0] + (time - times[0]) % period, times, condition.flow))


@dataclass(frozen=True)
class FaceFlow:
    name: str
    flow: float          # into the model positive, as the conditions table takes an inflow
    expected: str        # "in", "out", or "none" for a wall
    waveform: float | None = None

    @property
    def right_way(self) -> bool:
        if self.expected == "in":
            return self.flow > 0
        if self.expected == "out":
            return self.flow < 0
        return True

    def describe(self) -> str:
        if self.expected == "none":
            return f"{self.name}: {abs(self.flow):.3g} mL/s through the wall"
        way = "in" if self.flow > 0 else "out"
        text = f"{self.name}: {abs(self.flow):.3g} mL/s {way}"
        if self.waveform is not None:
            text += f" (the waveform gives {self.waveform:.3g} at that time)"
        if not self.right_way:
            text += f" -- it should flow {self.expected}"
        return text


def face_flows(integrals: FaceIntegrals, conditions_by_name, step=None) -> list:
    """Which way the flow went through each face at `step` (the last by default), against which
    way its condition says it should.

    svMultiPhysics integrates the velocity over each face along the face's outward normal, so
    outflow is positive; here it is turned round to the conditions table's sense, an inflow
    positive. An inflow should come in at about its waveform's value at that time; an outlet
    should let flow out, which in the first steps from rest every outlet does; a wall should let
    nothing through, which with its no-slip condition it cannot.
    """
    row = integrals.at(step)
    if row is None:
        return []
    _step, time, values = row
    flows = []
    for name, value in zip(integrals.names, values):
        condition = conditions_by_name.get(name)
        if isinstance(condition, bcs.Inflow):
            flows.append(FaceFlow(name, -value, "in", _inflow_at(condition, time)))
        elif condition is not None:
            flows.append(FaceFlow(name, -value, "out"))
        else:
            flows.append(FaceFlow(name, -value, "none"))
    return flows


def flow_lines(flows, time=None) -> list:
    """The flows as sentences, the faces going the wrong way first."""
    if not flows:
        return ["The solver wrote no flow through the faces (Outputs: Flow per face is off)."]
    wrong = [flow for flow in flows if not flow.right_way]
    walls = [flow for flow in flows if flow.expected == "none"]
    inflow = sum(flow.flow for flow in flows if flow.expected == "in" and flow.flow > 0)
    leaky = [flow for flow in walls if inflow and abs(flow.flow) > 1e-3 * inflow]
    head = (f"At {time:g} s: " if time is not None else "")
    if wrong or leaky:
        head += "the flow goes the wrong way through " + ", ".join(f.name for f in wrong + leaky) + "."
    else:
        head += "the flow comes in through every inflow and leaves through every outlet."
    return [head] + [flow.describe() for flow in wrong + leaky] + \
        [flow.describe() for flow in flows if flow not in wrong and flow not in leaky and flow.expected != "none"]


# -- the velocity, drawn ----------------------------------------------------------------------------
VELOCITY_ARRAY = "Velocity"
SPEED_ARRAY = "Speed"


def velocity_arrows(grid, count=3000, arrow_length=None, seed=1) -> vtk.vtkPolyData:
    """Arrows along the velocity at about `count` points of a result, as long as the speed.

    A result's coordinates are the mesh's own, in millimetres, and its velocity the solver's, in
    cm/s, so the arrows are scaled to the mesh: the fastest a twentieth of its size, unless an
    `arrow_length` in mm per cm/s is given. Each carries its speed, to be coloured by.
    """
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    velocity = grid.GetPointData().GetArray(VELOCITY_ARRAY)
    if velocity is None:
        raise ValueError(f"The result has no {VELOCITY_ARRAY}: Outputs: Velocity is off.")
    points = vtk.vtkPolyData()
    points.SetPoints(grid.GetPoints())
    points.GetPointData().AddArray(velocity)
    # Spread through the volume, stratified in space: taken in order, the first `count` points
    # are one end of the vessel, and a randomised stride can give one more than asked for.
    mask = vtk.vtkMaskPoints()
    mask.SetInputData(points)
    mask.SetMaximumNumberOfPoints(count)
    mask.RandomModeOn()
    mask.SetRandomModeType(2)
    mask.SetRandomSeed(seed)
    mask.GenerateVerticesOn()
    mask.Update()
    sample = vtk.vtkPolyData()
    sample.DeepCopy(mask.GetOutput())
    sampled = vtk_to_numpy(sample.GetPointData().GetArray(VELOCITY_ARRAY))
    speed = numpy_to_vtk(np.linalg.norm(sampled, axis=1).astype(float), deep=True)
    speed.SetName(SPEED_ARRAY)
    sample.GetPointData().AddArray(speed)
    sample.GetPointData().SetActiveScalars(SPEED_ARRAY)
    sample.GetPointData().SetActiveVectors(VELOCITY_ARRAY)
    fastest = float(np.linalg.norm(vtk_to_numpy(velocity), axis=1).max()) if velocity.GetNumberOfTuples() else 0.0
    if arrow_length is None:
        bounds = grid.GetBounds()
        diagonal = math.dist(bounds[0::2], bounds[1::2])
        arrow_length = (diagonal / 20.0) / fastest if fastest > 0 else 1.0
    # Held: a source made in the call would be collected before the glyph filter runs it.
    arrow = vtk.vtkArrowSource()
    glyph = vtk.vtkGlyph3D()
    glyph.SetInputData(sample)
    glyph.SetSourceConnection(arrow.GetOutputPort())
    glyph.SetVectorModeToUseVector()
    glyph.SetScaleModeToScaleByVector()
    glyph.SetColorModeToColorByScalar()
    glyph.SetScaleFactor(arrow_length)
    glyph.OrientOn()
    glyph.Update()
    arrows = vtk.vtkPolyData()
    arrows.DeepCopy(glyph.GetOutput())
    scalars = arrows.GetPointData().GetScalars()
    if scalars is not None:
        scalars.SetName(SPEED_ARRAY)
        arrows.GetPointData().SetActiveScalars(SPEED_ARRAY)
    return arrows


def read_result(path) -> vtk.vtkUnstructuredGrid:
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = vtk.vtkUnstructuredGrid()
    grid.DeepCopy(reader.GetOutput())
    if grid.GetNumberOfPoints() == 0:
        raise ValueError(f"{path} holds no mesh.")
    return grid
