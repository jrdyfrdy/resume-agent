"""Turn a `.tex` string into a PDF, using whatever compiler is on this machine.

Spec 6.4 and spec 5 (`compile_pdf`).

Two rules govern this module, and both come from CLAUDE.md:

1. **Never raise on a compile failure.** A broken document is *data* -- it is the
   feedback signal that M5's repair loop consumes. Swallowing the log, or
   letting an exception escape, destroys the only ground truth this system has
   about whether the LaTeX is correct.
2. **Fail with an install message, not a stack trace,** when no compiler exists.

The fallback chain is tectonic -> latexmk -> Docker, in that order, because that
is also the order of decreasing reproducibility: tectonic pins its own package
versions, a local TeX Live install is whatever the user happens to have, and
Docker is correct but slow enough to make an iterative repair loop unpleasant.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CompilerName = Literal["tectonic", "latexmk", "docker"]

# Generous: a cold tectonic run downloads its package bundle over the network.
# Warm runs on this template take about a second.
COMPILE_TIMEOUT_SECONDS = 300

DOCKER_IMAGE = "texlive/texlive"

# Env var escape hatch, mostly for tests: it lets a test point the resolver at a
# path that does not exist and assert the no-compiler branch without uninstalling
# anything.
TECTONIC_ENV_VAR = "RESUME_AGENT_TECTONIC"

INSTALL_MESSAGE = """\
No LaTeX compiler found. resume-agent needs one of these:

  tectonic  (recommended -- single binary, fetches only the packages it needs)
      Windows : download tectonic-<version>-x86_64-pc-windows-msvc.zip from
                https://github.com/tectonic-typesetting/tectonic/releases
                and extract tectonic.exe to %LOCALAPPDATA%\\Programs\\tectonic\\
      macOS   : brew install tectonic
      Linux   : cargo install tectonic   (or use the release binary)

  latexmk   (part of a full TeX Live / MiKTeX install)

  docker    (slowest; resume-agent will run the texlive/texlive image)

Already installed somewhere unusual? Point resume-agent at it directly:
  set RESUME_AGENT_TECTONIC=C:\\path\\to\\tectonic.exe
"""


@dataclass(frozen=True)
class CompilerCommand:
    """A located compiler: which one it is, and how to start it."""

    name: CompilerName
    argv: list[str]  # the executable prefix; per-run arguments are added later


@dataclass(frozen=True)
class CompileResult:
    """The outcome of one compile. Always returned, never raised."""

    ok: bool
    pdf_path: Path | None
    log: str
    compiler: CompilerName | None
    returncode: int | None


# --- locating a compiler ---------------------------------------------------


def _tectonic_candidates() -> list[Path]:
    """Where a tectonic binary might live, most explicit first."""
    candidates: list[Path] = []

    override = os.environ.get(TECTONIC_ENV_VAR)
    if override:
        candidates.append(Path(override))

    on_path = shutil.which("tectonic")
    if on_path:
        candidates.append(Path(on_path))

    # The install location this project's setup instructions use on Windows.
    # tectonic ships as a bare .exe with no installer, so it is common for it to
    # be present but not on PATH -- worth looking before giving up.
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "Programs" / "tectonic" / "tectonic.exe")

    # The equivalent convention on macOS/Linux.
    candidates.append(Path.home() / ".local" / "bin" / "tectonic")

    return candidates


def find_compiler() -> CompilerCommand | None:
    """Return the first available compiler in the spec 6.4 fallback chain.

    Returns `None` if nothing is installed; callers surface `INSTALL_MESSAGE`.
    """
    for candidate in _tectonic_candidates():
        if candidate.is_file():
            return CompilerCommand(name="tectonic", argv=[str(candidate)])

    latexmk = shutil.which("latexmk")
    if latexmk:
        return CompilerCommand(name="latexmk", argv=[latexmk])

    docker = shutil.which("docker")
    if docker:
        return CompilerCommand(name="docker", argv=[docker])

    return None


# --- running it ------------------------------------------------------------


def _build_argv(command: CompilerCommand, tex_path: Path, out_dir: Path) -> list[str]:
    """Compose the full command line for one compile."""
    if command.name == "tectonic":
        return [
            *command.argv,
            # --keep-logs is not optional for us: overfull hbox warnings only
            # appear in the .log file, and `inspect_output` needs them.
            "--keep-logs",
            "--print",
            "--chatter",
            "minimal",
            "--outdir",
            str(out_dir),
            str(tex_path),
        ]

    if command.name == "latexmk":
        return [
            *command.argv,
            "-pdf",
            # Without nonstopmode a LaTeX error opens an interactive prompt and
            # the subprocess hangs until the timeout instead of returning a log.
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={out_dir}",
            str(tex_path),
        ]

    # Docker: mount the output directory (which already contains the .tex) and
    # run latexmk inside it. Relative paths only, since /work is the container's
    # view of out_dir.
    return [
        *command.argv,
        "run",
        "--rm",
        "-v",
        f"{out_dir.resolve()}:/work",
        "-w",
        "/work",
        DOCKER_IMAGE,
        "latexmk",
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]


def compile_tex(tex_source: str, out_dir: Path, job_name: str = "resume") -> CompileResult:
    """Write `tex_source` to `out_dir/job_name.tex` and compile it.

    Never raises. Every failure mode -- no compiler, a TeX error, a timeout, a
    missing PDF -- comes back as a `CompileResult` with `ok=False` and a log.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tex_path = out_dir / f"{job_name}.tex"
    tex_path.write_text(tex_source, encoding="utf-8")

    command = find_compiler()
    if command is None:
        return CompileResult(
            ok=False, pdf_path=None, log=INSTALL_MESSAGE, compiler=None, returncode=None
        )

    argv = _build_argv(command, tex_path, out_dir)

    try:
        completed = subprocess.run(  # noqa: S603 - argv is built here, never user input
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=COMPILE_TIMEOUT_SECONDS,
            check=False,
        )
        returncode: int | None = completed.returncode
        console = completed.stdout + completed.stderr
    except subprocess.TimeoutExpired:
        returncode = None
        console = (
            f"[resume-agent] {command.name} exceeded "
            f"{COMPILE_TIMEOUT_SECONDS}s and was killed.\n"
        )
    except OSError as exc:
        # The binary existed a moment ago but could not be executed (permissions,
        # wrong architecture). Still not an exception the caller has to handle.
        returncode = None
        console = f"[resume-agent] could not run {command.name}: {exc}\n"

    log = _collect_log(out_dir, job_name, console)

    pdf_path = out_dir / f"{job_name}.pdf"
    # The return code alone is not trustworthy: latexmk exits 0 on some runs that
    # produced no PDF, and tectonic can emit warnings with a zero status. The PDF
    # existing on disk is the real success condition.
    ok = returncode == 0 and pdf_path.is_file()

    return CompileResult(
        ok=ok,
        pdf_path=pdf_path if pdf_path.is_file() else None,
        log=log,
        compiler=command.name,
        returncode=returncode,
    )


def _collect_log(out_dir: Path, job_name: str, console: str) -> str:
    """Join the engine's console output with the `.log` file it wrote.

    Both halves matter and they carry different things: `! ` error lines and the
    "Overfull \\hbox" warnings live in the .log file, while a compiler that failed
    to start at all only ever speaks on stderr.
    """
    parts = [f"===== {job_name}: compiler console =====", console.rstrip()]

    log_file = out_dir / f"{job_name}.log"
    if log_file.is_file():
        parts.append(f"===== {job_name}.log =====")
        # LaTeX logs are latin-1-ish and occasionally contain raw bytes; replace
        # rather than fail, since a log we cannot decode is still better than
        # no log at all.
        parts.append(log_file.read_text(encoding="utf-8", errors="replace").rstrip())

    return "\n".join(parts) + "\n"
