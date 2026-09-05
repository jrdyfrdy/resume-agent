"""Write the artifacts and the run record. Spec 5, `finalize`.

    "Copy artifacts to `out/{company}_{role}_{date}/`, write `run.json` (JobSpec,
     fit report, bullet IDs used, model versions, token cost, git SHA of the
     profile), insert a tracker row.

     That `run.json` is your future dataset. Six months from now you'll want to
     ask 'which bullets appear in applications that got callbacks?' -- you can
     only answer that if you logged it from the start."

The tracker database is M7. `run.json` is written now, because the argument for
it is entirely about starting early: a record you begin keeping in six months
tells you nothing about the six months you already spent applying.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

from resume_agent.graph.state import AgentState
from resume_agent.kb.loader import load_profile
from resume_agent.latex.compile import compile_tex
from resume_agent.latex.env import render_template
from resume_agent.llm import JUDGE_MODEL, PARSE_MODEL, prompt_version

logger = logging.getLogger(__name__)

# Prompts whose versions go into run.json. Changing any of them changes the
# output, so a run record that did not name them could not be reproduced.
TRACKED_PROMPTS = [
    "parse_jd",
    "score_fit",
    "tailor_bullets",
    "verify_grounding",
    "write_cover_letter",
    "verify_letter",
]

COVER_LETTER_TEMPLATE = "cover_letter.tex.j2"


def slugify(text: str) -> str:
    """`Halvorsen & Bright R&D` -> `halvorsen-bright-rd`, safe as a directory name."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "unknown"


def profile_git_sha(profile_path: Path) -> str | None:
    """The profile's git revision, when it is under version control.

    Spec 5 asks for this so a run can be tied to the exact career data that
    produced it. Returns None rather than raising when the profile is not in a
    repository -- which is the normal case for `profile/`, since it is gitignored.
    """
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no user input
            ["git", "rev-parse", "HEAD"],
            cwd=profile_path if profile_path.is_dir() else profile_path.parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def finalize(state: AgentState) -> dict:
    """Copy the artifacts into a named run directory and write `run.json`."""
    options = state["options"]
    job = state.get("job_spec")

    company = slugify(job.company) if job else "unknown"
    role = slugify(job.title) if job else "unknown"
    run_dir = Path(options.out_dir) / f"{company}_{role}_{date.today().isoformat()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if state.get("tex_source"):
        (run_dir / "resume.tex").write_text(state["tex_source"], encoding="utf-8")
    if state.get("compile_log"):
        (run_dir / "compile.log").write_text(state["compile_log"], encoding="utf-8")

    pdf_path = state.get("pdf_path")
    final_pdf = None
    if pdf_path and Path(pdf_path).is_file():
        final_pdf = run_dir / "resume.pdf"
        shutil.copyfile(pdf_path, final_pdf)

    letter = state.get("cover_letter")
    if letter is not None:
        _write_cover_letter(state, letter, run_dir)

    fit = state.get("fit_report")
    run_record = {
        "date": date.today().isoformat(),
        "job_spec": job.model_dump() if job else None,
        "fit_report": fit.model_dump() if fit else None,
        # The two lists that answer "which bullets did I actually send?" and
        # "which ones did the verifier refuse to let through?".
        "selected_bullet_ids": state.get("selected", []),
        "tailored_bullets": [t.model_dump() for t in state.get("tailored", [])],
        "dropped_bullets": state.get("dropped_bullets", []),
        "cover_letter": letter.model_dump() if letter else None,
        "letter_attempts": state.get("letter_attempts", 0),
        "models": {"generation": PARSE_MODEL, "judge": JUDGE_MODEL},
        "prompt_versions": {name: prompt_version(name) for name in TRACKED_PROMPTS},
        "profile_git_sha": profile_git_sha(Path(state["profile_path"])),
        "line_budget": state.get("line_budget"),
        "page_count": state.get("page_count"),
        "grounding_attempts": state.get("grounding_attempts", 0),
        "layout_attempts": state.get("layout_attempts", 0),
        "critiques": state.get("critiques", []),
        "errors": state.get("errors", []),
    }
    (run_dir / "run.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")

    logger.info("finalize: wrote %s", run_dir)
    return {
        "out_dir": str(run_dir),
        "pdf_path": str(final_pdf) if final_pdf else state.get("pdf_path"),
    }


def _write_cover_letter(state: AgentState, letter, run_dir: Path) -> None:
    """Render and compile the letter into the run directory.

    A failed letter compile is recorded, not fatal. The resume is the artifact
    the run exists to produce, and the letter has no layout loop behind it --
    its length is already enforced by word count before it ever reaches LaTeX,
    so there is no feedback for a compile failure to drive.
    """
    profile = load_profile(Path(state["profile_path"]))
    job = state.get("job_spec")

    tex_source = render_template(
        COVER_LETTER_TEMPLATE,
        {
            "identity": {
                "name": profile.identity.name,
                "email": profile.identity.email,
                "phone": profile.identity.phone,
                "location": profile.identity.location,
            },
            "company": job.company if job else "Unknown",
            "date": date.today().strftime("%d %B %Y"),
            "paragraphs": letter.paragraphs,
        },
    )
    (run_dir / "cover_letter.tex").write_text(tex_source, encoding="utf-8")

    result = compile_tex(tex_source, run_dir / "work", job_name="cover_letter")
    if result.ok and result.pdf_path:
        shutil.copyfile(result.pdf_path, run_dir / "cover_letter.pdf")
        logger.info("finalize: cover letter written (%d words)", letter.word_count)
    else:
        (run_dir / "cover_letter.log").write_text(result.log, encoding="utf-8")
        logger.error("finalize: cover letter did not compile; see cover_letter.log")
