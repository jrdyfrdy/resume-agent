"""The eval harness. Spec 9, and M8's definition of done.

    uv run python evals/run_eval.py                       # baseline
    uv run python evals/run_eval.py --variant worsened    # sabotaged prompt
    uv run python evals/run_eval.py --check-against evals/baselines/main.json

Spec 9 on why this exists at all:

    "Then make prompt changes gated: a PR that drops mean judge score by >0.3
     doesn't merge. This is the habit that separates AI engineers from people
     who tweak prompts until the vibes improve."

So the deliverable is not the table. It is `--check-against` exiting non-zero,
and the demonstration that a deliberately worsened prompt makes it do so.

**LangSmith.** Spec 9 says to wire this to LangSmith datasets and `evaluate()`.
CLAUDE.md lists langsmith as "optional, env-gated", and doing it unconditionally
would put a second account and API key in front of a harness that works fine
writing JSON locally. `EvalReport.to_dict()` is the shape a LangSmith dataset
row wants; attaching it is a small function and a decision about accounts, not a
redesign.

**Cost.** A full 15-JD run generates fifteen resumes: parse, score, tailor,
judge. Roughly $10-15 the first time and near-free afterwards, because M5's
caching keys on the posting and the prompt hash. `--limit` exists for that
reason. The deterministic half is free always.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.checks import DeterministicResult, run_deterministic_checks  # noqa: E402
from evals.judges.resume_judge import JudgeScores, judge_resume  # noqa: E402
from resume_agent.graph.build import build_graph, initial_state  # noqa: E402
from resume_agent.graph.state import RunOptions  # noqa: E402
from resume_agent.llm import PROMPT_OVERRIDE_ENV_VAR, has_credentials  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
JD_DIR = REPO_ROOT / "evals" / "datasets" / "jds"
RESULTS_DIR = REPO_ROOT / "evals" / "results"
BASELINE_DIR = REPO_ROOT / "evals" / "baselines"
VARIANT_DIR = REPO_ROOT / "evals" / "variants"

# Spec 9: "a PR that drops mean judge score by >0.3 doesn't merge".
REGRESSION_THRESHOLD = 0.3

# Spec 9: "Human spot-check: 3 resumes per eval run, reviewed by you."
SPOT_CHECK_COUNT = 3

# Prompt-name -> file, per variant. A variant is a real prompt file on disk,
# not a string patched in at runtime, so the thing being measured is the thing
# a person would actually commit.
VARIANTS: dict[str, dict[str, Path]] = {
    "worsened": {"tailor_bullets": VARIANT_DIR / "tailor_bullets.worsened.md"},
}


@dataclass
class CaseResult:
    """One JD's outcome."""

    jd_name: str
    deterministic: DeterministicResult
    judge: JudgeScores | None = None
    error: str | None = None
    # M11: who scored the evidence, how long scoring took, and what it led to --
    # the comparison `--compare` makes between `--decisions llm` and `jev`.
    scoring: dict = field(default_factory=dict)
    covered: int | None = None
    selected: list[str] = field(default_factory=list)

    @property
    def mean_score(self) -> float | None:
        return self.judge.mean if self.judge else None


@dataclass
class EvalReport:
    variant: str
    ran_at: str
    cases: list[CaseResult] = field(default_factory=list)
    decisions: str = "llm"

    @property
    def scored(self) -> list[CaseResult]:
        return [c for c in self.cases if c.judge is not None]

    @property
    def mean_judge_score(self) -> float:
        """The single number the gate compares. 0.0 when nothing scored."""
        scores = [c.judge.mean for c in self.scored]
        return round(statistics.fmean(scores), 3) if scores else 0.0

    @property
    def deterministic_pass_rate(self) -> float:
        if not self.cases:
            return 0.0
        return round(sum(1 for c in self.cases if c.deterministic.passed) / len(self.cases), 3)

    def dimension_means(self) -> dict[str, float]:
        from evals.judges.resume_judge import DIMENSIONS

        if not self.scored:
            return dict.fromkeys(DIMENSIONS, 0.0)
        return {
            d: round(statistics.fmean(getattr(c.judge, d) for c in self.scored), 3)
            for d in DIMENSIONS
        }

    def to_dict(self) -> dict:
        return {
            "variant": self.variant,
            "decisions": self.decisions,
            "ran_at": self.ran_at,
            "mean_judge_score": self.mean_judge_score,
            "deterministic_pass_rate": self.deterministic_pass_rate,
            "dimension_means": self.dimension_means(),
            "cases": [
                {
                    "jd_name": c.jd_name,
                    "deterministic": asdict(c.deterministic),
                    "judge": c.judge.as_dict() if c.judge else None,
                    "reasoning": c.judge.reasoning if c.judge else None,
                    "error": c.error,
                    "scoring": c.scoring,
                    "covered": c.covered,
                    "selected": c.selected,
                }
                for c in self.cases
            ],
        }


# ---------------------------------------------------------------------------


def run_eval(
    jd_paths: list[Path],
    profile_dir: Path,
    *,
    variant: str | None = None,
    out_dir: Path | None = None,
    judge: bool = True,
    decisions: str = "llm",
) -> EvalReport:
    """Run every JD through the graph, then score the result.

    `decisions="jev"` scores the evidence with Jev instead of the model (M11
    J3); everything else about the run is the same, so `--compare` can put the
    two side by side.
    """
    _choose_scorer(decisions)
    if variant:
        overrides = VARIANTS.get(variant)
        if overrides is None:
            raise SystemExit(f"unknown variant {variant!r}; known: {sorted(VARIANTS)}")
        os.environ[PROMPT_OVERRIDE_ENV_VAR] = ",".join(
            f"{name}={path}" for name, path in overrides.items()
        )

    report = EvalReport(
        variant=variant or "baseline",
        ran_at=datetime.now().isoformat(timespec="seconds"),
        decisions=decisions,
    )
    graph = build_graph()
    out_dir = out_dir or (RESULTS_DIR / "runs")

    for path in jd_paths:
        name = path.stem
        options = RunOptions(
            out_dir=str(out_dir / name),
            # The letter is scored by its own checks in M6; the resume judge has
            # nothing to say about it, and 15 extra draft-plus-judge cycles would
            # roughly double the cost of a run for no signal here.
            write_cover_letter=False,
        )
        try:
            state = graph.invoke(
                initial_state(path.read_text(encoding="utf-8"), profile_dir, options),
                {"recursion_limit": 120},
            )
        except Exception as exc:  # noqa: BLE001 - one bad JD must not end the run
            report.cases.append(
                CaseResult(
                    jd_name=name,
                    deterministic=DeterministicResult(
                        jd_name=name,
                        compiled=False,
                        page_count=None,
                        overfull_boxes=0,
                        first_error=str(exc),
                    ),
                    error=str(exc),
                )
            )
            print(f"  {name}: FAILED ({exc})")
            continue

        deterministic = run_deterministic_checks(name, state)
        scores = (
            judge_resume(state["job_spec"], state.get("tailored", []))
            if judge and state.get("job_spec")
            else None
        )
        fit = state.get("fit_report")
        report.cases.append(
            CaseResult(
                jd_name=name,
                deterministic=deterministic,
                judge=scores,
                scoring=state.get("scoring") or {},
                covered=len(fit.covered) if fit else None,
                selected=list(state.get("selected", [])),
            )
        )
        print(
            f"  {name}: {'pass' if deterministic.passed else 'FAIL'}"
            f"{f' | judge {scores.mean:.2f}' if scores else ''}"
            f" | scored by {(state.get('scoring') or {}).get('by', '?')}"
        )

    return report


def _choose_scorer(decisions: str) -> None:
    """Point the `score` node at the model or at Jev for this process."""
    from resume_agent import decisions as jev  # noqa: PLC0415

    if decisions not in ("llm", "jev"):
        raise SystemExit(f"--decisions must be llm or jev, not {decisions!r}")
    if decisions == "jev":
        if not jev.jev_enabled():
            raise SystemExit(f"--decisions jev needs {jev.API_KEY_ENV_VAR} set.")
        os.environ[jev.SCORING_ENV_VAR] = "1"
    else:
        os.environ.pop(jev.SCORING_ENV_VAR, None)


def compare(before: dict, after: dict) -> str:
    """Two saved results side by side: the evidence `--decisions jev` has to give.

    Scoring changes what gets *selected*, so the question is not only whether
    the judge score holds but whether the same achievements go out, and whether
    as many requirements are covered.
    """
    by_name = {c["jd_name"]: c for c in before["cases"]}
    lines = [
        "",
        f"{before.get('decisions', '?')} -> {after.get('decisions', '?')}",
        "",
        f"{'jd':<28}{'covered':>9}{'same picks':>12}{'judge':>13}{'scoring s':>13}",
        "-" * 75,
    ]
    for case in after["cases"]:
        old = by_name.get(case["jd_name"])
        if old is None:
            continue
        a, b = set(old.get("selected", [])), set(case.get("selected", []))
        overlap = f"{len(a & b) / len(a | b):.0%}" if a | b else "-"
        judge_a = (old.get("judge") or {}).get("mean")
        judge_b = (case.get("judge") or {}).get("mean")
        judge = f"{judge_a:.2f}->{judge_b:.2f}" if judge_a and judge_b else "-"
        secs = [(c.get("scoring") or {}) for c in (old, case)]
        seconds = "->".join(
            "cached" if s.get("cached") else f"{s.get('seconds', 0):.1f}" for s in secs
        )
        lines.append(
            f"{case['jd_name']:<28}{old.get('covered')!s:>4}->{case.get('covered')!s:<4}"
            f"{overlap:>12}{judge:>13}{seconds:>13}"
        )
    lines += [
        "-" * 75,
        f"mean judge score: {before['mean_judge_score']:.3f} -> {after['mean_judge_score']:.3f}",
        "",
        "Switch the default only if the judge score holds within 0.1 and coverage "
        "does not drop (M11 J3). For a fair time, run each with an empty "
        "RESUME_AGENT_CACHE_DIR.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------


def render_table(report: EvalReport) -> str:
    """The table `make eval` prints."""
    lines = [
        "",
        f"eval: {report.variant}   ({report.ran_at})",
        "",
        f"{'jd':<28}{'det':>5}{'rel':>5}{'spec':>5}{'ats':>5}{'tone':>5}{'filler':>7}{'mean':>7}",
        "-" * 67,
    ]
    for case in report.cases:
        det = "ok" if case.deterministic.passed else "FAIL"
        if case.judge:
            j = case.judge
            lines.append(
                f"{case.jd_name:<28}{det:>5}{j.relevance:>5}{j.specificity:>5}"
                f"{j.ats_alignment:>5}{j.tone_match:>5}{j.absence_of_filler:>7}"
                f"{j.mean:>7.2f}"
            )
        else:
            lines.append(
                f"{case.jd_name:<28}{det:>5}{'-':>5}{'-':>5}{'-':>5}{'-':>5}{'-':>7}{'-':>7}"
            )

    means = report.dimension_means()
    lines += [
        "-" * 67,
        f"{'mean':<28}{'':>5}{means['relevance']:>5.1f}{means['specificity']:>5.1f}"
        f"{means['ats_alignment']:>5.1f}{means['tone_match']:>5.1f}"
        f"{means['absence_of_filler']:>7.1f}{report.mean_judge_score:>7.2f}",
        "",
        f"deterministic pass rate : {report.deterministic_pass_rate:.0%}",
        f"mean judge score        : {report.mean_judge_score:.3f}",
    ]

    failures = [c for c in report.cases if not c.deterministic.passed]
    if failures:
        lines.append("")
        lines.append("deterministic failures:")
        for case in failures:
            lines.append(f"  {case.jd_name}: {'; '.join(case.deterministic.failures)}")

    # Spec 9: "Human spot-check: 3 resumes per eval run, reviewed by you.
    # Judges drift; your eye is the calibration set."
    if report.scored:
        worst = sorted(report.scored, key=lambda c: c.judge.mean)[:SPOT_CHECK_COUNT]
        lines += ["", f"spot-check these {len(worst)} by hand (lowest scoring):"]
        for case in worst:
            lines.append(f"  {case.jd_name} ({case.judge.mean:.2f}) -> {case.judge.reasoning[:90]}")

    return "\n".join(lines) + "\n"


def check_regression(
    report: EvalReport, baseline: dict, threshold: float = REGRESSION_THRESHOLD
) -> tuple[bool, str]:
    """Spec 9's gate. Returns `(ok, message)`."""
    before = baseline.get("mean_judge_score", 0.0)
    after = report.mean_judge_score
    delta = round(after - before, 3)

    if delta < -threshold:
        return False, (
            f"REGRESSION: mean judge score {before:.3f} -> {after:.3f} ({delta:+.3f}), "
            f"which exceeds the {threshold} threshold."
        )
    direction = "improved" if delta > 0 else "held" if delta == 0 else "slipped"
    return True, f"mean judge score {before:.3f} -> {after:.3f} ({delta:+.3f}); {direction}."


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the eval set. Spec 9.")
    parser.add_argument("--profile", default="profile.example")
    parser.add_argument("--variant", default=None, help=f"one of {sorted(VARIANTS)}")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N JDs")
    parser.add_argument("--no-judge", action="store_true", help="deterministic checks only (free)")
    parser.add_argument("--check-against", default=None, help="baseline JSON to gate against")
    parser.add_argument("--save-baseline", default=None, help="write the result as a baseline")
    parser.add_argument(
        "--decisions", default="llm", choices=["llm", "jev"],
        help="who scores the evidence: the model (default) or Jev (M11)",
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("BEFORE", "AFTER"), default=None,
        help="print two saved results side by side and stop",
    )
    args = parser.parse_args()

    if args.compare:
        before, after = (json.loads(Path(p).read_text(encoding="utf-8")) for p in args.compare)
        print(compare(before, after))
        return

    if not args.no_judge and not has_credentials():
        raise SystemExit(
            "This needs model credentials (see `resume-agent check-credentials`). "
            "Run with --no-judge for the free "
            "deterministic checks only (they still need cached runs to check)."
        )

    jd_paths = sorted(JD_DIR.glob("*.txt"))[: args.limit]
    print(f"Running {len(jd_paths)} JDs ({args.variant or 'baseline'})...")

    report = run_eval(
        jd_paths, Path(args.profile), variant=args.variant, judge=not args.no_judge,
        decisions=args.decisions,
    )
    print(render_table(report))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_path = RESULTS_DIR / f"{report.variant}-{report.decisions}-{stamp}.json"
    results_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"results: {results_path}")

    if args.save_baseline:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        Path(args.save_baseline).write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        print(f"baseline written: {args.save_baseline}")

    if args.check_against:
        baseline = json.loads(Path(args.check_against).read_text(encoding="utf-8"))
        ok, message = check_regression(report, baseline)
        print(message)
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
