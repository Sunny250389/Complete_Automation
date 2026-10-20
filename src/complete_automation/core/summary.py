"""Markdown summary of shared-core JSON reports, for CI job summaries and terminals.

    python -m complete_automation.core.summary reports/<run_id>/results*.json [--output FILE]

It reads what :class:`~complete_automation.core.reporting.JsonReporter` wrote, rebuilds every
:class:`TestResult` through the result contract and recounts with :func:`summarize`. It computes no
metric and changes no verdict. Several files (pytest-xdist workers, separate CI jobs) are combined
into one summary, each module counted on its own row.

Results that did not pass are grouped by what went wrong, from the result's ``status`` and
``failure.category``: quality gates (``threshold``), assertions, services and infrastructure
(``infrastructure``, ``timeout``), other errors, incomplete, and skips grouped by their reason.

Exit status: 0 once the reports were read (pytest's own exit status says whether the run failed),
2 when no report was found or a file is not a valid report.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import ValidationError

from complete_automation.core.reporting import summarize
from complete_automation.core.results import Status, TestResult

_MESSAGE_LIMIT = 160
_ARTIFACT_LIMIT = 3
_GROUPS = (
    ("Quality-gate failures", lambda r: r.status is Status.FAILED and _category(r) == "threshold"),
    ("Assertion failures", lambda r: r.status is Status.FAILED and _category(r) != "threshold"),
    ("Service or infrastructure errors", lambda r: r.status is Status.ERROR and _category(r) in ("infrastructure", "timeout")),
    ("Other errors", lambda r: r.status is Status.ERROR and _category(r) not in ("infrastructure", "timeout")),
    ("Incomplete (no verdict)", lambda r: r.status is Status.INCOMPLETE),
)  # fmt: skip


class ReportError(Exception):
    """A report file is missing or is not a shared-core JSON report."""


def _category(result: TestResult) -> str:
    return result.failure.category.value if result.failure else ""


def _cell(text: object) -> str:
    """One Markdown table cell: a single line, pipes escaped."""
    return " ".join(str(text).split()).replace("|", "\\|")


def load_reports(paths: Iterable[Path]) -> tuple[list[dict[str, object]], list[TestResult]]:
    """The ``run`` sections and every result of the given report files."""
    runs: list[dict[str, object]] = []
    results: list[TestResult] = []
    for path in paths:
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
            runs.append(document["run"])
            results += [TestResult.from_dict(item) for item in document["results"]]
        except (OSError, ValueError, KeyError, TypeError, ValidationError) as error:
            raise ReportError(f"{path}: not a readable shared-core report ({type(error).__name__})") from None
    return runs, results


def render_markdown(runs: Sequence[dict[str, object]], results: Sequence[TestResult], *, title: str = "Test results") -> str:
    summary = summarize(results)
    lines = [f"## {title}", ""]
    for run in runs:
        lines.append(f"Run `{run.get('run_id')}`, environment `{run.get('environment')}`")
    lines += [
        "",
        "| Total | Passed | Failed | Error | Skipped | Incomplete | Duration (s) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {summary.total} | {summary.passed} | {summary.failed} | {summary.error} | {summary.skipped} "
        f"| {summary.incomplete} | {summary.duration_seconds:.1f} |",
        "",
        "| Module | Total | Passed | Failed | Error | Skipped | Incomplete |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for module, counts in summary.by_module.items():
        lines.append(
            f"| {_cell(module)} | {counts.total} | {counts.passed} | {counts.failed} | {counts.error} "
            f"| {counts.skipped} | {counts.incomplete} |"
        )
    for heading, belongs in _GROUPS:
        group = [result for result in results if belongs(result)]
        if group:
            lines += ["", f"### {heading} ({len(group)})", ""] + _failure_table(group)
    skipped = [result for result in results if result.status is Status.SKIPPED]
    if skipped:
        lines += ["", f"### Skipped ({len(skipped)})", "", "| Count | Modules | Reason |", "| ---: | --- | --- |"]
        reasons: dict[str, list[TestResult]] = {}
        for result in skipped:
            reasons.setdefault(str(result.metadata.get("skip_reason", "skipped")), []).append(result)
        for reason, group in sorted(reasons.items(), key=lambda item: (-len(item[1]), item[0])):
            modules = ", ".join(sorted({result.module for result in group}))
            lines.append(f"| {len(group)} | {_cell(modules)} | {_cell(reason)} |")
    return "\n".join(lines) + "\n"


def _failure_table(results: Sequence[TestResult]) -> list[str]:
    lines = ["| Module | Test | Category | Duration (s) | Message | Failed metrics | Artifacts |",
             "| --- | --- | --- | ---: | --- | --- | --- |"]  # fmt: skip
    for result in results:
        message = result.failure.message if result.failure else ""
        if len(message) > _MESSAGE_LIMIT:
            message = message[: _MESSAGE_LIMIT - 1] + "…"
        metrics = ", ".join(
            f"{metric.name}={metric.value}" + (f" (threshold {metric.threshold})" if metric.threshold is not None else "")
            for metric in result.metrics
            if metric.passed is False
        )
        paths = [artifact.path for artifact in result.artifacts]
        artifacts = ", ".join(paths[:_ARTIFACT_LIMIT]) + (f" (+{len(paths) - _ARTIFACT_LIMIT})" if len(paths) > _ARTIFACT_LIMIT else "")
        lines.append(
            f"| {_cell(result.module)} | `{_cell(result.test_id)}` | {_cell(_category(result))} "
            f"| {result.duration_seconds:.2f} | {_cell(message)} | {_cell(metrics)} | {_cell(artifacts)} |"
        )
    return lines


def _expand(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True)) if glob.has_magic(pattern) else [pattern]
        paths += [Path(match) for match in matches]
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m complete_automation.core.summary", description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+", help="results.json files or glob patterns (quoted)")
    parser.add_argument("--output", type=Path, help="append the summary to this file instead of printing it")
    parser.add_argument("--title", default="Test results")
    args = parser.parse_args(argv)
    paths = _expand(args.reports)
    if not paths:
        print(f"no report matched {args.reports}", file=sys.stderr)
        return 2
    try:
        runs, results = load_reports(paths)
    except ReportError as error:
        print(error, file=sys.stderr)
        return 2
    text = render_markdown(runs, results, title=args.title)
    if args.output is None:
        sys.stdout.write(text)
    else:
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
