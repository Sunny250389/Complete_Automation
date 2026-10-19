"""Quality gates, suite aggregates and baseline comparison.

A rule whose metric is missing fails (``metric_unavailable``); it never passes by default.
Judge errors and cases without a response are counted as *not passed* in every pass rate, and
are excluded from means, so an error can never raise a score.

Aggregate metric names (``<metric>`` is a check name or ``judge.<criterion>``):

* ``<metric>.pass_rate``: passed / cases that declared the metric (errors and not-evaluated count
  as not passed); only for metrics with a pass/fail rule;
* ``<metric>.mean``: mean value over cases that produced one (judge scores, lengths, ratios);
* ``<metric>.errors``: number of cases where the evaluator could not reach a verdict;
* ``case_pass_rate``, ``model_error_rate``, ``evaluator_error_rate``;
* ``latency_seconds.mean`` / ``.max`` and ``output_tokens.mean`` over responses that provide them.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping

from complete_automation.llm_evaluation.errors import BaselineError
from complete_automation.llm_evaluation.schemas import (
    OPERATOR_SYMBOLS,
    BaselineComparison,
    BaselineRule,
    EvaluationRecord,
    GateOutcome,
    MetricComparison,
    MetricOutcome,
    Outcome,
    RecordStatus,
    SuiteGateResult,
    ThresholdRule,
)

_EPS = 1e-12


def _satisfied(rule: ThresholdRule, observed: float) -> bool:
    match rule.operator:
        case "ge":
            return observed >= rule.value
        case "gt":
            return observed > rule.value
        case "le":
            return observed <= rule.value
        case "lt":
            return observed < rule.value
        case _:
            return math.isclose(observed, rule.value, rel_tol=0.0, abs_tol=rule.tolerance + _EPS)


def evaluate_rules(
    rules: Iterable[ThresholdRule], values: Mapping[str, float | None], scope: str
) -> tuple[GateOutcome, ...]:
    outcomes = []
    for rule in rules:
        observed = values.get(rule.metric)
        symbol = OPERATOR_SYMBOLS[rule.operator]
        if observed is None:
            kind, reason = "metric_unavailable", f"{rule.metric} was not measured, so {rule.expression} cannot be confirmed"
        elif _satisfied(rule, observed):
            kind, reason = "passed", f"{observed:.6g} {symbol} {rule.value:g}"
        else:
            kind, reason = "threshold_failed", f"observed {observed:.6g}, required {symbol} {rule.value:g}"
        outcomes.append(
            GateOutcome(scope=scope, metric=rule.metric, operator=rule.operator, threshold=rule.value,  # type: ignore[arg-type]
                        observed=observed, kind=kind, required=rule.required, reason=reason)  # type: ignore[arg-type]
        )
    return tuple(outcomes)


def record_values(metrics: Iterable[MetricOutcome]) -> dict[str, float | None]:
    """Numeric value of each metric that produced one (errors produce none)."""
    return {m.name: m.value for m in metrics if m.outcome is not Outcome.ERROR and m.value is not None}


def aggregate(records: Iterable[EvaluationRecord]) -> dict[str, float | None]:
    items = list(records)
    result: dict[str, float | None] = {}
    total = len(items)
    if total:
        result["case_pass_rate"] = sum(r.status is RecordStatus.PASSED for r in items) / total
        result["model_error_rate"] = sum(r.status is RecordStatus.MODEL_ERROR for r in items) / total
        result["evaluator_error_rate"] = sum(r.status is RecordStatus.EVALUATOR_ERROR for r in items) / total
    declared: dict[str, list[MetricOutcome]] = {}
    for record in items:
        for metric in record.metrics:
            declared.setdefault(metric.name, []).append(metric)
    for name, metrics in sorted(declared.items()):
        if name in ("latency_seconds", "output_tokens", "response_chars"):
            values = [m.value for m in metrics if m.value is not None]
            if values:
                result[f"{name}.mean"] = statistics.fmean(values)
                if name == "latency_seconds":
                    result[f"{name}.max"] = max(values)
            continue
        ruled = [m for m in metrics if m.outcome is not Outcome.SCORED]
        if ruled:
            result[f"{name}.pass_rate"] = sum(m.outcome is Outcome.PASSED for m in ruled) / len(ruled)
            result[f"{name}.errors"] = float(sum(m.outcome is Outcome.ERROR for m in ruled))
        values = [m.value for m in metrics if m.value is not None and m.outcome is not Outcome.ERROR]
        if values:
            result[f"{name}.mean"] = statistics.fmean(values)
    return result


def suite_gates(rules: Iterable[ThresholdRule], aggregates: Mapping[str, float | None]) -> SuiteGateResult:
    outcomes = evaluate_rules(rules, aggregates, "suite")
    return SuiteGateResult(
        passed=all(o.kind == "passed" for o in outcomes if o.required), outcomes=outcomes
    )


def compare_to_baseline(
    baseline_id: str,
    baseline_metrics: Mapping[str, float | None],
    current: Mapping[str, float | None],
    rules: Iterable[BaselineRule] = (),
) -> BaselineComparison:
    """Differences are always reported; only a rule can turn a worsening into a failure.

    A stochastic model produces different text from run to run, so a changed value alone is not a
    failure.
    """
    ruled = {r.metric: r for r in rules}
    for name in ruled:
        if name not in baseline_metrics and name not in current:
            raise BaselineError(f"baseline rule for {name!r}: the metric is in neither baseline nor run")
    comparisons = []
    for name in sorted(set(baseline_metrics) | set(current)):
        before, after = baseline_metrics.get(name), current.get(name)
        diff = None if before is None or after is None else after - before
        rule = ruled.get(name)
        if rule is None:
            outcome, reason = ("unchanged" if diff == 0 else "changed"), None
        elif diff is None:
            outcome, reason = "unavailable", f"the {'baseline' if before is None else 'current'} value is missing"
        else:
            worse = -diff if rule.direction == "higher_is_better" else diff
            if worse > rule.max_drop + _EPS:
                outcome, reason = "regressed", f"worsened by {worse:.6g}, allowed {rule.max_drop:g}"
            else:
                outcome, reason = "passed", None
        comparisons.append(
            MetricComparison(metric=name, baseline=before, current=after, difference=diff,
                             outcome=outcome, reason=reason)  # type: ignore[arg-type]
        )
    failed = any(
        c.outcome in ("regressed", "unavailable") and ruled[c.metric].required
        for c in comparisons if c.metric in ruled
    )
    return BaselineComparison(baseline_id=baseline_id, comparisons=tuple(comparisons), passed=not failed)
