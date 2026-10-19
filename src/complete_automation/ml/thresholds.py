"""Quality gates: threshold rules applied to computed metrics.

Rules are always explicit (metric, operator, value). A rule whose metric is missing, undefined
or unsupported *fails* (``metric_unavailable``) unless it is marked ``required=False``; it never
passes by default. Threshold failures and metric-calculation failures are separate outcomes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from complete_automation.ml.errors import MLConfigurationError
from complete_automation.ml.metrics import direction_of
from complete_automation.ml.schemas import (
    OPERATOR_SYMBOLS,
    Direction,
    GateOutcome,
    GateSummary,
    MetricValue,
    Operator,
    TaskType,
    ThresholdRule,
)

_EPS = 1e-12


def _satisfied(rule: ThresholdRule, observed: float) -> bool:
    match rule.operator:
        case Operator.GE:
            return observed >= rule.value
        case Operator.GT:
            return observed > rule.value
        case Operator.LE:
            return observed <= rule.value
        case Operator.LT:
            return observed < rule.value
        case Operator.EQ:
            return math.isclose(observed, rule.value, rel_tol=0.0, abs_tol=rule.tolerance + _EPS)


def evaluate_gates(
    rules: Iterable[ThresholdRule], metrics: Iterable[MetricValue], task: TaskType
) -> GateSummary:
    """Apply every rule; the summary passes only if all *required* rules passed."""
    by_name = {m.name: m for m in metrics}
    outcomes: list[GateOutcome] = []
    warnings: list[str] = []
    for rule in rules:
        symbol = OPERATOR_SYMBOLS[rule.operator]
        direction = direction_of(task, rule.metric)
        if (direction is Direction.HIGHER_IS_BETTER and rule.operator in (Operator.LE, Operator.LT)) or (
            direction is Direction.LOWER_IS_BETTER and rule.operator in (Operator.GE, Operator.GT)
        ):
            warnings.append(
                f"{rule.expression}: {rule.metric} is {direction.value.replace('_', ' ')}; "
                f"check that {symbol} is the intended direction"
            )
        found = by_name.get(rule.metric)
        if found is None:
            outcomes.append(_outcome(rule, None, "metric_unavailable", "the metric was not computed"))
        elif found.value is None:
            why = found.reason or "no scalar value"
            outcomes.append(_outcome(rule, None, "metric_unavailable", f"metric is {found.status.value}: {why}"))
        elif _satisfied(rule, found.value):
            outcomes.append(_outcome(rule, found.value, "passed", f"{found.value:.10g} {symbol} {rule.value:.10g}"))
        else:
            outcomes.append(_outcome(
                rule, found.value, "threshold_failed",
                f"observed {found.value:.10g}, required {symbol} {rule.value:.10g}"))
    passed = all(o.kind == "passed" for o in outcomes if o.required)
    return GateSummary(passed=passed, outcomes=tuple(outcomes), warnings=tuple(warnings))


def _outcome(rule: ThresholdRule, observed: float | None, kind: str, reason: str) -> GateOutcome:
    return GateOutcome(
        metric=rule.metric, operator=rule.operator, threshold=rule.value, observed=observed,
        kind=kind, required=rule.required, reason=reason,  # type: ignore[arg-type]
    )


def parse_thresholds(spec: Mapping[str, Any]) -> tuple[ThresholdRule, ...]:
    """Build rules from configuration such as ``{"f1": {"ge": 0.9}, "rmse": {"le": 5.0}}``.

    A metric may carry several operators. Unknown operators raise ``MLConfigurationError``.
    """
    rules: list[ThresholdRule] = []
    for metric_name, operators in spec.items():
        if not isinstance(operators, Mapping) or not operators:
            raise MLConfigurationError(f"thresholds for {metric_name!r} must map operators to values")
        for operator, value in operators.items():
            try:
                rules.append(ThresholdRule(metric=metric_name, operator=Operator(operator), value=value))
            except ValueError as error:
                raise MLConfigurationError(
                    f"invalid threshold {metric_name} {operator} {value!r} "
                    f"(operators: {[o.value for o in Operator]}): {error}"
                ) from None
    return tuple(rules)
