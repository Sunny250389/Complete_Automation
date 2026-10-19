"""Versioned judge rubrics.

Each rubric states what is scored, what the scale endpoints mean, and which test-case data the
criterion needs. The judge is told to score only against this text; a change of wording is a new
``version`` so reports stay comparable. Scores are estimates by a language model, not ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

from complete_automation.llm_evaluation.schemas import CriterionSpec


@dataclass(frozen=True)
class Rubric:
    name: str
    version: str
    text: str
    needs_reference: bool = False
    needs_context: bool = False
    sees_context: bool = False
    """Whether the judge is shown the context (only criteria that need it)."""
    sees_reference: bool = False
    sees_response: bool = True


_SCALE_NOTE = "Use the whole scale; the lowest value means the criterion is not met at all."

BUILTIN_RUBRICS: dict[str, Rubric] = {
    r.name: r
    for r in (
        Rubric(
            "correctness", "1",
            "Does the RESPONSE state the same facts and conclusion as the REFERENCE answer? "
            "Judge agreement with the reference only, not the outside world. Extra correct detail is "
            "fine; a contradiction or a missing key fact lowers the score. " + _SCALE_NOTE,
            needs_reference=True, sees_reference=True,
        ),
        Rubric(
            "relevance", "1",
            "Does the RESPONSE address the QUESTION that was asked? Penalise off-topic content and "
            "answers to a different question. Do not judge factual accuracy. " + _SCALE_NOTE,
        ),
        Rubric(
            "completeness", "1",
            "Does the RESPONSE cover every part of the QUESTION or task? Penalise omitted parts. "
            "Do not judge accuracy or style. " + _SCALE_NOTE,
        ),
        Rubric(
            "clarity", "1",
            "Is the RESPONSE easy to understand: well organised, unambiguous, free of needless "
            "padding? Do not judge accuracy. " + _SCALE_NOTE,
        ),
        Rubric(
            "instruction_following", "1",
            "Does the RESPONSE obey every explicit instruction and constraint in the QUESTION "
            "(format, length, tone, scope)? Penalise each violated instruction. " + _SCALE_NOTE,
        ),
        Rubric(
            "groundedness", "1",
            "Is every claim in the RESPONSE supported by the CONTEXT? A claim that the CONTEXT does "
            "not state or imply is unsupported, even if true in the real world. A claim that "
            "repeats an error in the CONTEXT is still supported. Do not judge real-world "
            "correctness. List unsupported claims as evidence. " + _SCALE_NOTE,
            needs_context=True, sees_context=True,
        ),
        Rubric(
            "context_relevance", "1",
            "Is the CONTEXT relevant to the QUESTION, that is, does it contain information needed "
            "to answer it? Judge the CONTEXT only, not the RESPONSE. " + _SCALE_NOTE,
            needs_context=True, sees_context=True, sees_response=False,
        ),
    )
}


def resolve_rubric(criterion: CriterionSpec) -> Rubric:
    """The rubric for ``criterion``: the built-in one, or a custom rubric supplied with it."""
    builtin = BUILTIN_RUBRICS.get(criterion.name)
    if criterion.rubric is not None:
        return Rubric(
            criterion.name, criterion.rubric_version or "custom", criterion.rubric,
            needs_reference=builtin.needs_reference if builtin else False,
            needs_context=builtin.needs_context if builtin else False,
            sees_context=builtin.sees_context if builtin else False,
            sees_reference=builtin.sees_reference if builtin else False,
            sees_response=builtin.sees_response if builtin else True,
        )
    if builtin is None:
        raise ValueError(
            f"unknown criterion {criterion.name!r}: use one of {sorted(BUILTIN_RUBRICS)} "
            "or supply a rubric"
        )
    return builtin
