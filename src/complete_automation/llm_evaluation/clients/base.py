"""The provider interface: what the evaluation code needs from a model, and nothing more.

To add a provider, write a class with these two methods (``generate`` and ``identify``), return a
normalised :class:`ModelResponse`, and raise :class:`ModelInvocationError` subclasses for failures.
Metric and runner code never imports a concrete client.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from complete_automation.llm_evaluation.schemas import ModelConfig, ModelRequest, ModelResponse


@runtime_checkable
class ModelClient(Protocol):
    provider: str

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Call the model once and return the normalised response.

        A complete response has ``status == COMPLETED``. A stream that ends early returns
        ``PARTIAL`` with the text received so far. A call that produced nothing raises a
        ``ModelInvocationError`` subclass. Implementations never retry on their own.
        """
        ...

    def identify(self, config: ModelConfig) -> str | None:
        """A version or digest for the configured model, or ``None`` when the provider has none."""
        ...
