"""Settings for the LLM evaluation module, read from ``modules.llm_evaluation``.

Example (``config/environments/<env>.yaml``)::

    modules:
      llm_evaluation:
        base_url: http://localhost:11434   # Ollama
        model: <name from `ollama list`>   # never defaulted: the model under test is your choice
        judge_model: null                  # optional, independent of `model`
        timeout_seconds: 120
        live: false                        # true enables the real-model tests
        persist_raw: true                  # false: reports keep metrics, not prompts or responses

Environment variables use the shared prefix: ``CAUTO_MODULES__LLM_EVALUATION__MODEL=...``.
No secret is read here; this module only talks to the endpoint you configure.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from complete_automation.llm_evaluation.errors import LLMConfigurationError
from complete_automation.llm_evaluation.schemas import GenerationConfig, JudgeConfig, ModelConfig


class LLMEvaluationSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    data_root: Path = Path("data/llm_evaluation")
    baseline_dir: Path = Path("data/llm_evaluation/baselines")
    base_url: str = "http://localhost:11434"
    model: str | None = None
    judge_base_url: str | None = None
    """Defaults to ``base_url``."""
    judge_model: str | None = None
    timeout_seconds: float = Field(default=120.0, gt=0)
    live: bool = False
    """Opt in to tests that call a real model."""
    require_model: bool = False
    """When ``live`` and the model is unreachable or not installed: fail instead of skip."""
    persist_raw: bool = True
    """Store prompts, responses, references and context in reports and artifacts."""
    publish_artifacts: bool = True
    max_file_bytes: int = Field(default=20_000_000, ge=1)

    def with_base_dir(self, base_dir: Path) -> LLMEvaluationSettings:
        base = Path(base_dir)

        def resolve(path: Path) -> Path:
            return path if path.is_absolute() else (base / path).resolve()

        return self.model_copy(
            update={"data_root": resolve(self.data_root), "baseline_dir": resolve(self.baseline_dir)}
        )

    def target_model(self, generation: GenerationConfig | None = None, **overrides: object) -> ModelConfig:
        if not self.model:
            raise LLMConfigurationError(
                "no model configured: set modules.llm_evaluation.model "
                "(or CAUTO_MODULES__LLM_EVALUATION__MODEL) to a name from `ollama list`"
            )
        return ModelConfig(
            model=self.model, base_url=self.base_url, timeout_seconds=self.timeout_seconds,
            generation=generation or GenerationConfig(), **overrides,  # type: ignore[arg-type]
        )

    def judge(self, generation: GenerationConfig | None = None, **overrides: object) -> JudgeConfig:
        if not self.judge_model:
            raise LLMConfigurationError(
                "no judge model configured: set modules.llm_evaluation.judge_model "
                "(or CAUTO_MODULES__LLM_EVALUATION__JUDGE_MODEL)"
            )
        return JudgeConfig(
            model=ModelConfig(
                model=self.judge_model, base_url=self.judge_base_url or self.base_url,
                timeout_seconds=self.timeout_seconds,
                generation=generation or GenerationConfig(temperature=0.0),
            ),
            **overrides,  # type: ignore[arg-type]
        )
