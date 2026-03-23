# Complete_Automation
This framework can automate UI, API, and LLM models

World-class pytest automation foundation for LLM evaluation using DeepEval.

## Framework Highlights

- **LLM-first project structure** with reusable modules for metrics, app wrappers, config, utilities, and reports.
- **Excel-driven test data** (`.xlsx`) so QA and product teams can maintain cases without touching Python code.
- **Centralized setup/teardown** through `conftest.py` fixtures.
- **Pytest-ready defaults** with `pytest.ini` conventions and discoverable test locations.
- **DeepEval integration** for `AnswerRelevancyMetric` and `ContextualPrecisionMetric`.

## Folder Structure

```text
.
├── llm_automation/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── llm_app.py
│   ├── data/
│   │   └── testdata/
│   │       └── llm_eval_cases.xlsx
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── deepeval_metrics.py
│   ├── reporting/
│   │   ├── __init__.py
│   │   └── report_manager.py
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_answer_relevancy.py
│   │   └── test_contextual_precision.py
│   └── utils/
│       ├── __init__.py
│       └── excel_reader.py
├── pytest.ini
└── requirements.txt
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Environment Variables

- `CONFIDENT_API_KEY`: Optional. If present, logs into Confident AI in session setup.
- `DEEPEVAL_MODEL`: Optional model selector (default `gpt-4.1`).
- `CONFIDENT_ALIAS`: Optional alias for dataset/reporting flows.
- `LOG_LEVEL`: Optional logging level.

## Extend This Framework

1. Replace `llm_automation/core/llm_app.py` with your actual LLM app or agent call.
2. Add new rows to `llm_automation/data/testdata/llm_eval_cases.xlsx`.
3. Add new metric files under `llm_automation/metrics` and tests under `llm_automation/tests`.
4. Add CI pipeline steps to execute `pytest` and publish `reports/` artifacts.