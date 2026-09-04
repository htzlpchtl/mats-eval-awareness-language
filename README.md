# Evaluation-awareness language transfer

Research code for testing how language-invariant linear activation signals from
evaluation/deployment-labelled prompts are in Qwen3.5-9B.

The staged protocol and scientific constraints are defined in
`RESEARCH_SPEC.md`. Work stops at every checkpoint for human review.

## Local setup

Python 3.11 is preferred.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest
```

The Stage 0 tests are synthetic and require neither research data nor model
weights.

