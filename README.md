# AI Triage Agentic System — Kaggle KTAS Research Mode

This project is a **research-only AI triage workflow** using the public Kaggle Emergency Service KTAS dataset while MIMIC-IV-ED access is pending.

## Clinical safety status

- **Not for clinical use.**
- **KTAS is not Manchester Triage Scale.**
- **No KTAS-to-Manchester mapping is implemented.**
- Model outputs are research estimates only.
- Human clinical review is required for every output.

## Current dataset

Raw file expected at:

```bash
data/raw/kaggle_ktas/data.csv
```

If you have already downloaded `data.csv` somewhere else (e.g. your Downloads folder), copy or move it into that exact path before running the pipeline. On Windows PowerShell, from the project root:

```powershell
Copy-Item "C:\Users\<you>\Downloads\data.csv" "data\raw\kaggle_ktas\data.csv"
```

The supplied Kaggle CSV is semicolon-separated, Latin-1 compatible, and contains decimal commas in `KTAS duration_min`. Dirty placeholders such as `#BOÞ!` and `??` are converted to nulls.

## Main workflow

```text
Kaggle KTAS CSV
→ KTAS adapter and schema validation
→ Processed triage-time case JSONL
→ KTAS label builder
→ Model training: Dummy, Logistic Regression, Random Forest, GaussianNB
→ Deterministic safety review (Manchester engine, leakage guard, safety review agent)
→ ML KTAS research estimate
→ AutoGen clinician chat agent (explains the above; never decides)
→ Human review / governance audit
→ Streamlit dashboard (Triage Review, Clinician Chat, Governance,
  Review Queue, Audit Log, Model Performance)
```

## Run from a clean checkout

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/run_ktas_pipeline.py
pytest
uvicorn app.main:app --reload
streamlit run frontend/app.py
```

## Important outputs

```text
data/processed/triage_cases_sample.jsonl
data/processed/triage_input_only_sample.jsonl
data/processed/retrospective_labels_sample.jsonl
data/processed/ktas_labels.jsonl
data/processed/model_evaluation_report.json
data/processed/dataset_audit_report.json
data/processed/missing_triage_inputs_report.json
data/models/registry.json
```

## Target policy

Main target:

```text
label_ktas_expert = KTAS_expert, values 1–5
```

Secondary target:

```text
label_ktas_emergency = 1 if KTAS_expert in {1,2,3}, else 0
```

Blocked from triage-support model features:

```text
KTAS_RN
KTAS_expert
mistriage
Error_group
Diagnosis in ED
Disposition
Length of stay_min
KTAS duration_min
```

`KTAS_RN` is preserved for audit but excluded from the main model because it is already a nurse triage decision.

## AutoGen integration

Real `autogen-agentchat` multi-agent orchestration for the explanation/chat
layer, in `app/agents/autogen_team.py`. The design rationale is documented in
full at the top of that file; in short: AutoGen explains already-computed
evidence, it never makes a clinical decision. The deterministic Manchester
engine, leakage guard, safety review, and ML prediction are completely
unchanged by this integration.

Try it from the command line:

```bash
# Deterministic baseline (no LLM, works without any credentials)
python scripts/autogen_agentchat_team.py --mode deterministic --stay-id 1

# Real AutoGen chat agent (requires Azure OpenAI credentials in .env;
# without them, prints a clear NOT_CONFIGURED message rather than failing)
python scripts/autogen_agentchat_team.py --mode chat --question "Tell me about stay 1"
```

Or via the API once `uvicorn app.main:app --reload` is running:

```bash
curl -X POST http://127.0.0.1:8000/chat/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me about stay 1"}'
```

Or from the Streamlit UI's "💬 Clinician Chat" tab.

All three entry points are backed by the same `run_single_question()` function
and the same safety filter, so they cannot silently diverge in behaviour.

The AutoGen integration tests (`tests/test_autogen_team.py`) use AutoGen's own
`ReplayChatCompletionClient` test infrastructure, which genuinely exercises
the tool-calling machinery (the agent really does call the real evidence
lookup function) without needing a live Azure credential. What those tests
cannot verify is how a real Azure OpenAI deployment behaves in practice
against the system prompt -- that should be checked manually against a real
deployment before relying on this for any demo.

## MIMIC/UHL future phase

The project keeps MIMIC paths in `app/config.py`, but the active default is `kaggle_ktas`. MIMIC-IV-ED and UHL validation must not be claimed until the relevant access, governance, data dictionary, and validation approvals are actually in place.
