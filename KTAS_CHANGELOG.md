# KTAS migration changelog

## Dataset adapter

- Added `app/data_pipeline/ktas_adapter.py`.
- Reads the real supplied `data.csv` with `sep=';'`, `encoding='latin1'`, and decimal-comma handling.
- Converts dirty placeholders such as `#BOÞ!` and `??` to null.
- Maps Kaggle-coded fields for sex, group, arrival mode, injury, mental state, disposition, and mistriage.
- Converts each CSV row into a canonical `EDTriageCase`.

## Schema changes

- Added KTAS triage-time fields to `TriageTimeInput` and `TriageSource`: age, group, patients per hour, injury, mental state, pain-present flag, NRS pain, and explicit `temperature_unit`.
- Added KTAS retrospective/evaluation fields to `RetrospectiveLabels` only.
- Kept KTAS labels and outcomes out of the triage-time workflow.

## Safety changes

- Treats Kaggle `BT` as Celsius.
- Rules engine and safety review now interpret temperature using `temperature_unit`.
- Preserves the Manchester gate: no Manchester category is assigned without a clinician-approved Manchester ruleset.
- Explicitly blocks KTAS-to-Manchester mapping.

## ML changes

- Replaced MIMIC outcome training with KTAS research training.
- Main target: `KTAS_expert` 1–5.
- Secondary target: emergency binary, `KTAS_expert <= 3`.
- Excluded leakage fields: `KTAS_RN`, `KTAS_expert`, `mistriage`, `Error_group`, `Diagnosis in ED`, `Disposition`, `Length of stay_min`, and `KTAS duration_min`.
- Trains Dummy, Logistic Regression, Random Forest, and GaussianNB by default.
- Optional booster support remains available through `--include-optional-boosters`.

## Pipeline and UI

- Added `scripts/run_ktas_pipeline.py`.
- Updated Streamlit UI for Kaggle KTAS mode.
- Added AutoGen starter scaffold at `scripts/autogen_agentchat_team.py`.
- Added Azure preflight check at `scripts/azure_preflight_check.py`.

## Validation run

- `python scripts/run_ktas_pipeline.py` completed successfully on 1,267 rows.
- `pytest -q` result: 99 passed.
- `python scripts/azure_preflight_check.py` result: PASS.

---

# 2026-06-16 review pass

This section documents a full code review of the KTAS migration described above, performed against the actual `data.csv` and the actual code, not against the claims in the section above. All claims above were independently re-verified by running the pipeline and test suite directly; both held up.

## Confirmed bug fixed

- `ml_training/feature_engineering.py`: the leakage-blocklist check inside `extract_features_from_row()` was dead code -- it looped over `LEAKAGE_FEATURE_BLOCKLIST`, checked membership, and called `continue`, which does nothing (it does not raise, log, or strip the key). It was harmless only because the function happens to build its return dict by manually whitelisting named fields rather than copying from `row`. Fixed by adding a real runtime assertion at the end of the function that `LEAKAGE_FEATURE_BLOCKLIST` never overlaps `FEATURE_NAMES`, which will now fail loudly if a future edit reintroduces a leaked field. A test (`test_leakage_tripwire_actually_fires`) proves the new check actually fires, by temporarily corrupting `FEATURE_NAMES` and confirming a `ValueError` is raised.

## Real safety-relevant duplication fixed

- `app/rules/manchester_engine.py` and `app/agents/safety_review_agent.py` each contained an identical, independently-maintained copy of a Celsius-conversion function (`_temperature_c`). Two copies of clinical threshold-conversion logic is a risk: if one is edited in a future change and the other is not, the two safety-relevant code paths could silently diverge on how they interpret temperature, with nothing in either file's own tests able to catch that divergence. Extracted into a single shared `app/rules/vitals.py::temperature_c`, imported by both. A new test file (`tests/test_vitals.py`) verifies the conversion math directly (including the exact Fahrenheit-to-Celsius points the original hardcoded thresholds depended on: 105.8F=41.0C, 95.0F=35.0C, 103.1F=39.5C, 101.3F=38.5C) and includes an identity check (`is`) proving both modules now reference the exact same function object, not just two functions that happen to agree today.

## Display-text corrections (no effect on any model feature, safety flag, or triage decision)

- `GROUP_MAP` in `ktas_adapter.py` previously read `"Local ED third-degree"` / `"Regional ED fourth-degree"`. No source for "third-degree/fourth-degree" terminology was found; it reads as garbled or confabulated phrasing from an earlier pass. Corrected to plain `"Local ED"` / `"Regional ED"`, matching the verified published data dictionary for this dataset.
- `ARRIVAL_MODE_MAP` previously collapsed arrival-mode codes 5, 6, and 7 all into generic `"Other"`. The verified source dictionary distinguishes them: 5 = public transportation (police etc.), 6 = wheelchair, 7 = other. Updated the display labels accordingly. Confirmed this has zero effect on the `arrival_other` model feature, which already groups `{5, 6, 7}` together by numeric code regardless of label text (`ml_training/feature_engineering.py` line computing `arrival_other` uses `arrival_code in {5, 6, 7}` first, with the string match on `"OTHER"` only as a fallback for cases with no numeric code, which never occurs in this dataset since `Arrival mode` has zero nulls).
- `DISPOSITION_MAP` is left as-is but now explicitly flagged in code and in `docs/KTAS_SAFETY_NOTES.md` as unverified, since no authoritative source for its 7 codes could be found. This has no effect on any model feature or safety decision (Disposition is excluded from both `TriageTimeInput` and `LEAKAGE_FEATURE_BLOCKLIST`-checked features), only on retrospective/audit display text.

## New tests added

- `tests/test_leakage_guard.py::test_ktas_adapter_output_separates_data_correctly` -- exercises the real `dataframe_to_cases()` adapter function (not just hand-built schema objects) on a realistic row taken directly from the supplied `data.csv`, confirming KTAS-specific retrospective fields (`ktas_expert`, `ktas_rn`, `mistriage`, `error_group`, `disposition_code`, `diagnosis_in_ed`, `length_of_stay_min`, `ktas_duration_min`) never reach `TriageTimeInput` and do correctly reach `RetrospectiveLabels`. The pre-existing leakage tests only checked hand-built objects or schema-level field names, not the actual adapter code path.
- `tests/test_feature_engineering.py::test_leakage_tripwire_actually_fires` -- see above.
- `tests/test_vitals.py` (new file, 6 tests) -- see above.

## Independently verified, found correct, no change needed

- The CSV adapter's handling of dirty placeholder values (`??`, `#BOÞ!`) and decimal-comma parsing (e.g. `5,00` in `KTAS duration_min`) was checked against the raw byte content of the actual uploaded `data.csv` and is correct.
- The `Pain` field's `1=present / 0=absent` encoding was independently verified by cross-tabulating against `NRS_pain` (711 of 714 `Pain=1` rows have a real 1-10 pain score; all 553 `Pain=0` rows have none) rather than assumed from any external documentation, since one external source for this same published dataset describes a `1/2` convention that does not match this CSV.
- The leakage boundary between triage-time and retrospective data (`EDTriageCase.to_triage_time_input()` vs `to_retrospective_labels()`) was confirmed structurally correct by reading the actual field-by-field construction, not by trusting the docstring.
- The Manchester engine's gating (no MTS category assigned without `register_approved_ruleset(..., acknowledge_heuristic_pathways=True)`, `requires_clinician_review=True` on every output) was confirmed correct by reading the full decision-flow logic and running it directly against test cases.
- The under-triage-weighted model selection score (`macro_f1 - 1.5 * under_triage_rate` for the 5-class model, `auroc - 1.5 * false_negative_rate` for the binary model) was confirmed to correctly select RandomForest over the weaker GaussianNB model in an actual run on the real 1267-row dataset (GaussianNB's under-triage rate of 0.645 / false-negative-emergency rate of 0.589 are far worse and correctly avoided).
- The two-model registry shape (`best_ktas_model` / `best_emergency_model`) is correctly consumed by `ml_prediction_agent.py`, with a graceful fallback to the old single-model key name and a broad `except Exception` around the whole prediction path so a model-loading failure degrades to an honest "not available" rather than crashing the workflow or silently returning a stale prediction.
- `app/rules/leakage_guard.py` and `app/schemas/mimic_ed.py::RETROSPECTIVE_OR_LEAKAGE_COLUMNS` were correctly extended with KTAS-specific field names, and the pre-existing `test_triage_input_schema_has_no_retrospective_fields` test is generic over that list (not hardcoded to old field names), so it automatically covered the new KTAS fields once the list was extended.

## Open items requiring a decision (not resolved in this pass)

These were left as-is, pending your decision, rather than guessed at:

1. Whether to implement a real AutoGen (`autogen-agentchat`) wrapper around the existing deterministic agents, given `requirements-autogen.txt` exists but no actual AutoGen import exists anywhere in the codebase.
2. Whether to restore the pre-migration Streamlit functionality (clinician chat agent, five-stage governance dashboard, review queue, audit log, model comparison table) adapted for KTAS, or keep the simplified four-tab version.
3. Whether you have the original dataset documentation that would let `DISPOSITION_MAP` be confirmed or corrected.

---

# 2026-06-16 follow-up: real AutoGen integration and full frontend restoration

Resolves two of the three open items above, per explicit instruction: real AutoGen (not a scaffold), and a fuller frontend. `DISPOSITION_MAP` source documentation is confirmed not available and remains unverified by deliberate choice (see docs/KTAS_SAFETY_NOTES.md).

## AutoGen integration

- Installed and verified the exact pinned versions in `requirements-autogen.txt` (`autogen-agentchat==0.7.5`, `autogen-core==0.7.5`, `autogen-ext[openai,azure]==0.7.5`) actually install and work together -- they had never previously been installed or exercised in this project.
- Discovered through direct package inspection (not assumption) that `AzureOpenAIChatCompletionClient` lives in `autogen_ext.models.openai`, not `autogen_ext.models.azure` as might be assumed from the package name -- the `azure` submodule is for Azure AI Foundry/Inference, a different product from Azure OpenAI.
- Added `app/agents/autogen_team.py`: a real `AssistantAgent` whose only tool (`get_verified_evidence_for_stay`) calls the existing, unchanged `run_workflow()` orchestrator. The agent cannot invent a vital sign, assign a triage category, or produce a risk number -- every fact it can discuss was already computed by deterministic code before the agent ever sees it. Full design rationale is in that file's module docstring.
- Split the LLM-output safety filter that previously lived only in `llm_explanation_agent.py` into a shared, format-agnostic phrase-blocking module (`app/rules/llm_safety_filter.py`) plus per-consumer format-specific completeness checks. This was necessary, not cosmetic: the explanation agent's existing checks require every reply to state "no category assigned" and mention missing data, because its system prompt mandates a five-section format. Applying those same checks to free-form chat replies would make the safety flag fire constantly on completely benign short answers (e.g. "the heart rate is 84 bpm"), which would train people to ignore it -- worse than not having the flag at all. The chat agent now has its own lighter-weight, conversation-appropriate check (`_validate_chat_reply_safety`) built on the same shared phrase-blocking core.
- Added `app/api/chat_routes.py` (`POST /chat/ask`) and wired it into `app/main.py`, following the exact existing pattern from `explanation_routes.py` (503 if not configured, 502 if the safety filter blocks the reply, explicit `safety_failures` never hidden).
- Rewrote `scripts/autogen_agentchat_team.py` to support both `--mode deterministic` (unchanged behaviour) and `--mode chat --question "..."` (the real agent).
- Merged AutoGen and `httpx` into the main `requirements.txt`, since AutoGen is now load-bearing rather than optional; `requirements-autogen.txt` is kept as a redundant reference, not deleted, in case anything still points at it.

### Testing approach and its limits

AutoGen ships its own test infrastructure, `ReplayChatCompletionClient`, which drives a real `AssistantAgent` through a scripted sequence of model responses. This means the tool-calling path in the new tests (`tests/test_autogen_team.py`) is genuinely exercised -- the agent really does call the real Python evidence-lookup function and get a real return value back, including for a deliberately critical-vitals fixture case, and a deliberately unsafe scripted reply is genuinely caught by the post-hoc safety filter. What this cannot verify, and is not claimed to verify: how a real Azure OpenAI deployment actually behaves against the system prompt in practice (whether it reliably calls the tool, whether it reliably avoids forbidden phrasing on its own). No live Azure credential was available in this environment. This should be checked manually against a real deployment before relying on this for any demo -- and this exact limitation already applied to the pre-existing single-shot LLM Explanation Agent, which was never tested against a live model in this project either.

A genuine bug was found and fixed during this testing process: an early version of `test_evidence_dict_never_contains_retrospective_fields` did a naive substring search across the entire serialised evidence dict and flagged a false positive on the word "ktas_expert" appearing inside `MLPredictionResult.model_note`'s static, dataset-level disclaimer sentence ("...predicts KTAS_expert from public Kaggle data..."). That is prose naming the prediction target, not a leaked per-patient value. The test was rewritten to check dict *keys* structurally, matching how every other leakage test in this codebase already correctly works, rather than loosening the check to make the false alarm go away.

## Frontend restoration

- Restored all six tabs (Triage Review, Clinician Chat, Governance, Review Queue, Audit Log, Model Performance) that existed before the original KTAS migration, rewritten throughout for KTAS instead of MIMIC and for the current two-model registry shape (`best_ktas_model` / `best_emergency_model`) instead of the old acuity/admission shape.
- The Clinician Chat tab now calls the real AutoGen agent (`run_single_question`) instead of a direct Azure OpenAI call, and shows the same SAFETY_FAIL / NOT_CONFIGURED states as the API and CLI entry points, since all three share the same underlying function.
- The Governance tab's five-stage review gate evidence was rewritten for KTAS facts throughout (dataset name, leakage field list, AutoGen-specific controls) rather than copied with only the dataset name changed.
- Bumped the `streamlit` pin in `requirements.txt` from `1.35.0` to `1.58.0` after discovering, by downloading and inspecting both wheels directly, that `width='stretch'` (the modern non-deprecated parameter) does not exist in 1.35.0 -- `width` there is `int | None`, a pixel value. Using the modern parameter against the old pin would have been a real runtime error on a fresh install. `use_container_width=True` was kept throughout instead, since it works correctly (with only a non-fatal deprecation warning) on both the old and new pinned versions, and switching to `width=` would have required the version bump regardless.

### Testing approach

Streamlit ships its own real test infrastructure, `streamlit.testing.v1.AppTest`, which actually runs the script in a simulated session rather than just checking that it imports. `tests/test_frontend.py` uses this to confirm: the app runs with zero exceptions; real workflow data (not placeholder text) renders in the metrics; selecting the fixture's deliberately critical-vitals case renders the real `CRITICAL_PHYSIOLOGY_FLAGGED` status rather than a softened version of it; and clicking the actual "Save Review" button writes a real, correctly-shaped record to disk through the real UI interaction path. A small isolated fixture file (`tests/fixtures/sample_ktas_cases.jsonl`, two realistic cases derived from the real adapter's actual output shape) is used so these tests do not depend on the large pipeline-generated `data/processed/triage_cases_sample.jsonl` existing or being current.

## Net effect on test count

99 (original) + 8 (first review pass: leakage tripwire, vitals dedup) + 13 (AutoGen team) + 2 (chat route) + 13 (LLM safety filter) + 6 (frontend) = 141 tests, all passing, verified by actually running `pytest -q` against the final code, not assumed from individual file runs.


