"""
Tests for the Streamlit frontend (frontend/app.py), using Streamlit's own
AppTest framework (streamlit.testing.v1), which actually runs the script in
a simulated session rather than just checking that it imports or parses.

These tests use the same fixture cases as the AutoGen tests so they do not
depend on the large pipeline-generated data/processed/triage_cases_sample.jsonl
existing or being current.
"""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


FRONTEND_PATH = Path(__file__).parent.parent / "frontend" / "app.py"
FIXTURES = Path(__file__).parent / "fixtures" / "sample_ktas_cases.jsonl"


@pytest.fixture
def isolated_processed_dir(tmp_path, monkeypatch):
    """
    Points settings.processed_dir at a temporary directory pre-populated
    with the small test fixture cases, so frontend tests do not depend on
    (or pollute) the real data/processed directory used by the actual
    pipeline.
    """
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "triage_cases_sample.jsonl").write_text(
        FIXTURES.read_text(encoding="utf-8"), encoding="utf-8"
    )
    # missing_triage_inputs_report.json is read by the Review Queue and
    # Audit Log tabs; provide a minimal valid one so those tabs render
    # their real content rather than just the "not found" branch.
    (processed / "missing_triage_inputs_report.json").write_text(
        json.dumps(
            {
                "cases_with_missing_triage_inputs": 0,
                "missing_case_percent": 0.0,
                "missing_cases": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("app.config.settings.processed_dir", processed)
    return processed


class TestFrontendRendersWithoutErrors:
    def test_app_runs_with_no_exceptions(self, isolated_processed_dir):
        at = AppTest.from_file(str(FRONTEND_PATH))
        at.run(timeout=60)
        assert list(at.exception) == []

    def test_six_tabs_present_and_titles_correct(self, isolated_processed_dir):
        at = AppTest.from_file(str(FRONTEND_PATH))
        at.run(timeout=60)
        assert "Kaggle KTAS Research Mode" in at.title[0].value

    def test_real_case_data_renders_in_metrics(self, isolated_processed_dir):
        """
        Confirms the rendered metrics reflect the REAL deterministic
        workflow output for the fixture's first case, not placeholder text.
        """
        at = AppTest.from_file(str(FRONTEND_PATH))
        at.run(timeout=60)
        metric_labels_values = {m.label: m.value for m in at.metric}
        assert metric_labels_values.get("Chief Complaint") == "right ocular pain"
        assert metric_labels_values.get("Stay ID") == "1"

    def test_critical_case_safety_assessment_is_not_silently_softened(self, isolated_processed_dir):
        """
        Selects the critical-vitals fixture case (stay 2) and confirms the
        rendered safety assessment reflects the real CRITICAL_PHYSIOLOGY_FLAGGED
        status -- i.e. the frontend is not quietly downgrading or hiding a
        dangerous result.
        """
        at = AppTest.from_file(str(FRONTEND_PATH))
        at.run(timeout=60)

        ed_stay_selectbox = next(sb for sb in at.selectbox if sb.label == "ED Stay")
        critical_option = next(
            opt for opt in ed_stay_selectbox.options if opt.startswith("Stay 2")
        )
        ed_stay_selectbox.set_value(critical_option)
        at.run(timeout=60)

        assert list(at.exception) == []
        error_texts = " ".join(e.value for e in at.error)
        assert "CRITICAL PHYSIOLOGY FLAGGED" in error_texts.upper().replace("_", " ")


class TestReviewSubmissionWritesRealRecord:
    def test_clicking_save_review_writes_a_real_record_to_disk(self, isolated_processed_dir):
        review_path = isolated_processed_dir / "human_reviews.jsonl"
        assert not review_path.exists()

        at = AppTest.from_file(str(FRONTEND_PATH))
        at.run(timeout=60)

        submit_button = next(
            b for b in at.button if "Save Review to Audit Log" in b.label
        )
        submit_button.click()
        at.run(timeout=60)

        assert list(at.exception) == []
        assert review_path.exists()
        lines = review_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["stay_id"] == 1
        assert record["reviewer_role"] == "triage_nurse"


class TestChatTabDegradesGracefullyWithoutAzure:
    def test_chat_tab_shows_not_configured_warning(self, isolated_processed_dir, monkeypatch):
        for key in (
            "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_VERSION",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(
            "app.agents.autogen_team.ENV_PATH", Path("/nonexistent/.env")
        )

        at = AppTest.from_file(str(FRONTEND_PATH))
        at.run(timeout=60)

        assert list(at.exception) == []
        warning_texts = " ".join(w.value for w in at.warning)
        assert "Azure OpenAI is not configured" in warning_texts
