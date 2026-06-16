"""Local/Azure preflight checks for the KTAS research app."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings


def _autogen_importable() -> bool:
    try:
        import autogen_agentchat  # noqa: F401
        import autogen_core  # noqa: F401
        import autogen_ext.models.openai  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> int:
    from app.agents.autogen_team import load_azure_config

    checks = {
        "raw_ktas_csv_exists": settings.raw_ktas_csv.exists(),
        "processed_cases_exists": (settings.processed_dir / "triage_cases_sample.jsonl").exists(),
        "ktas_labels_exists": (settings.processed_dir / "ktas_labels.jsonl").exists(),
        "model_registry_exists": settings.model_registry_path.exists(),
        "schema_report_exists": (settings.processed_dir / "schema_report.json").exists(),
        "model_evaluation_report_exists": (settings.processed_dir / "model_evaluation_report.json").exists(),
        "autogen_importable": _autogen_importable(),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "clinical_use": "not_for_clinical_use",
        "dataset": "Kaggle-KTAS",
        "manchester_mapping": "not_implemented",
        "autogen_chat_agent_configured": load_azure_config() is not None,
    }
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
