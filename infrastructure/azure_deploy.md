# Azure deployment notes — KTAS research mode

This project can be deployed as a research demo to Azure App Service, but it is **not for clinical use**.

Before deployment run:

```bash
python scripts/run_ktas_pipeline.py
python scripts/azure_preflight_check.py
pytest
```

The deployed `/health` endpoint must continue to report:

```text
clinical_use = not_for_clinical_use
manchester_mapping = not_implemented
human_review_required = true
```

Do not deploy this as a clinical triage system. KTAS is not Manchester and UHL validation has not been performed.
