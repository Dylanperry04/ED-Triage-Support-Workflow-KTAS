"""
Project-wide configuration.

The default active dataset is the public Kaggle KTAS emergency-service triage
CSV supplied for the current project phase. MIMIC-IV-ED paths are preserved so
that the dataset adapter can be swapped later after access is approved.
"""
from pathlib import Path
from pydantic import BaseModel


class Settings(BaseModel):
    model_config = {"protected_namespaces": ()}

    project_root: Path = Path(__file__).resolve().parents[1]
    data_root: Path = project_root / "data"

    # Kaggle KTAS current phase
    raw_ktas_dir: Path = data_root / "raw" / "kaggle_ktas"
    raw_ktas_csv: Path = raw_ktas_dir / "data.csv"

    # MIMIC paths preserved for the later approved-data phase
    raw_ed_dir: Path = data_root / "raw" / "mimic-iv-ed" / "2.2" / "ed"
    raw_demo_dir: Path = data_root / "raw" / "mimic-iv-ed-demo" / "2.2" / "ed"

    processed_dir: Path = data_root / "processed"
    models_dir: Path = data_root / "models"
    model_registry_path: Path = models_dir / "registry.json"

    # "kaggle_ktas" | "demo" | "full"
    active_dataset: str = "kaggle_ktas"

    @property
    def active_raw_dir(self) -> Path:
        if self.active_dataset == "kaggle_ktas":
            return self.raw_ktas_dir
        return self.raw_demo_dir if self.active_dataset == "demo" else self.raw_ed_dir


settings = Settings()
