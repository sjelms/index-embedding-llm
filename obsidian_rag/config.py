from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VAULT_PATH = Path(
    "/Users/stephenelms/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault"
)


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def _resolve_path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


@dataclass(slots=True)
class AppConfig:
    project_root: Path
    vault_path: Path
    index_db_path: Path
    lm_studio_base_url: str
    embedding_model_hint: str
    top_k_default: int

    @classmethod
    def load(cls) -> "AppConfig":
        load_dotenv(PROJECT_ROOT / ".env")
        vault_path = _resolve_path(os.getenv("VAULT_PATH"), DEFAULT_VAULT_PATH)
        index_db_path = _resolve_path(
            os.getenv("INDEX_DB_PATH"),
            PROJECT_ROOT / "vault-index.db",
        )
        lm_studio_base_url = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234").rstrip("/")
        embedding_model_hint = os.getenv("EMBEDDING_MODEL", "google/embedding-gemma-300m").strip()
        top_k_default = int(os.getenv("TOP_K_DEFAULT", "5"))
        return cls(
            project_root=PROJECT_ROOT,
            vault_path=vault_path,
            index_db_path=index_db_path,
            lm_studio_base_url=lm_studio_base_url,
            embedding_model_hint=embedding_model_hint,
            top_k_default=top_k_default,
        )

    def as_dict(self) -> dict[str, str | int]:
        return {
            "project_root": str(self.project_root),
            "vault_path": str(self.vault_path),
            "index_db_path": str(self.index_db_path),
            "lm_studio_base_url": self.lm_studio_base_url,
            "embedding_model_hint": self.embedding_model_hint,
            "top_k_default": self.top_k_default,
        }
