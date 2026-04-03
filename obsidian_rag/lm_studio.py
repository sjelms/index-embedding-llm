from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class LMStudioError(RuntimeError):
    pass


@dataclass(slots=True)
class ModelDescriptor:
    key: str
    type: str
    aliases: set[str]


class LMStudioClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict[str, object]:
        try:
            models = self.list_models()
            return {"ok": True, "models": len(models), "error": None}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "models": 0, "error": str(exc)}

    def list_models(self) -> list[ModelDescriptor]:
        requests = self._requests()
        response = requests.get(f"{self.base_url}/v1/models", timeout=10)
        response.raise_for_status()
        payload = response.json()
        raw_models = payload.get("data") or payload.get("models") or []
        models: list[ModelDescriptor] = []
        for raw_model in raw_models:
            aliases = {
                str(value).strip().lower()
                for value in (
                    raw_model.get("id"),
                    raw_model.get("key"),
                    raw_model.get("name"),
                    raw_model.get("display_name"),
                    raw_model.get("identifier"),
                )
                if value
            }
            key = (
                raw_model.get("id")
                or raw_model.get("key")
                or raw_model.get("identifier")
                or raw_model.get("name")
            )
            if not key:
                continue
            model_type = str(raw_model.get("type") or raw_model.get("architecture") or "").lower()
            models.append(ModelDescriptor(key=key, type=model_type, aliases=aliases))
        return models

    def resolve_embedding_model(self, hint: str | None, indexed_model: str | None = None) -> str:
        models = self.list_models()
        embedding_models = [model for model in models if self._is_embedding_model(model)]
        if not embedding_models:
            raise LMStudioError("No embedding models are available from LM Studio.")
        if indexed_model:
            for model in embedding_models:
                if indexed_model.lower() in model.aliases or indexed_model.lower() == model.key.lower():
                    if hint and self._hint_resolves_to_different_model(hint, indexed_model, embedding_models):
                        raise LMStudioError(
                            f"Configured embedding model '{hint}' does not match indexed model '{indexed_model}'. "
                            "Switch EMBEDDING_MODEL or rebuild the index."
                        )
                    return model.key
            raise LMStudioError(
                f"Indexed embedding model '{indexed_model}' is not currently available in LM Studio."
            )
        if hint:
            normalized_hint = hint.lower().strip()
            exact = [model for model in embedding_models if normalized_hint == model.key.lower() or normalized_hint in model.aliases]
            if exact:
                return exact[0].key
            partial = [
                model
                for model in embedding_models
                if any(normalized_hint in alias for alias in model.aliases)
            ]
            if partial:
                return partial[0].key
        for model in embedding_models:
            if any("embeddinggemma" in alias or "embedding-gemma" in alias for alias in model.aliases):
                return model.key
        return embedding_models[0].key

    def embed_texts(self, inputs: list[str], model: str) -> list[list[float]]:
        requests = self._requests()
        response = requests.post(
            f"{self.base_url}/v1/embeddings",
            json={"model": model, "input": inputs},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])
        embeddings = [item["embedding"] for item in data]
        if len(embeddings) != len(inputs):
            raise LMStudioError("Embedding response count did not match request count.")
        return embeddings

    @staticmethod
    def _is_embedding_model(model: ModelDescriptor) -> bool:
        if "embedding" in model.type:
            return True
        return any("embedding" in alias for alias in model.aliases)

    @staticmethod
    def _hint_resolves_to_different_model(hint: str, indexed_model: str, models: list[ModelDescriptor]) -> bool:
        normalized_hint = hint.lower().strip()
        for model in models:
            if normalized_hint == model.key.lower() or normalized_hint in model.aliases:
                return model.key.lower() != indexed_model.lower()
        return False

    @staticmethod
    def _requests() -> Any:
        import requests

        return requests
