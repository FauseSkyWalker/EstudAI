from typing import Any


class SentenceTransformersEmbedder:
    def __init__(
        self, model_name: str, query_params: dict[str, Any] | None = None, doc_params: dict[str, Any] | None = None
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.query_params = query_params or {}
        self.doc_params = doc_params or {}

    def embed(
        self,
        texts: list[str],
        query: bool = False,
    ) -> list[list[float]]:
        return self.model.encode(texts, **(self.query_params if query else self.doc_params)).tolist()
