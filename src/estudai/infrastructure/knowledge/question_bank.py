"""Banco de questões normalizadas em JSONL (gerado por `estudai ingest-questions`)."""

from collections.abc import Iterable
from pathlib import Path

from estudai.domain.curriculum import Topico
from estudai.domain.questoes import Particao, Questao


class JsonlQuestionBank:
    def __init__(self, questoes: Iterable[Questao]) -> None:
        self._questoes = list(questoes)
        self._por_id = {q.id: q for q in self._questoes}

    @classmethod
    def load(cls, caminho: Path) -> "JsonlQuestionBank":
        if not caminho.exists():
            raise FileNotFoundError(f"{caminho} não existe; rode `estudai ingest-questions`")
        with caminho.open(encoding="utf-8") as f:
            return cls(Questao.model_validate_json(linha) for linha in f if linha.strip())

    @staticmethod
    def save(questoes: Iterable[Questao], caminho: Path) -> None:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        with caminho.open("w", encoding="utf-8") as f:
            for q in questoes:
                f.write(q.model_dump_json() + "\n")

    def find(self, topico: Topico | None = None, particao: Particao | None = None) -> list[Questao]:
        return [
            q
            for q in self._questoes
            if (topico is None or q.topico == topico) and (particao is None or q.particao == particao)
        ]

    def get(self, questao_id: str) -> Questao | None:
        return self._por_id.get(questao_id)
