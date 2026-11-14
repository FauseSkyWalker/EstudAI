"""Sessões concluídas em formato JSON: um arquivo por sessão (`<raiz>/<aluno>/<id>.json`)."""

import re
from pathlib import Path

from pydantic import ValidationError

from estudai.application.sessoes import SessaoRegistrada, SessaoResumo

_ID_VALIDO = re.compile(r"[A-Za-z0-9_.-]+")


class JsonSessionRepository:
    def __init__(self, raiz: Path) -> None:
        self._raiz = raiz

    def _dir(self, aluno_id: str) -> Path:
        if not _ID_VALIDO.fullmatch(aluno_id) or aluno_id in {".", ".."}:
            raise ValueError(f"aluno_id inválido: {aluno_id!r}")
        return self._raiz / aluno_id

    def _arquivo(self, aluno_id: str, sessao_id: str) -> Path:
        if not _ID_VALIDO.fullmatch(sessao_id) or sessao_id in {".", ".."}:
            raise ValueError(f"sessao_id inválido: {sessao_id!r}")
        return self._dir(aluno_id) / f"{sessao_id}.json"

    def save(self, sessao: SessaoRegistrada) -> None:
        caminho = self._arquivo(sessao.aluno_id, sessao.id)
        if caminho.exists():
            raise FileExistsError(f"a sessão {sessao.id} de {sessao.aluno_id} já existe")
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_text(sessao.model_dump_json(indent=2), encoding="utf-8")

    def load(self, aluno_id: str, sessao_id: str) -> SessaoRegistrada:
        caminho = self._arquivo(aluno_id, sessao_id)
        if not caminho.exists():
            raise FileNotFoundError(f"sessão {sessao_id} de {aluno_id} não encontrada")
        return SessaoRegistrada.model_validate_json(caminho.read_text(encoding="utf-8"))

    def list(self, aluno_id: str) -> list[SessaoResumo]:
        diretorio = self._dir(aluno_id)
        if not diretorio.exists():
            return []
        resumos: list[SessaoResumo] = []
        for caminho in diretorio.glob("*.json"):
            try:
                sessao = SessaoRegistrada.model_validate_json(caminho.read_text(encoding="utf-8"))
            except (OSError, ValidationError):
                continue
            resumos.append(sessao.resumo())
        return sorted(resumos, key=lambda s: s.finalizada_em, reverse=True)
