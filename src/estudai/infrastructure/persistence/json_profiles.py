"""Repositório de perfis em JSON: um arquivo por versão (`<raiz>/<aluno>/v0003.json`).

Versões nunca são sobrescritas, o que preserva o histórico completo do diagnóstico.
"""

import re
from pathlib import Path

from estudai.domain.perfil import PerfilAluno

_ID_VALIDO = re.compile(r"[A-Za-z0-9_.-]+")


class JsonProfileRepository:
    def __init__(self, raiz: Path) -> None:
        self._raiz = raiz

    def _dir(self, aluno_id: str) -> Path:
        if not _ID_VALIDO.fullmatch(aluno_id) or aluno_id in {".", ".."}:
            raise ValueError(f"aluno_id inválido: {aluno_id!r}")
        return self._raiz / aluno_id

    def _versoes(self, aluno_id: str) -> list[Path]:
        diretorio = self._dir(aluno_id)
        return sorted(diretorio.glob("v*.json")) if diretorio.exists() else []

    def load(self, aluno_id: str) -> PerfilAluno | None:
        versoes = self._versoes(aluno_id)
        return PerfilAluno.model_validate_json(versoes[-1].read_text(encoding="utf-8")) if versoes else None

    def save(self, perfil: PerfilAluno) -> None:
        caminho = self._dir(perfil.aluno_id) / f"v{perfil.versao:04d}.json"
        if caminho.exists():
            raise FileExistsError(f"a versão {perfil.versao} de {perfil.aluno_id} já existe")
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_text(perfil.model_dump_json(indent=2), encoding="utf-8")

    def history(self, aluno_id: str) -> list[PerfilAluno]:
        return [PerfilAluno.model_validate_json(p.read_text(encoding="utf-8")) for p in self._versoes(aluno_id)]
