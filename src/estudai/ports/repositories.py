from typing import Protocol

from estudai.domain.perfil import PerfilAluno


class ProfileRepository(Protocol):
    def load(self, aluno_id: str) -> PerfilAluno | None:
        """Última versão do perfil, ou None se o aluno é novo."""
        ...

    def save(self, perfil: PerfilAluno) -> None:
        """Persiste uma nova versão; versões anteriores são preservadas."""
        ...

    def history(self, aluno_id: str) -> list[PerfilAluno]: ...
