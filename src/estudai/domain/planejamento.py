"""Planejamento determinístico de sessão (Seção 3: sem planejamento global autônomo).

O objetivo de cada sessão e o plano de prática derivam apenas do perfil, o que torna o
fluxo reprodutível e barato, e isola o efeito da adaptatividade nos experimentos (C1 x C2).
"""

from pydantic import BaseModel

from estudai.domain.curriculum import Topico
from estudai.domain.perfil import NivelDominio, PerfilAluno
from estudai.domain.questoes import Dificuldade

_DIFICULDADE_POR_NIVEL = {
    None: Dificuldade.MEDIA,
    NivelDominio.INICIANTE: Dificuldade.FACIL,
    NivelDominio.INTERMEDIARIO: Dificuldade.MEDIA,
    NivelDominio.AVANCADO: Dificuldade.DIFICIL,
}


def dificuldade_para_nivel(nivel: NivelDominio | None) -> Dificuldade:
    return _DIFICULDADE_POR_NIVEL[nivel]


def _conceitos_prioritarios(perfil: PerfilAluno, topico: Topico) -> list[str]:
    """Lacunas primeiro, depois conceitos não dominados, depois o restante do tópico."""
    estado = perfil.estado(topico)
    dominados = set(estado.conceitos_dominados)
    ordem = list(estado.lacunas)
    ordem += [c for c in topico.conceitos if c not in dominados and c not in ordem]
    ordem += [c for c in topico.conceitos if c not in ordem]
    return ordem


class ObjetivoSessao(BaseModel):
    topico: Topico
    conceitos_alvo: list[str]
    nivel_aluno: NivelDominio | None
    descricao: str

    @classmethod
    def derivar(cls, perfil: PerfilAluno, topico: Topico, max_conceitos: int = 2) -> "ObjetivoSessao":
        alvo = _conceitos_prioritarios(perfil, topico)[:max_conceitos]
        return cls(
            topico=topico,
            conceitos_alvo=alvo,
            nivel_aluno=perfil.estado(topico).nivel,
            descricao=f"Ensinar {', '.join(alvo)} ({topico.value}) e verificar o entendimento.",
        )


class ItemPlanejado(BaseModel):
    conceito: str
    dificuldade: Dificuldade


def planejar_pratica(perfil: PerfilAluno, topico: Topico, n: int, foco: int = 3) -> list[ItemPlanejado]:
    """Distribui `n` itens em rodízio entre os `foco` conceitos mais prioritários."""
    conceitos = _conceitos_prioritarios(perfil, topico)[:foco]
    dificuldade = dificuldade_para_nivel(perfil.estado(topico).nivel)
    return [ItemPlanejado(conceito=conceitos[i % len(conceitos)], dificuldade=dificuldade) for i in range(n)]
