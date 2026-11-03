"""Memória episódica: transforma o que aconteceu num ciclo em um `EpisodioSessao`.

Regra pura (sem LLM): junta o objetivo, o resumo do Agente de Resumo, o relatório da prática e
as correções. O episódio é guardado no perfil e chega aos agentes do ciclo seguinte via `visao()`.
"""

from estudai.domain.correcao import Correcao, RelatorioPratica
from estudai.domain.curriculum import conceitos_validos
from estudai.domain.perfil import EpisodioSessao
from estudai.domain.planejamento import ObjetivoSessao
from estudai.domain.tutoria import Encerramento, ResumoEstudo, TranscricaoTutoria

MAX_ITENS = 5  # por lista: a memória vai em todo prompt, precisa ser curta


def _unicos(*listas: list[str]) -> tuple[str, ...]:
    vistos = dict.fromkeys(item.strip() for lista in listas for item in lista if item and item.strip())
    return tuple(vistos)[:MAX_ITENS]


def registrar_episodio(
    objetivo: ObjetivoSessao,
    transcricao: TranscricaoTutoria | None,
    resumo: ResumoEstudo | None,
    relatorio: RelatorioPratica | None,
    correcoes: list[Correcao],
) -> EpisodioSessao | None:
    """Episódio do ciclo, ou None se nada aconteceu (sem tutoria e sem prática)."""
    houve_tutoria = transcricao is not None and bool(transcricao.trocas)
    if not houve_tutoria and not correcoes:
        return None
    conceitos_com_erro = conceitos_validos(
        objetivo.topico, [e.conceito for c in correcoes for e in c.erros if e.conceito]
    )
    return EpisodioSessao(
        topico=objetivo.topico,
        conceitos_trabalhados=tuple(objetivo.conceitos_alvo),
        objetivo_atingido=houve_tutoria and transcricao.encerramento == Encerramento.OBJETIVO_ATINGIDO,
        pontos_chave=_unicos(resumo.pontos_chave if resumo else []),
        erros_comuns=_unicos(resumo.erros_comuns if resumo else [], relatorio.pontos_a_melhorar if relatorio else []),
        conceitos_com_erro=tuple(conceitos_com_erro),
        proximos_passos=_unicos(resumo.proximos_passos if resumo else [], relatorio.recomendacoes if relatorio else []),
        acertos=sum(c.correta for c in correcoes),
        respondidos=len(correcoes),
    )
