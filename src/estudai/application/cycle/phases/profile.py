"""Profile loading, updating, and episodic memory."""

from estudai.agents.diagnosticador import Evidencia
from estudai.application.cycle.deps import CycleDependencies
from estudai.application.mensageria import MessageLog
from estudai.application.pratica import ResultadoFornecimento
from estudai.domain.correcao import Correcao, RelatorioPratica
from estudai.domain.curriculum import Topico
from estudai.domain.memoria import registrar_episodio
from estudai.domain.mensagens import Papel, TipoMensagem
from estudai.domain.perfil import Diagnostico, PerfilAluno, RegistroErro
from estudai.domain.planejamento import ObjetivoSessao
from estudai.domain.tutoria import ResumoEstudo, TranscricaoTutoria


def load_profile(deps: CycleDependencies, aluno_id: str) -> tuple[PerfilAluno, int]:
    perfil = deps.perfis.load(aluno_id)
    if perfil is None:
        perfil = PerfilAluno.novo(aluno_id)
        if deps.config.adaptativo:
            deps.perfis.save(perfil)
    return perfil, perfil.versao


def update_profile(
    deps: CycleDependencies,
    perfil: PerfilAluno,
    topico: Topico,
    fornecimento: ResultadoFornecimento,
    correcoes: list[Correcao],
    log: MessageLog,
) -> tuple[PerfilAluno, Diagnostico | None]:
    if not correcoes:
        return perfil, None
    itens = {e.id: e for e in fornecimento.exercicios}
    evidencias = [
        Evidencia(
            item_id=c.item_id,
            conceitos=itens[c.item_id].conceitos,
            dificuldade=itens[c.item_id].dificuldade_alvo,
            correta=c.correta,
            erros=c.erros,
        )
        for c in correcoes
    ]
    diagnostico = deps.diagnosticador.estimar(perfil, topico, evidencias)
    erros = tuple(
        RegistroErro(topico=topico, item_id=c.item_id, tipo=e.tipo, conceito=e.conceito, descricao=e.descricao)
        for c in correcoes
        for e in c.erros
    )
    perfil = perfil.evoluir(
        "pratica",
        diagnostico=diagnostico,
        novos_erros=erros,
        acertos=sum(c.correta for c in correcoes),
        respondidos=len(correcoes),
    )
    deps.perfis.save(perfil)
    log.send(
        Papel.DIAGNOSTICADOR,
        Papel.ORQUESTRADOR,
        TipoMensagem.PERFIL,
        {"versao": perfil.versao, "diagnostico": diagnostico.model_dump(mode="json")},
        confianca=diagnostico.confianca,
    )
    return perfil, diagnostico


def record_episode(
    deps: CycleDependencies,
    perfil: PerfilAluno,
    objetivo: ObjetivoSessao,
    transcricao: TranscricaoTutoria | None,
    resumo: ResumoEstudo | None,
    relatorio: RelatorioPratica | None,
    correcoes: list[Correcao],
    log: MessageLog,
) -> PerfilAluno:
    episodio = registrar_episodio(objetivo, transcricao, resumo, relatorio, correcoes)
    if episodio is None:
        return perfil
    perfil = perfil.evoluir("memoria_episodica", topico=objetivo.topico, episodio=episodio)
    deps.perfis.save(perfil)
    log.send(
        Papel.RESUMO,
        Papel.ORQUESTRADOR,
        TipoMensagem.PERFIL,
        {"versao": perfil.versao, "episodio": episodio.model_dump(mode="json")},
    )
    return perfil
