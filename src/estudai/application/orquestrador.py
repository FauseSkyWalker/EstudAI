"""Orquestrador centralizado e sequencial (Seção 4).

Mantém o estado compartilhado (perfil versionado), invoca os agentes na ordem do fluxo da
Figura 1, aplica os limites de execução e registra toda troca como `Mensagem` do protocolo.

Ciclo: [diagnóstico inicial] -> tutoria -> resumo -> prática -> correção -> atualização do
perfil -> relatório -> memória episódica. Com `adaptativo=False` (condição C2) o perfil fica
congelado: não há diagnóstico, atualização nem memória, e todos os agentes enxergam sempre o mesmo estado.
"""

import contextlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property

from langchain_core.callbacks.manager import dispatch_custom_event
from pydantic import BaseModel

from estudai.agents.base import AgentOutputError
from estudai.agents.conteudo import AgenteConteudo
from estudai.agents.corretor import Corretor
from estudai.agents.diagnosticador import Diagnosticador, Evidencia
from estudai.agents.resumo import AgenteResumo
from estudai.agents.tools import REGISTRO_CONTEUDO, RegistroConteudo
from estudai.agents.tutor import Tutor
from estudai.application.mensageria import MessageLog
from estudai.application.pratica import ExerciseProvider, ResultadoFornecimento
from estudai.application.progresso import EVENTO_FIGURA, EVENTO_NOTA, ObservadorProgresso, ProgressoCallback
from estudai.application.telemetria import UsageLedger, UsageSummary
from estudai.application.validacao import ValidadorExercicios
from estudai.domain.correcao import Correcao, RelatorioPratica, RespostaAluno, acertou
from estudai.domain.curriculum import Disciplina, Topico
from estudai.domain.figura import Figura, PedidoFigura
from estudai.domain.material import PacoteMaterial
from estudai.domain.memoria import registrar_episodio
from estudai.domain.mensagens import Mensagem, Papel, TipoMensagem
from estudai.domain.notificacao import Notificacao, TipoNotificacao
from estudai.domain.perfil import Diagnostico, PerfilAluno, RegistroErro
from estudai.domain.planejamento import ObjetivoSessao
from estudai.domain.questoes import Exercicio, OrigemItem, Particao
from estudai.domain.tutoria import Encerramento, ResumoEstudo, TranscricaoTutoria, TrocaTutoria, TurnoTutor
from estudai.orchestration.graph import build_cycle_graph
from estudai.orchestration.state import CycleState
from estudai.ports.figures import FigureRenderer
from estudai.ports.knowledge import QuestionBank
from estudai.ports.learner import Learner
from estudai.ports.repositories import ProfileRepository


@dataclass(frozen=True)
class ConfigSessao:
    adaptativo: bool = True
    max_rodadas_tutoria: int = 6
    exercicios_por_sessao: int = 8
    conceitos_por_sessao: int = 2
    seed: int = 0


class ResultadoCiclo(BaseModel):
    aluno_id: str
    topico: Topico
    adaptativo: bool
    versao_perfil_inicial: int
    versao_perfil_final: int
    sondagem: list[str]
    diagnostico_inicial: Diagnostico | None
    objetivo: ObjetivoSessao
    material: PacoteMaterial
    transcricao: TranscricaoTutoria
    resumo: ResumoEstudo | None
    fornecimento: ResultadoFornecimento
    correcoes: list[Correcao]
    removidos: list[str]
    diagnostico_final: Diagnostico | None
    relatorio: RelatorioPratica | None
    mensagens: list[Mensagem]
    uso: UsageSummary
    latencia_s: float


class Orquestrador:
    def __init__(
        self,
        *,
        diagnosticador: Diagnosticador,
        tutores: Mapping[Disciplina, Tutor],
        conteudo: AgenteConteudo,
        resumo: AgenteResumo,
        corretor: Corretor,
        fornecedor: ExerciseProvider,
        validador: ValidadorExercicios,
        perfis: ProfileRepository,
        banco: QuestionBank,
        config: ConfigSessao | None = None,
        ledger: UsageLedger | None = None,
        figuras: FigureRenderer | None = None,
        max_figuras: int = 2,
        max_figuras_exercicio: int = 1,
    ) -> None:
        self._diagnosticador = diagnosticador
        self._tutores = tutores
        self._conteudo = conteudo
        self._resumo = resumo
        self._corretor = corretor
        self._fornecedor = fornecedor
        self._validador = validador
        self._perfis = perfis
        self._banco = banco
        self._config = config or ConfigSessao()
        self._ledger = ledger or UsageLedger()
        self._figuras = figuras  # None = o Tutor não desenha figuras
        self._max_figuras = max_figuras
        self._max_figuras_exercicio = max_figuras_exercicio

    @property
    def config(self) -> ConfigSessao:
        return self._config

    @cached_property
    def _graph(self):
        return build_cycle_graph(self)

    def executar_ciclo(
        self,
        aluno_id: str,
        topico: Topico,
        aluno: Learner,
        observador: ObservadorProgresso | None = None,
    ) -> ResultadoCiclo:
        start = time.perf_counter()
        mark = self._ledger.mark()
        message_log = MessageLog()
        config = {"callbacks": [ProgressoCallback(observador)]} if observador else None
        state: CycleState = self._graph.invoke(
            {"student_id": aluno_id, "topic": topico, "learner": aluno, "log": message_log}, config=config
        )

        return ResultadoCiclo(
            aluno_id=aluno_id,
            topico=topico,
            adaptativo=self._config.adaptativo,
            versao_perfil_inicial=state["initial_version"],
            versao_perfil_final=state["profile"].versao,
            sondagem=state.get("probe_ids", []),
            diagnostico_inicial=state.get("initial_diagnosis"),
            objetivo=state["objective"],
            material=state["material"],
            transcricao=state["transcript"],
            resumo=state.get("summary"),
            fornecimento=state["provision"],
            correcoes=state.get("corrections", []),
            removidos=state.get("removed_ids", []),
            diagnostico_final=state.get("final_diagnosis"),
            relatorio=state.get("report"),
            mensagens=message_log.mensagens,
            uso=self._ledger.summary(since=mark),
            latencia_s=time.perf_counter() - start,
        )

    # --- etapas -------------------------------------------------------------------------

    def _load_profile(self, aluno_id: str) -> PerfilAluno:
        perfil = self._perfis.load(aluno_id)
        if perfil is None:
            perfil = PerfilAluno.novo(aluno_id)
            if self._config.adaptativo:
                self._perfis.save(perfil)
        return perfil

    def _run_initial_diagnosis(
        self, perfil: PerfilAluno, topico: Topico, aluno: Learner, log: MessageLog
    ) -> tuple[PerfilAluno, Diagnostico | None, list[str]]:
        banco = self._banco.find(topico, Particao.BANCO)
        sondagem = self._diagnosticador.selecionar_sondagem(topico, banco, seed=self._config.seed)
        evidencias = []
        for questao in sondagem:
            log.send(Papel.DIAGNOSTICADOR, Papel.ALUNO, TipoMensagem.PEDIDO, {"item_id": questao.id})
            resposta = aluno.solve(questao.id, questao)
            log.send(Papel.ALUNO, Papel.DIAGNOSTICADOR, TipoMensagem.RESPOSTA, resposta)
            evidencias.append(
                Evidencia(
                    item_id=questao.id,
                    conceitos=questao.conceitos,
                    dificuldade=questao.dificuldade,
                    correta=acertou(resposta, questao.gabarito),
                )
            )
        if not evidencias:
            return perfil, None, []
        diagnostico = self._diagnosticador.estimar(perfil, topico, evidencias)
        perfil = perfil.evoluir(
            "diagnostico_inicial",
            diagnostico=diagnostico,
            acertos=sum(e.correta for e in evidencias),
            respondidos=len(evidencias),
        )
        self._perfis.save(perfil)
        log.send(
            Papel.DIAGNOSTICADOR,
            Papel.ORQUESTRADOR,
            TipoMensagem.DIAGNOSTICO,
            diagnostico,
            confianca=diagnostico.confianca,
            referencias=[q.id for q in sondagem],
        )
        return perfil, diagnostico, [q.id for q in sondagem]

    def _retrieve_material(self, perfil: PerfilAluno, objetivo: ObjetivoSessao, log: MessageLog) -> PacoteMaterial:
        nivel = objetivo.nivel_aluno.value if objetivo.nivel_aluno else "não diagnosticado"
        pedido = f"{objetivo.descricao} Nível do aluno: {nivel}."
        log.send(Papel.TUTOR, Papel.CONTEUDO, TipoMensagem.PEDIDO, {"pedido": pedido})
        material = self._conteudo.buscar(pedido, objetivo.topico)
        log.send(Papel.CONTEUDO, Papel.TUTOR, TipoMensagem.MATERIAL, material, referencias=material.referencias)
        return material

    def _run_tutoring(
        self,
        perfil: PerfilAluno,
        objetivo: ObjetivoSessao,
        material: PacoteMaterial,
        aluno: Learner,
        log: MessageLog,
    ) -> tuple[TranscricaoTutoria, PacoteMaterial]:
        tutor = self._tutores[objetivo.topico.disciplina]
        visao = perfil.visao(objetivo.topico)
        transcricao = TranscricaoTutoria(objetivo=objetivo, encerramento=Encerramento.LIMITE_RODADAS)
        alvo = set(objetivo.conceitos_alvo)
        for rodada in range(1, self._config.max_rodadas_tutoria + 1):
            registro = RegistroConteudo(remetente=Papel.TUTOR, limite=1)
            token = REGISTRO_CONTEUDO.set(registro)
            try:
                turno = tutor.proximo_turno(visao, objetivo, material, transcricao)
            except AgentOutputError:
                material = self._merge_conteudo(registro, material, log)
                transcricao.encerramento = Encerramento.FALHA
                break
            finally:
                REGISTRO_CONTEUDO.reset(token)
            material = self._merge_conteudo(registro, material, log)
            turno = self._draw_figures(turno)
            log.send(Papel.TUTOR, Papel.ALUNO, TipoMensagem.TUTORIA, turno, referencias=material.referencias)
            concluido = turno.objetivo_atingido or alvo <= set(
                transcricao.conceitos_validados + turno.conceitos_validados
            )
            if concluido:
                aluno.notify(Notificacao(tipo=TipoNotificacao.TUTORIA, texto=turno.explicacao, figuras=turno.imagens))
                transcricao.trocas.append(TrocaTutoria(rodada=rodada, turno=turno))
                transcricao.encerramento = Encerramento.OBJETIVO_ATINGIDO
                break
            resposta = aluno.reply_to_tutor(turno)
            log.send(Papel.ALUNO, Papel.TUTOR, TipoMensagem.RESPOSTA, {"texto": resposta})
            transcricao.trocas.append(TrocaTutoria(rodada=rodada, turno=turno, resposta_aluno=resposta))
        return transcricao, material

    def _merge_conteudo(self, registro: RegistroConteudo, material: PacoteMaterial, log: MessageLog) -> PacoteMaterial:
        """Registra as consultas feitas pelo agente e une o material recuperado ao pacote da sessão."""
        for pedido, pacote in registro.trocas:
            log.send(registro.remetente, Papel.CONTEUDO, TipoMensagem.PEDIDO, {"pedido": pedido})
            log.send(
                Papel.CONTEUDO,
                registro.remetente,
                TipoMensagem.MATERIAL,
                pacote,
                referencias=pacote.referencias,
            )
            material = material.mesclar(pacote)
        return material

    def _draw_figures(self, turno: TurnoTutor) -> TurnoTutor:
        """Desenha as figuras pedidas pelo Tutor. Cada falha descarta apenas aquela figura, para a
        transcrição não citar algo que o aluno não viu."""
        if not turno.figuras:  # sem figuras no turno
            return turno
        if self._figuras is None:  # sem renderizador
            return turno.model_copy(update={"figuras": []})

        aceitos: list[PedidoFigura] = []
        imagens: list[Figura] = []
        for indice, pedido in enumerate(turno.figuras[: max(0, self._max_figuras)], start=1):
            _avisar_progresso({"agente": "Tutor", "descricao": f"desenhando figura {indice}: {pedido.descricao}"})
            try:
                imagem = self._figuras.render(pedido)
            except Exception as exc:  # noqa: BLE001 - figura nunca deve descartar o ciclo
                _avisar_progresso({"agente": "Tutor", "descricao": f"figura {indice} descartada ({str(exc)[:120]})"})
                continue
            aceitos.append(pedido)
            imagens.append(imagem)
        return turno.model_copy(update={"figuras": aceitos, "imagens": imagens})

    def _summarize_tutoring(
        self, transcricao: TranscricaoTutoria, material: PacoteMaterial, aluno: Learner, log: MessageLog
    ) -> ResumoEstudo | None:
        if not transcricao.trocas:
            return None
        try:
            resumo = self._resumo.resumir_tutoria(transcricao, material)
        except AgentOutputError:
            return None
        log.send(Papel.RESUMO, Papel.ALUNO, TipoMensagem.RESUMO, resumo)
        aluno.notify(Notificacao(tipo=TipoNotificacao.RESUMO, texto=_format_summary(resumo)))
        return resumo

    def _provide_exercises(
        self, perfil: PerfilAluno, topico: Topico, excluir: frozenset[str], log: MessageLog
    ) -> ResultadoFornecimento:
        # o Gerador de Exercícios pode consultar o Agente de Conteúdo durante a geração
        registro = RegistroConteudo(remetente=Papel.GERADOR_EXERCICIOS)
        token = REGISTRO_CONTEUDO.set(registro)
        try:
            fornecimento = self._fornecedor.fornecer(perfil, topico, self._config.exercicios_por_sessao, excluir)
        finally:
            REGISTRO_CONTEUDO.reset(token)
        for pedido, pacote in registro.trocas:
            log.send(Papel.GERADOR_EXERCICIOS, Papel.CONTEUDO, TipoMensagem.PEDIDO, {"pedido": pedido})
            log.send(
                Papel.CONTEUDO,
                Papel.GERADOR_EXERCICIOS,
                TipoMensagem.MATERIAL,
                pacote,
                referencias=pacote.referencias,
            )
        for descartado in fornecimento.descartados:
            log.send(
                Papel.PAINEL_RESOLUTORES,
                Papel.GERADOR_EXERCICIOS,
                TipoMensagem.VALIDACAO,
                {"aprovado": False, "motivo": descartado.motivo},
            )
        total = len(fornecimento.exercicios)
        com_figura = sum(bool(exercicio.figuras) for exercicio in fornecimento.exercicios)
        _avisar_progresso(
            {
                "agente": "Gerador de Exercícios",
                "descricao": f"{com_figura}/{total} questões geradas pediram figura",
            },
            evento=EVENTO_NOTA,
        )
        exercicios = [self._draw_exercise_figures(exercicio) for exercicio in fornecimento.exercicios]
        return fornecimento.model_copy(update={"exercicios": exercicios})

    def _draw_exercise_figures(self, exercicio: Exercicio) -> Exercicio:
        """Renderiza as figuras da questão aprovada. Falhas omitem só a imagem; as legendas ficam
        no texto para quem não vê figuras (painel, corretor e aluno simulado)."""
        if not exercicio.figuras:
            return exercicio
        pedidos = exercicio.figuras[: max(0, self._max_figuras_exercicio)]
        descricoes = [pedido.descricao for pedido in pedidos]
        if self._figuras is None:
            return exercicio.model_copy(update={"figuras": [], "imagens": [], "descricoes_imagens": []})

        imagens: list[Figura] = []
        for indice, pedido in enumerate(pedidos, start=1):
            _avisar_progresso(
                {
                    "agente": "Gerador de Exercícios",
                    "descricao": f"desenhando figura {indice} da questão {exercicio.id}: {pedido.descricao}",
                }
            )
            try:
                imagens.append(self._figuras.render(pedido))
            except Exception as exc:  # noqa: BLE001 - figura não pode descartar a questão
                _avisar_progresso(
                    {
                        "agente": "Gerador de Exercícios",
                        "descricao": (f"figura {indice} da questão {exercicio.id} descartada ({str(exc)[:120]})"),
                    }
                )
        return exercicio.model_copy(update={"figuras": pedidos, "imagens": imagens, "descricoes_imagens": descricoes})

    def _run_practice(
        self, exercicios: list[Exercicio], aluno: Learner, log: MessageLog
    ) -> tuple[list[Correcao], list[str]]:
        correcoes, removidos = [], []
        for item in exercicios:
            log.send(
                Papel.GERADOR_EXERCICIOS,
                Papel.ALUNO,
                TipoMensagem.EXERCICIO,
                item.model_dump(mode="json", include={"id", "topico", "enunciado", "alternativas"}),
                referencias=item.referencias,
            )
            resposta = aluno.solve(item.id, item)
            log.send(Papel.ALUNO, Papel.CORRETOR, TipoMensagem.RESPOSTA, resposta)
            correcao = self._grade_answer(item, resposta)

            if correcao.gabarito_suspeito and item.origem == OrigemItem.GERADO:
                revalidacao = self._validador.validar(item)
                log.send(Papel.PAINEL_RESOLUTORES, Papel.ORQUESTRADOR, TipoMensagem.VALIDACAO, revalidacao)
                if not revalidacao.aprovado:
                    removidos.append(item.id)
                    aviso = f"O exercício {item.id} foi removido: o gabarito não foi confirmado pelo painel."
                    log.send(Papel.ORQUESTRADOR, Papel.ALUNO, TipoMensagem.NOTIFICACAO, {"texto": aviso})
                    aluno.notify(Notificacao(tipo=TipoNotificacao.ITEM_REMOVIDO, texto=aviso, item_id=item.id))
                    continue

            log.send(Papel.CORRETOR, Papel.DIAGNOSTICADOR, TipoMensagem.CORRECAO, correcao)
            aluno.notify(
                Notificacao(
                    tipo=TipoNotificacao.FEEDBACK_EXERCICIO,
                    texto=correcao.feedback,
                    item_id=item.id,
                    correta=correcao.correta,
                )
            )
            correcoes.append(correcao)
        return correcoes, removidos

    def _grade_answer(self, item: Exercicio, resposta: RespostaAluno) -> Correcao:
        try:
            return self._corretor.corrigir(item, resposta)
        except AgentOutputError:
            correta = acertou(resposta, item.gabarito)
            return Correcao(
                item_id=item.id,
                escolha=resposta.escolha,
                gabarito=item.gabarito,
                correta=correta,
                nota=float(correta),
                feedback="(análise detalhada indisponível)",
            )

    def _update_profile(
        self,
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
        diagnostico = self._diagnosticador.estimar(perfil, topico, evidencias)
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
        self._perfis.save(perfil)
        log.send(
            Papel.DIAGNOSTICADOR,
            Papel.ORQUESTRADOR,
            TipoMensagem.PERFIL,
            {"versao": perfil.versao, "diagnostico": diagnostico.model_dump(mode="json")},
            confianca=diagnostico.confianca,
        )
        return perfil, diagnostico

    def _build_report(
        self,
        antes: PerfilAluno,
        depois: PerfilAluno,
        topico: Topico,
        correcoes: list[Correcao],
        aluno: Learner,
        log: MessageLog,
    ) -> RelatorioPratica | None:
        if not correcoes:
            return None
        try:
            relatorio = self._resumo.resumir_pratica(correcoes, antes.visao(topico), depois.visao(topico))
        except AgentOutputError:
            return None
        log.send(Papel.RESUMO, Papel.ALUNO, TipoMensagem.RESUMO, relatorio)
        aluno.notify(Notificacao(tipo=TipoNotificacao.RELATORIO, texto=relatorio.desempenho))
        return relatorio

    def _record_episode(
        self,
        perfil: PerfilAluno,
        objetivo: ObjetivoSessao,
        transcricao: TranscricaoTutoria | None,
        resumo: ResumoEstudo | None,
        relatorio: RelatorioPratica | None,
        correcoes: list[Correcao],
        log: MessageLog,
    ) -> PerfilAluno:
        """Memória episódica (Seção 3): guarda a síntese do ciclo no perfil para os próximos ciclos."""
        episodio = registrar_episodio(objetivo, transcricao, resumo, relatorio, correcoes)
        if episodio is None:
            return perfil
        perfil = perfil.evoluir("memoria_episodica", topico=objetivo.topico, episodio=episodio)
        self._perfis.save(perfil)
        log.send(
            Papel.RESUMO,
            Papel.ORQUESTRADOR,
            TipoMensagem.PERFIL,
            {"versao": perfil.versao, "episodio": episodio.model_dump(mode="json")},
        )
        return perfil


def _format_summary(resumo: ResumoEstudo) -> str:
    partes = [f"# {resumo.titulo}", *[f"- {p}" for p in resumo.pontos_chave]]
    if resumo.formulas:
        partes += ["Fórmulas:", *[f"  {f}" for f in resumo.formulas]]
    return "\n".join(partes)


def _avisar_progresso(dados: dict, *, evento: str = EVENTO_FIGURA) -> None:
    """Evento customizado para o `ProgressoCallback` (só há quem ouça dentro de uma execução do grafo)."""
    # fora de uma execução do LangChain (ex.: chamada direta em testes) não há quem ouça
    with contextlib.suppress(RuntimeError):
        dispatch_custom_event(evento, dados)
