"""Orquestrador centralizado e sequencial (Seção 4).

Mantém o estado compartilhado (perfil versionado), invoca os agentes na ordem do fluxo da
Figura 1, aplica os limites de execução e registra toda troca como `Mensagem` do protocolo.

Ciclo: [diagnóstico inicial] -> tutoria -> resumo -> prática -> correção -> atualização do
perfil -> relatório. Com `adaptativo=False` (condição C2) o perfil fica congelado: não há
diagnóstico nem atualização, e todos os agentes enxergam sempre o mesmo estado.
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import BaseModel

from estudai.agents.base import AgentOutputError
from estudai.agents.conteudo import AgenteConteudo
from estudai.agents.corretor import Corretor
from estudai.agents.diagnosticador import Diagnosticador, Evidencia
from estudai.agents.resumo import AgenteResumo
from estudai.agents.tutor import Tutor
from estudai.application.mensageria import MessageLog
from estudai.application.pratica import ExerciseProvider, ResultadoFornecimento
from estudai.application.telemetria import UsageLedger, UsageSummary
from estudai.application.validacao import ValidadorExercicios
from estudai.domain.correcao import Correcao, RelatorioPratica, RespostaAluno
from estudai.domain.curriculum import Disciplina, Topico
from estudai.domain.material import PacoteMaterial
from estudai.domain.mensagens import Mensagem, Papel, TipoMensagem
from estudai.domain.perfil import Diagnostico, PerfilAluno, RegistroErro
from estudai.domain.planejamento import ObjetivoSessao
from estudai.domain.questoes import Exercicio, OrigemItem, Particao
from estudai.domain.tutoria import Encerramento, ResumoEstudo, TranscricaoTutoria, TrocaTutoria
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

    @property
    def config(self) -> ConfigSessao:
        return self._config

    def executar_ciclo(self, aluno_id: str, topico: Topico, aluno: Learner) -> ResultadoCiclo:
        inicio, marca, log = time.perf_counter(), self._ledger.mark(), MessageLog()
        perfil = self._carregar_perfil(aluno_id)
        versao_inicial = perfil.versao

        diagnostico_inicial, sondagem = None, []
        if self._config.adaptativo and not perfil.diagnosticado(topico):
            perfil, diagnostico_inicial, sondagem = self._diagnostico_inicial(perfil, topico, aluno, log)

        # Macrofase 1: tutoria
        objetivo = ObjetivoSessao.derivar(perfil, topico, self._config.conceitos_por_sessao)
        material = self._recuperar_material(perfil, objetivo, log)
        transcricao = self._tutoria(perfil, objetivo, material, aluno, log)
        resumo = self._resumir_tutoria(transcricao, material, aluno, log)

        # Macrofase 2: prática
        fornecimento = self._fornecer(perfil, topico, frozenset(sondagem), log)
        correcoes, removidos = self._pratica(fornecimento.exercicios, aluno, log)

        perfil_antes, diagnostico_final = perfil, None
        if self._config.adaptativo:
            perfil, diagnostico_final = self._atualizar_perfil(perfil, topico, fornecimento, correcoes, log)
        relatorio = self._relatorio(perfil_antes, perfil, topico, correcoes, aluno, log)

        return ResultadoCiclo(
            aluno_id=aluno_id,
            topico=topico,
            adaptativo=self._config.adaptativo,
            versao_perfil_inicial=versao_inicial,
            versao_perfil_final=perfil.versao,
            sondagem=sondagem,
            diagnostico_inicial=diagnostico_inicial,
            objetivo=objetivo,
            material=material,
            transcricao=transcricao,
            resumo=resumo,
            fornecimento=fornecimento,
            correcoes=correcoes,
            removidos=removidos,
            diagnostico_final=diagnostico_final,
            relatorio=relatorio,
            mensagens=log.mensagens,
            uso=self._ledger.summary(since=marca),
            latencia_s=time.perf_counter() - inicio,
        )

    # --- etapas -------------------------------------------------------------------------

    def _carregar_perfil(self, aluno_id: str) -> PerfilAluno:
        perfil = self._perfis.load(aluno_id)
        if perfil is None:
            perfil = PerfilAluno.novo(aluno_id)
            if self._config.adaptativo:
                self._perfis.save(perfil)
        return perfil

    def _diagnostico_inicial(
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
                    correta=_acertou(resposta, questao.gabarito),
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

    def _recuperar_material(self, perfil: PerfilAluno, objetivo: ObjetivoSessao, log: MessageLog) -> PacoteMaterial:
        nivel = objetivo.nivel_aluno.value if objetivo.nivel_aluno else "não diagnosticado"
        pedido = f"{objetivo.descricao} Nível do aluno: {nivel}."
        log.send(Papel.TUTOR, Papel.CONTEUDO, TipoMensagem.PEDIDO, {"pedido": pedido})
        material = self._conteudo.buscar(pedido, objetivo.topico)
        log.send(Papel.CONTEUDO, Papel.TUTOR, TipoMensagem.MATERIAL, material, referencias=material.referencias)
        return material

    def _tutoria(
        self,
        perfil: PerfilAluno,
        objetivo: ObjetivoSessao,
        material: PacoteMaterial,
        aluno: Learner,
        log: MessageLog,
    ) -> TranscricaoTutoria:
        tutor = self._tutores[objetivo.topico.disciplina]
        visao = perfil.visao(objetivo.topico)
        transcricao = TranscricaoTutoria(objetivo=objetivo, encerramento=Encerramento.LIMITE_RODADAS)
        alvo = set(objetivo.conceitos_alvo)
        for rodada in range(1, self._config.max_rodadas_tutoria + 1):
            try:
                turno = tutor.proximo_turno(visao, objetivo, material, transcricao)
            except AgentOutputError:
                transcricao.encerramento = Encerramento.FALHA
                break
            log.send(Papel.TUTOR, Papel.ALUNO, TipoMensagem.TUTORIA, turno, referencias=material.referencias)
            concluido = turno.objetivo_atingido or alvo <= set(
                transcricao.conceitos_validados + turno.conceitos_validados
            )
            if concluido:
                aluno.notify(turno.explicacao)
                transcricao.trocas.append(TrocaTutoria(rodada=rodada, turno=turno))
                transcricao.encerramento = Encerramento.OBJETIVO_ATINGIDO
                break
            resposta = aluno.reply_to_tutor(turno)
            log.send(Papel.ALUNO, Papel.TUTOR, TipoMensagem.RESPOSTA, {"texto": resposta})
            transcricao.trocas.append(TrocaTutoria(rodada=rodada, turno=turno, resposta_aluno=resposta))
        return transcricao

    def _resumir_tutoria(
        self, transcricao: TranscricaoTutoria, material: PacoteMaterial, aluno: Learner, log: MessageLog
    ) -> ResumoEstudo | None:
        if not transcricao.trocas:
            return None
        try:
            resumo = self._resumo.resumir_tutoria(transcricao, material)
        except AgentOutputError:
            return None
        log.send(Papel.RESUMO, Papel.ALUNO, TipoMensagem.RESUMO, resumo)
        aluno.notify(_formatar_resumo(resumo))
        return resumo

    def _fornecer(
        self, perfil: PerfilAluno, topico: Topico, excluir: frozenset[str], log: MessageLog
    ) -> ResultadoFornecimento:
        fornecimento = self._fornecedor.fornecer(perfil, topico, self._config.exercicios_por_sessao, excluir)
        for descartado in fornecimento.descartados:
            log.send(
                Papel.PAINEL_RESOLUTORES,
                Papel.GERADOR_EXERCICIOS,
                TipoMensagem.VALIDACAO,
                {"aprovado": False, "motivo": descartado.motivo},
            )
        return fornecimento

    def _pratica(
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
            correcao = self._corrigir(item, resposta)

            if correcao.gabarito_suspeito and item.origem == OrigemItem.GERADO:
                revalidacao = self._validador.validar(item)
                log.send(Papel.PAINEL_RESOLUTORES, Papel.ORQUESTRADOR, TipoMensagem.VALIDACAO, revalidacao)
                if not revalidacao.aprovado:
                    removidos.append(item.id)
                    aviso = f"O exercício {item.id} foi removido: o gabarito não foi confirmado pelo painel."
                    log.send(Papel.ORQUESTRADOR, Papel.ALUNO, TipoMensagem.NOTIFICACAO, {"texto": aviso})
                    aluno.notify(aviso)
                    continue

            log.send(Papel.CORRETOR, Papel.DIAGNOSTICADOR, TipoMensagem.CORRECAO, correcao)
            aluno.notify(f"[{item.id}] {'Correta' if correcao.correta else 'Incorreta'}. {correcao.feedback}")
            correcoes.append(correcao)
        return correcoes, removidos

    def _corrigir(self, item: Exercicio, resposta: RespostaAluno) -> Correcao:
        try:
            return self._corretor.corrigir(item, resposta)
        except AgentOutputError:
            correta = _acertou(resposta, item.gabarito)
            return Correcao(
                item_id=item.id,
                escolha=resposta.escolha,
                gabarito=item.gabarito,
                correta=correta,
                nota=float(correta),
                feedback="(análise detalhada indisponível)",
            )

    def _atualizar_perfil(
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

    def _relatorio(
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
        aluno.notify(relatorio.desempenho)
        return relatorio


def _acertou(resposta: RespostaAluno, gabarito: str) -> bool:
    return (resposta.escolha or "").strip().upper() == gabarito


def _formatar_resumo(resumo: ResumoEstudo) -> str:
    partes = [f"# {resumo.titulo}", *[f"- {p}" for p in resumo.pontos_chave]]
    if resumo.formulas:
        partes += ["Fórmulas:", *[f"  {f}" for f in resumo.formulas]]
    return "\n".join(partes)
