"""Protocolo experimental

Para cada condição, persona, tópico: pré-teste na bateria fixa -> N ciclos do pipeline ->
transições de aprendizagem julgadas por juiz independente -> pós-teste.
"""

import contextlib
import time
from collections.abc import Callable
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from estudai.agents.base import AgentOutputError
from estudai.application.cycle.config import ConfigSessao
from estudai.application.cycle.result import ResultadoCiclo
from estudai.application.orquestrador import Orquestrador
from estudai.application.telemetria import UsageLedger
from estudai.domain.curriculum import Topico
from estudai.domain.perfil import NivelDominio
from estudai.domain.questoes import Particao
from estudai.domain.tutoria import TranscricaoTutoria
from estudai.evaluation.aluno_simulado import AlunoSimulado
from estudai.evaluation.artefatos import (
    anexar,
    carregar_resultados,
    diretorio_mensagens,
    diretorio_usage,
    regravar,
    salvar_snapshot,
)
from estudai.evaluation.estatistica import media
from estudai.evaluation.juizes import JuizItem, JuizTransicao, JuizTutoria, VeredictoTransicao
from estudai.evaluation.personas import Persona
from estudai.evaluation.registros import (
    Condicao,
    ItemRegistro,
    RegistroCiclo,
    ResultadoExecucao,
    TransicaoRegistro,
    registro_descartado,
)
from estudai.ports.knowledge import QuestionBank
from estudai.ports.repositories import ProfileRepository

FabricaOrquestrador = Callable[..., Orquestrador]
FabricaPerfis = Callable[[str], ProfileRepository]
AtualizarCusto = Callable[[], float]


class Experimento:
    def __init__(
        self,
        *,
        fabrica_orquestrador: FabricaOrquestrador,
        fabrica_perfis: FabricaPerfis,
        banco: QuestionBank,
        llm_aluno: BaseChatModel | None,
        juizes: list[BaseChatModel],
        config_base: ConfigSessao,
        ciclos: int = 1,
        seed: int = 0,
        avaliar_itens: bool = True,
        ledger_sistema: UsageLedger | None = None,
        ledger_avaliacao: UsageLedger | None = None,
    ) -> None:
        if not juizes:
            raise ValueError("ao menos um juiz é necessário")
        self._fabrica_orquestrador = fabrica_orquestrador
        self._fabrica_perfis = fabrica_perfis
        self._banco = banco
        self._llm_aluno = llm_aluno
        self._juizes_transicao = [JuizTransicao(j) for j in juizes]
        self._juizes_tutoria = [JuizTutoria(j) for j in juizes]
        self._juizes_item = [JuizItem(j) for j in juizes]
        self._config_base = config_base
        self._ciclos = ciclos
        self._seed = seed
        self._avaliar_itens = avaliar_itens
        self._ledger_sistema = ledger_sistema or UsageLedger()
        self._ledger_avaliacao = ledger_avaliacao or UsageLedger()
        # tempo gasto pelo juiz de transição, que roda dentro do ciclo mas não é latência do sistema
        self._tempo_avaliacao = 0.0
        # definidos por `executar` a partir do arquivo de resultados
        self._raiz_resultados: Path | None = None
        self._dir_mensagens: Path | None = None
        self._dir_usage: Path | None = None

    def executar(
        self,
        condicoes: list[Condicao],
        personas: list[Persona],
        topicos: list[Topico],
        saida: Path,
        progresso: Callable[[str], None] = lambda _: None,
        *,
        refazer_falhas: bool = False,
        custo_atual: AtualizarCusto | None = None,
        max_custo_usd: float | None = None,
    ) -> list[ResultadoExecucao]:
        feitos = {r.chave: r for r in carregar_resultados(saida)} if saida.exists() else {}
        self._raiz_resultados = saida.parent
        self._dir_mensagens = diretorio_mensagens(saida)
        self._dir_usage = diretorio_usage(saida)
        resultados = []
        for condicao in condicoes:
            for persona in personas:
                for topico in topicos:
                    chave = f"{condicao.value}|{persona.id}|{topico.value}|{self._seed}"
                    existente = feitos.get(chave)
                    if existente is not None and not (refazer_falhas and existente.erro is not None):
                        progresso(f"{chave}: já executado, pulando")
                        resultados.append(existente)
                        continue
                    if existente is not None:  # refazendo uma execução que falhou
                        feitos.pop(chave)
                        regravar(saida, list(feitos.values()))
                    progresso(f"{chave}: executando")
                    resultado = self.executar_um(condicao, persona, topico)
                    anexar(saida, resultado)
                    resultados.append(resultado)
                    progresso(f"{chave}: ganho={resultado.ganho:+.2f} erro={resultado.erro}")
                    if max_custo_usd is not None and custo_atual is not None and custo_atual() >= max_custo_usd:
                        progresso(f"limite de custo atingido (US$ {custo_atual():.4f}); interrompendo")
                        return resultados
        return resultados

    def executar_um(self, condicao: Condicao, persona: Persona, topico: Topico) -> ResultadoExecucao:
        aluno_id = f"{condicao.value}-{persona.id}-{topico.value}-s{self._seed}"
        config = ConfigSessao(
            adaptativo=condicao.adaptativo,
            max_rodadas_tutoria=self._config_base.max_rodadas_tutoria,
            exercicios_por_sessao=self._config_base.exercicios_por_sessao,
            conceitos_por_sessao=self._config_base.conceitos_por_sessao,
            seed=self._seed,
        )
        aluno = AlunoSimulado(persona, topico, self._llm_aluno, seed=self._seed)
        aprendidos: list[str] = []
        transicoes: list[TransicaoRegistro] = []

        def transicionar(transcricao: TranscricaoTutoria, _material) -> None:
            for conceito in transcricao.objetivo.conceitos_alvo:
                na_lacuna = conceito in aluno.lacunas
                aprendido, avaliacoes, erro = (
                    self._avaliar_transicao(transcricao, conceito) if na_lacuna else (False, [], None)
                )
                transicoes.append(
                    TransicaoRegistro(
                        conceito=conceito,
                        na_lacuna=na_lacuna,
                        aprendido=aprendido,
                        erro=erro,
                        avaliacoes=avaliacoes,
                    )
                )
                if aprendido:
                    aluno.aprender([conceito])
                    aprendidos.append(conceito)

        orquestrador = self._fabrica_orquestrador(
            config, condicao.fonte, self._fabrica_perfis(aluno_id), on_tutoring=transicionar
        )
        bateria = self._banco.find(topico, Particao.BATERIA)
        if not bateria:
            raise RuntimeError(f"bateria vazia para {topico.value}; rode `estudai ingest-questions`")

        resultado = ResultadoExecucao(
            condicao=condicao,
            persona_id=persona.id,
            topico=topico,
            seed=self._seed,
            pre_teste=media(aluno.responder_bateria(bateria)) or 0.0,
            pos_teste=0.0,
        )
        mark_avaliacao_inicio = self._ledger_avaliacao.mark()
        try:
            for ciclo in range(1, self._ciclos + 1):
                aprendidos.clear()
                transicoes.clear()
                mark_sistema = self._ledger_sistema.mark()
                mark_avaliacao = self._ledger_avaliacao.mark()
                nivel_inicial = persona.nivel_verdadeiro(topico, aluno.dominados)
                tempo_aluno, tempo_avaliacao = aluno.tempo_llm_s, self._tempo_avaliacao
                saida = orquestrador.executar_ciclo(aluno_id, topico, aluno)
                nivel_final = persona.nivel_verdadeiro(topico, aluno.dominados)
                latencia = saida.latencia_s - (aluno.tempo_llm_s - tempo_aluno)
                latencia -= self._tempo_avaliacao - tempo_avaliacao  # juiz de transição não é latência do sistema
                registro = self._registrar(
                    ciclo, nivel_inicial, nivel_final, saida, latencia, list(aprendidos), list(transicoes)
                )
                nome = f"{aluno_id}-ciclo{ciclo}"
                registro.mensagens = len(saida.mensagens)
                registro.arquivo_mensagens = salvar_snapshot(
                    self._dir_mensagens, self._raiz_resultados, nome, saida.mensagens
                )
                registro.arquivo_usage = salvar_snapshot(
                    self._dir_usage,
                    self._raiz_resultados,
                    nome,
                    self._ledger_sistema.records(mark_sistema) + self._ledger_avaliacao.records(mark_avaliacao),
                )
                resultado.ciclos.append(registro)
        except Exception as exc:  # noqa: BLE001 - registra a falha e segue o experimento
            resultado.erro = f"{type(exc).__name__}: {exc}"
        resultado.pos_teste = media(aluno.responder_bateria(bateria)) or 0.0
        resultado.uso_avaliacao = self._ledger_avaliacao.summary(since=mark_avaliacao_inicio)
        return resultado

    def _registrar(
        self,
        ciclo: int,
        nivel_inicial: NivelDominio,
        nivel_final: NivelDominio,
        saida: ResultadoCiclo,
        latencia_s: float,
        aprendidos: list[str],
        transicoes: list[TransicaoRegistro],
    ) -> RegistroCiclo:
        perfil_inicial = {"nivel": saida.objetivo.nivel_aluno, "conceitos_alvo": saida.objetivo.conceitos_alvo}
        notas = _tentar_todos(
            lambda j: j.avaliar(saida.transcricao, saida.material, perfil_inicial), self._juizes_tutoria
        )
        entregues = [e for e in saida.fornecimento.exercicios if e.id not in saida.removidos]
        avaliacoes = (
            [_tentar_todos(lambda j, e=e: j.avaliar(e), self._juizes_item) for e in entregues]
            if self._avaliar_itens
            else []
        )
        itens = [
            ItemRegistro(
                id=e.id,
                origem=e.origem.value,
                conceitos=e.conceitos,
                dificuldade=e.dificuldade_alvo.value,
                enunciado=e.enunciado,
                alternativas=e.alternativas,
                gabarito=e.gabarito,
                descricoes_imagens=e.descricoes_imagens,
                resolucao=e.resolucao,
                codigo_verificacao=e.codigo_verificacao,
                referencias=e.referencias,
                avaliacoes=avaliacoes[i] if i < len(avaliacoes) else [],
                votos_painel=(
                    [v.resposta for v in validacao.votos]
                    if (validacao := saida.fornecimento.validacoes.get(e.id)) is not None
                    else []
                ),
            )
            for i, e in enumerate(entregues)
        ]
        descartados = [registro_descartado(item) for item in saida.fornecimento.descartados]
        uso = saida.uso.total
        return RegistroCiclo(
            ciclo=ciclo,
            nivel_verdadeiro_inicial=nivel_inicial,
            nivel_verdadeiro_final=nivel_final,
            nivel_estimado_inicial=saida.diagnostico_inicial.nivel if saida.diagnostico_inicial else None,
            nivel_estimado_final=saida.diagnostico_final.nivel if saida.diagnostico_final else None,
            encerramento_tutoria=saida.transcricao.encerramento,
            conceitos_aprendidos=aprendidos,
            transicoes=transicoes,
            notas_tutoria=notas,
            avaliacoes_itens=avaliacoes,
            votos_painel=[[v.resposta for v in val.votos] for val in saida.fornecimento.validacoes.values()],
            itens=itens,
            descartados=descartados,
            uso_por_agente=saida.uso.por_agente,
            itens_entregues=len(entregues),
            itens_removidos=len(saida.removidos),
            itens_descartados=len(saida.fornecimento.descartados),
            taxa_descarte=saida.fornecimento.taxa_descarte,
            acertos_pratica=sum(c.correta for c in saida.correcoes),
            custo_usd=uso.cost_usd,
            tokens_entrada=uso.input_tokens,
            tokens_saida=uso.output_tokens,
            latencia_s=latencia_s,
        )

    def _avaliar_transicao(
        self, transcricao: TranscricaoTutoria, conceito: str
    ) -> tuple[bool, list[VeredictoTransicao | None], str | None]:
        if not transcricao.trocas:
            return False, [None] * len(self._juizes_transicao), "transcrição vazia"
        inicio = time.perf_counter()
        avaliacoes: list[VeredictoTransicao | None] = []
        erro: str | None = None
        try:
            for juiz in self._juizes_transicao:
                try:
                    avaliacoes.append(juiz.avaliar(transcricao, conceito))
                except AgentOutputError as exc:
                    avaliacoes.append(None)
                    erro = erro or str(exc)
        finally:
            self._tempo_avaliacao += time.perf_counter() - inicio
        primario = avaliacoes[0] if avaliacoes else None
        return (primario.aprendeu if primario is not None else False), avaliacoes, erro


def _tentar_todos(fn, juizes: list) -> list:
    resultados = []
    for juiz in juizes:
        with contextlib.suppress(AgentOutputError):  # juiz que falha não vota
            resultados.append(fn(juiz))
    return resultados
