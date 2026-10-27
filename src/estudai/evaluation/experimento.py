"""Protocolo experimental (Seção 5.3).

Para cada (condição, persona, tópico): pré-teste na bateria fixa -> N ciclos do pipeline ->
transições de aprendizagem julgadas por juiz independente -> pós-teste. Cada execução usa um
repositório de perfis isolado (aluno novo) e é gravada em JSONL assim que termina, o que
permite retomar um experimento interrompido sem refazer execuções.
"""

import contextlib
import json
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from estudai.agents.base import AgentOutputError
from estudai.application.orquestrador import ConfigSessao, Orquestrador, ResultadoCiclo
from estudai.domain.curriculum import Topico
from estudai.domain.perfil import NivelDominio
from estudai.domain.questoes import Particao
from estudai.domain.tutoria import TranscricaoTutoria
from estudai.evaluation.aluno_simulado import AlunoSimulado
from estudai.evaluation.juizes import AvaliacaoItem, JuizItem, JuizTransicao, JuizTutoria, NotasTutoria
from estudai.evaluation.personas import Persona
from estudai.ports.knowledge import QuestionBank
from estudai.ports.llm import LLMClient
from estudai.ports.repositories import ProfileRepository


class Condicao(StrEnum):
    C1 = "C1"  # sistema completo e adaptativo
    C2 = "C2"  # perfil congelado (isola o valor da adaptatividade)
    C3 = "C3"  # somente questões reais do banco (referência para a geração)

    @property
    def adaptativo(self) -> bool:
        return self != Condicao.C2

    @property
    def fonte(self) -> str:
        return "banco" if self == Condicao.C3 else "gerado"


class RegistroCiclo(BaseModel):
    ciclo: int
    nivel_verdadeiro: NivelDominio
    nivel_estimado: NivelDominio | None
    encerramento_tutoria: str | None
    conceitos_aprendidos: list[str]
    notas_tutoria: list[NotasTutoria]  # uma por juiz
    avaliacoes_itens: list[list[AvaliacaoItem]]  # [item][juiz]
    votos_painel: list[list[str]]  # [item aprovado][resolutor]
    itens_entregues: int
    itens_removidos: int
    itens_descartados: int
    taxa_descarte: float
    acertos_pratica: int
    custo_usd: float
    tokens_entrada: int
    tokens_saida: int
    latencia_s: float


class ResultadoExecucao(BaseModel):
    condicao: Condicao
    persona_id: str
    topico: Topico
    seed: int
    pre_teste: float
    pos_teste: float
    ciclos: list[RegistroCiclo] = Field(default_factory=list)
    erro: str | None = None

    @property
    def chave(self) -> str:
        return f"{self.condicao.value}|{self.persona_id}|{self.topico.value}|{self.seed}"

    @property
    def ganho(self) -> float:
        return self.pos_teste - self.pre_teste


FabricaOrquestrador = Callable[[ConfigSessao, str, ProfileRepository], Orquestrador]
FabricaPerfis = Callable[[str], ProfileRepository]


class Experimento:
    def __init__(
        self,
        *,
        fabrica_orquestrador: FabricaOrquestrador,
        fabrica_perfis: FabricaPerfis,
        banco: QuestionBank,
        llm_aluno: LLMClient | None,
        juizes: list[LLMClient],
        config_base: ConfigSessao,
        ciclos: int = 1,
        seed: int = 0,
        avaliar_itens: bool = True,
    ) -> None:
        if not juizes:
            raise ValueError("ao menos um juiz é necessário")
        self._fabrica_orquestrador = fabrica_orquestrador
        self._fabrica_perfis = fabrica_perfis
        self._banco = banco
        self._llm_aluno = llm_aluno
        self._juiz_transicao = JuizTransicao(juizes[0])
        self._juizes_tutoria = [JuizTutoria(j) for j in juizes]
        self._juizes_item = [JuizItem(j) for j in juizes]
        self._config_base = config_base
        self._ciclos = ciclos
        self._seed = seed
        self._avaliar_itens = avaliar_itens

    def executar(
        self,
        condicoes: list[Condicao],
        personas: list[Persona],
        topicos: list[Topico],
        saida: Path,
        progresso: Callable[[str], None] = lambda _: None,
    ) -> list[ResultadoExecucao]:
        feitos = {r.chave: r for r in carregar_resultados(saida)} if saida.exists() else {}
        resultados = []
        for condicao in condicoes:
            for persona in personas:
                for topico in topicos:
                    chave = f"{condicao.value}|{persona.id}|{topico.value}|{self._seed}"
                    if chave in feitos:
                        progresso(f"{chave}: já executado, pulando")
                        resultados.append(feitos[chave])
                        continue
                    progresso(f"{chave}: executando")
                    resultado = self.executar_um(condicao, persona, topico)
                    _anexar(saida, resultado)
                    resultados.append(resultado)
                    progresso(f"{chave}: ganho={resultado.ganho:+.2f} erro={resultado.erro}")
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
        orquestrador = self._fabrica_orquestrador(config, condicao.fonte, self._fabrica_perfis(aluno_id))
        aluno = AlunoSimulado(persona, topico, self._llm_aluno, seed=self._seed)
        bateria = self._banco.find(topico, Particao.BATERIA)
        if not bateria:
            raise RuntimeError(f"bateria vazia para {topico.value}; rode `estudai ingest-questions`")

        resultado = ResultadoExecucao(
            condicao=condicao,
            persona_id=persona.id,
            topico=topico,
            seed=self._seed,
            pre_teste=_media(aluno.responder_bateria(bateria)),
            pos_teste=0.0,
        )
        try:
            for ciclo in range(1, self._ciclos + 1):
                nivel_verdadeiro = persona.nivel_verdadeiro(topico, aluno.dominados)
                tempo_aluno = aluno.tempo_llm_s
                saida = orquestrador.executar_ciclo(aluno_id, topico, aluno)
                latencia = saida.latencia_s - (aluno.tempo_llm_s - tempo_aluno)
                resultado.ciclos.append(self._registrar(ciclo, nivel_verdadeiro, saida, aluno, latencia))
        except Exception as exc:  # noqa: BLE001 - registra a falha e segue o experimento
            resultado.erro = f"{type(exc).__name__}: {exc}"
        resultado.pos_teste = _media(aluno.responder_bateria(bateria))
        return resultado

    def _registrar(
        self,
        ciclo: int,
        nivel_verdadeiro: NivelDominio,
        saida: ResultadoCiclo,
        aluno: AlunoSimulado,
        latencia_s: float,
    ) -> RegistroCiclo:
        # Transição lacuna -> dominado: só se um juiz independente confirmar explicação + validação
        aprendidos = [
            c for c in saida.objetivo.conceitos_alvo if c in aluno.lacunas and self._aprendeu(saida.transcricao, c)
        ]
        aluno.aprender(aprendidos)

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
        diagnostico = saida.diagnostico_final or saida.diagnostico_inicial
        uso = saida.uso.total
        return RegistroCiclo(
            ciclo=ciclo,
            nivel_verdadeiro=nivel_verdadeiro,
            nivel_estimado=diagnostico.nivel if diagnostico else None,
            encerramento_tutoria=saida.transcricao.encerramento,
            conceitos_aprendidos=aprendidos,
            notas_tutoria=notas,
            avaliacoes_itens=avaliacoes,
            votos_painel=[[v.resposta for v in val.votos] for val in saida.fornecimento.validacoes.values()],
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

    def _aprendeu(self, transcricao: TranscricaoTutoria, conceito: str) -> bool:
        if not transcricao.trocas:
            return False
        try:
            return self._juiz_transicao.avaliar(transcricao, conceito).aprendeu
        except AgentOutputError:
            return False


def _tentar_todos(fn, juizes: list) -> list:
    resultados = []
    for juiz in juizes:
        with contextlib.suppress(AgentOutputError):  # juiz que falha não vota
            resultados.append(fn(juiz))
    return resultados


def _media(acertos: list[bool]) -> float:
    return sum(acertos) / len(acertos) if acertos else 0.0


def _anexar(caminho: Path, resultado: ResultadoExecucao) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("a", encoding="utf-8") as f:
        f.write(resultado.model_dump_json() + "\n")


def carregar_resultados(caminho: Path) -> list[ResultadoExecucao]:
    with caminho.open(encoding="utf-8") as f:
        return [ResultadoExecucao.model_validate(json.loads(linha)) for linha in f if linha.strip()]
