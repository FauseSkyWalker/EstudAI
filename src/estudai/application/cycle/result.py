"""Result of one complete study cycle."""

from pydantic import BaseModel

from estudai.application.pratica import ResultadoFornecimento
from estudai.application.telemetria import UsageSummary
from estudai.domain.correcao import Correcao, RelatorioPratica
from estudai.domain.curriculum import Topico
from estudai.domain.material import PacoteMaterial
from estudai.domain.mensagens import Mensagem
from estudai.domain.perfil import Diagnostico
from estudai.domain.planejamento import ObjetivoSessao
from estudai.domain.tutoria import ResumoEstudo, TranscricaoTutoria


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
