import pytest

from estudai.application.telemetria import UsageLedger
from estudai.domain.curriculum import Topico
from estudai.domain.material import TipoDocumento
from estudai.domain.perfil import PerfilAluno
from estudai.infrastructure.ingestion.wikibooks import PaginaWiki, fatiar
from estudai.infrastructure.knowledge.database import KnowledgeDatabase
from estudai.infrastructure.llm.decorators import RetryingLLMClient, TrackingLLMClient
from estudai.infrastructure.persistence.json_profiles import JsonProfileRepository
from estudai.infrastructure.sandbox.python_executor import SubprocessPythonExecutor
from estudai.ports.knowledge import SearchMode
from estudai.ports.llm import ChatMessage, LLMRequest, LLMResponse, TransientLLMError, Usage


@pytest.mark.parametrize("modo", list(SearchMode))
def test_busca_respeita_filtros(kb, modo):
    hits = kb.search("parábola vértice", modo, k=5, tipo=TipoDocumento.MATERIAL, topico=Topico.FUNCOES)
    assert hits and hits[0].document.id == "wb-quad"
    assert all(h.document.tipo == TipoDocumento.MATERIAL and h.document.topico == Topico.FUNCOES for h in hits)
    assert all(0 <= h.score <= 1 for h in hits)


def test_kb_persiste_e_recarrega(kb):
    recarregada = KnowledgeDatabase(kb.db.uri)
    assert len(recarregada) == len(kb)
    assert recarregada.search("movimento uniforme", k=1)[0].document.id == "wb-mru"


@pytest.mark.parametrize("modo", [SearchMode.SEMANTICA, SearchMode.HIBRIDA])
def test_sem_embedder_cai_para_bm25(kb, modo):
    assert kb.search("parábola vértice", modo, k=5) == kb.search("parábola vértice", SearchMode.TEXTUAL, k=5)


def test_repositorio_preserva_versoes(tmp_path):
    repo = JsonProfileRepository(tmp_path)
    v0 = PerfilAluno.novo("ana")
    repo.save(v0)
    repo.save(v0.evoluir("teste"))
    assert repo.load("ana").versao == 1
    assert [p.versao for p in repo.history("ana")] == [0, 1]
    with pytest.raises(FileExistsError):
        repo.save(v0)
    with pytest.raises(ValueError):
        repo.load("../fora")


@pytest.fixture(scope="module")
def executor():
    return SubprocessPythonExecutor(timeout_s=30)


def test_sandbox_executa_sympy(executor):
    r = executor.run("import sympy as sp\nx = sp.symbols('x')\nprint(sp.solve(x**2 - 4, x))")
    assert r.ok and r.stdout.strip() == "[-2, 2]"


@pytest.mark.parametrize(
    "codigo",
    [
        "import socket\nsocket.socket()",
        "import subprocess\nsubprocess.run(['echo', 'x'])",
        "open('arquivo.txt', 'w').write('x')",
        "import os\nos.remove('qualquer')",
    ],
)
def test_sandbox_bloqueia_operacoes_perigosas(executor, codigo):
    r = executor.run(codigo)
    assert not r.ok and "bloquead" in r.stderr


def test_sandbox_respeita_timeout(executor):
    r = executor.run("while True: pass", timeout_s=1)
    assert r.timed_out and not r.ok


class _Instavel:
    model = "m"

    def __init__(self, falhas: int):
        self.falhas = falhas

    def complete(self, request):
        if self.falhas:
            self.falhas -= 1
            raise TransientLLMError("429")
        return LLMResponse(text="{}", model="m", usage=Usage(input_tokens=1_000_000, output_tokens=500_000))


def _req():
    return LLMRequest(system="s", messages=[ChatMessage(role="user", content="u")], caller="tutor")


def test_retry_com_backoff_e_telemetria():
    esperas = []
    ledger = UsageLedger()
    cliente = TrackingLLMClient(
        RetryingLLMClient(_Instavel(falhas=2), sleep=esperas.append, base_delay_s=1),
        ledger,
        price_input_per_mtok=1.0,
        price_output_per_mtok=4.0,
    )
    cliente.complete(_req())
    assert esperas == [1, 2]
    resumo = ledger.summary()
    assert resumo.total.cost_usd == pytest.approx(3.0)
    assert resumo.por_agente["tutor"].calls == 1


def test_retry_desiste():
    with pytest.raises(TransientLLMError):
        RetryingLLMClient(_Instavel(falhas=5), max_attempts=2, sleep=lambda _: None).complete(_req())


def test_fatiamento_por_secao():
    texto = "Intro curta.\n\n== Vértice ==\n" + "\n".join(["Parágrafo sobre vértice " * 10] * 6)
    trechos = fatiar(PaginaWiki(titulo="Função quadrática", topico=Topico.FUNCOES, texto=texto), max_chars=400)
    assert len(trechos) > 1
    assert all(t.secao == "Vértice" for t in trechos)  # a introdução curta é descartada
    assert len({t.id for t in trechos}) == len(trechos)
