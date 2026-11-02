import pytest
from fakes import FakeChatModel

from estudai.agents.base import run_config
from estudai.application.telemetria import UsageLedger
from estudai.domain.curriculum import Topico
from estudai.domain.material import TipoDocumento
from estudai.domain.perfil import PerfilAluno
from estudai.infrastructure.ingestion.wikibooks import PaginaWiki, fatiar
from estudai.infrastructure.knowledge.database import KnowledgeDatabase
from estudai.infrastructure.llm.callbacks import UsageLedgerCallback
from estudai.infrastructure.llm.factory import GEMINI_OPENAI_BASE_URL, build_chat_model
from estudai.infrastructure.persistence.json_profiles import JsonProfileRepository
from estudai.infrastructure.sandbox.python_executor import SubprocessPythonExecutor
from estudai.ports.knowledge import SearchMode


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


def _modelo(handler, ledger, **kw) -> FakeChatModel:
    return FakeChatModel({"tutor": handler}, callbacks=[UsageLedgerCallback(ledger, **kw)])


def test_callback_registra_custo_por_agente():
    ledger = UsageLedger()
    _modelo('{"valor": 1}', ledger, price_in=1.0, price_out=4.0).invoke("oi", config=run_config("tutor"))
    resumo = ledger.summary()
    # o FakeChatModel reporta 100 tokens de entrada e 20 de saída
    assert resumo.total.cost_usd == pytest.approx((100 * 1.0 + 20 * 4.0) / 1e6)
    assert resumo.por_agente["tutor"].calls == 1


def test_callback_descarta_chamada_que_falhou():
    def falha(_):
        raise RuntimeError("429")

    ledger = UsageLedger()
    modelo = _modelo(falha, ledger)
    with pytest.raises(RuntimeError):
        modelo.invoke("oi", config=run_config("tutor"))
    callback = modelo.callbacks[0]
    assert ledger.summary().total.calls == 0 and not callback._starts


def test_factory_usa_endpoint_compativel_do_gemini():
    gemini = build_chat_model(
        provider="gemini", model="gemini-3-flash", api_key="k", temperature=0.7, max_retries=3, timeout_s=10
    )
    assert gemini.openai_api_base == GEMINI_OPENAI_BASE_URL
    assert (gemini.temperature, gemini.max_retries) == (0.7, 3)
    outro = build_chat_model(
        provider="openai_compat",
        model="m",
        api_key="k",
        base_url="https://x/v1",
        temperature=0,
        max_retries=0,
        timeout_s=10,
    )
    assert outro.openai_api_base == "https://x/v1"


def test_fatiamento_por_secao():
    texto = "Intro curta.\n\n== Vértice ==\n" + "\n".join(["Parágrafo sobre vértice " * 10] * 6)
    trechos = fatiar(PaginaWiki(titulo="Função quadrática", topico=Topico.FUNCOES, texto=texto), max_chars=400)
    assert len(trechos) > 1
    assert all(t.secao == "Vértice" for t in trechos)  # a introdução curta é descartada
    assert len({t.id for t in trechos}) == len(trechos)


def test_limitador_de_taxa_e_compartilhado_por_modelo():
    from estudai.bootstrap import Container
    from estudai.domain.mensagens import Papel
    from estudai.settings import ModelConfig, Settings

    limitado = ModelConfig(model="gemini-x", requests_per_minute=5)
    s = Settings(_env_file=None, gemini_api_key="k", economic=limitado, frontier=limitado)
    c = Container(s)
    tutor, juiz = c.fronteira(Papel.TUTOR), c.chat(limitado, Papel.JUIZ, ledger=UsageLedger())
    # a cota do fornecedor é por modelo: papéis, temperaturas e ledgers diferentes dividem o mesmo limitador
    assert tutor is not juiz and tutor.rate_limiter is juiz.rate_limiter is not None
    assert tutor.rate_limiter.requests_per_second == pytest.approx(5 / 60)
    assert c.chat(ModelConfig(model="sem-limite"), Papel.TUTOR).rate_limiter is None


def test_bateria_nunca_esvazia_o_banco():
    from conftest import questao

    from estudai.domain.questoes import Particao
    from estudai.infrastructure.ingestion.questoes import separar_bateria

    conceitos = list(Topico.FUNCOES.conceitos)
    poucas = [questao(i, [conceitos[i % len(conceitos)]]) for i in range(5)]
    particoes = [q.particao for q in separar_bateria(poucas, por_topico=8)]
    assert particoes.count(Particao.BATERIA) == 2 and particoes.count(Particao.BANCO) == 3

    muitas = [questao(i, [conceitos[i % len(conceitos)]]) for i in range(30)]
    assert [q.particao for q in separar_bateria(muitas, por_topico=8)].count(Particao.BATERIA) == 8
