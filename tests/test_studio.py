from types import SimpleNamespace

from fakes import FakeChatModel

from estudai.agents.conteudo import AgenteConteudo
from estudai.application.cycle.config import ConfigSessao
from estudai.interfaces.studio import conteudo_graph, graph


class _KBVazia:
    def search(self, *args, **kwargs):
        return []


def test_grafo_do_ciclo_tem_as_etapas():
    g = graph(SimpleNamespace(config=ConfigSessao()))
    nomes = set(g.get_graph().nodes)
    assert {
        "load_profile",
        "diagnose",
        "derive_objective",
        "retrieve_material",
        "tutor",
        "summarize",
        "provide_exercises",
        "practice",
        "update_profile",
        "build_report",
        "record_episode",
    } <= nomes


def test_grafo_do_conteudo_mostra_agente_de_ferramentas_sem_chamar_o_modelo():
    llm = FakeChatModel({})
    agente = AgenteConteudo(llm, _KBVazia())
    g = conteudo_graph(agente)
    assert {"model", "tools"} <= set(g.get_graph().nodes)
    assert llm.requests == []
