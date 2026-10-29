import pytest
from estudai.infrastructure.knowledge.hybrid import HybridKnowledgeBase

from estudai.domain.curriculum import Disciplina, Topico
from estudai.domain.material import TipoDocumento
from estudai.domain.questoes import Dificuldade, Particao, Questao
from estudai.infrastructure.knowledge.embedders import HashingEmbedder
from estudai.infrastructure.knowledge.question_bank import JsonlQuestionBank
from estudai.ports.knowledge import Document


def questao(i: int, conceitos: list[str], particao: Particao = Particao.BANCO, gabarito: str = "B") -> Questao:
    return Questao(
        id=f"Q{i}",
        fonte="teste",
        ano=2020,
        disciplina=Disciplina.MATEMATICA,
        topico=Topico.FUNCOES,
        conceitos=conceitos,
        dificuldade=Dificuldade.MEDIA,
        particao=particao,
        enunciado=f"Questão {i} sobre {', '.join(conceitos)}",
        alternativas=["1", "2", "3", "4", "5"],
        gabarito=gabarito,
    )


@pytest.fixture
def banco() -> JsonlQuestionBank:
    conceitos = list(Topico.FUNCOES.conceitos)
    questoes = [questao(i, [conceitos[i % len(conceitos)]]) for i in range(12)]
    questoes += [questao(100 + i, [c], Particao.BATERIA) for i, c in enumerate(conceitos)]
    return JsonlQuestionBank(questoes)


@pytest.fixture
def kb() -> HybridKnowledgeBase:
    docs = [
        Document(
            id="wb-afim",
            tipo=TipoDocumento.MATERIAL,
            fonte="wikibooks:Funções",
            topico=Topico.FUNCOES,
            texto="A função afim tem a forma f(x) = ax + b, cujo gráfico é uma reta de inclinação a.",
        ),
        Document(
            id="wb-quad",
            tipo=TipoDocumento.MATERIAL,
            fonte="wikibooks:Função quadrática",
            topico=Topico.FUNCOES,
            texto="A função quadrática f(x) = ax² + bx + c tem gráfico parabólico; o vértice fica em x = -b/2a.",
        ),
        Document(
            id="wb-mru",
            tipo=TipoDocumento.MATERIAL,
            fonte="wikibooks:Cinemática",
            topico=Topico.CINEMATICA,
            texto="No movimento uniforme a velocidade é constante e s = s0 + v t.",
        ),
        Document(
            id="Q1",
            tipo=TipoDocumento.QUESTAO,
            fonte="teste:2020",
            topico=Topico.FUNCOES,
            texto="Questão de vestibular sobre o vértice da parábola da função quadrática.",
        ),
    ]
    return HybridKnowledgeBase.build(docs, HashingEmbedder())
