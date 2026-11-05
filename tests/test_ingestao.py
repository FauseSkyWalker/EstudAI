"""Normalização estrutural das bases e descarte de itens insolúveis/contaminados."""

from estudai.domain.latex import latex_valido, para_inline
from estudai.infrastructure.ingestion import sources
from estudai.infrastructure.ingestion.classificador import Rotulo
from estudai.infrastructure.ingestion.normalizacao import limpar_texto, referencia_figura, texto_invalido
from estudai.infrastructure.ingestion.questoes import RotuloCache, _para_questao, ingerir_questoes
from estudai.infrastructure.ingestion.sources import QuestaoBruta
from estudai.interfaces.web.markdown import markdown_latex

SATELITE = (
    '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>r</mi><mo>=</mo><mfrac><mn>5865</mn>'
    "<mrow><mn>1</mn><mo>+</mo><mfenced><mrow><mn>0</mn><mo>,</mo><mn>15</mn></mrow></mfenced>"
    "<mo>&#xD7;</mo><mi>cos</mi><mfenced><mrow><mfenced><mrow><mn>0</mn><mo>,</mo><mn>06</mn>"
    "</mrow></mfenced><mo>&#xD7;</mo><mi>t</mi></mrow></mfenced></mrow></mfrac></math>"
)


def test_mathml_vira_latex_sem_marcacao():
    texto = limpar_texto(f"O raio é dado por {SATELITE} e varia.")

    assert "<math" not in texto and "</math>" not in texto
    assert r"\(r = \frac{5865}" in texto
    assert r"\times \cos" in texto
    assert texto.startswith("O raio é dado por")


def test_mathml_com_subscrito_e_entidade():
    texto = limpar_texto(
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mtext>M</mtext><mtext>W</mtext></msub></math>'
    )
    assert texto == r"\(\text{M}_{\text{W}}\)"


def test_entidades_e_tags_html_sao_limpas():
    assert limpar_texto("1 &#xD7; 2 &amp; 3") == "1 × 2 & 3"
    assert limpar_texto("<p>Oi</p> <b>mundo</b>") == "Oi mundo"


def test_texto_invalido_detecta_placeholder_e_mathml_truncado():
    assert texto_invalido("[[placeholder]]")
    assert texto_invalido("veja [IMAGE 0]")
    assert texto_invalido("<math><mi>x</mi>")
    assert texto_invalido("1 &amp; 2")
    assert not texto_invalido(r"função \(f(x) = x^2\)")


def test_referencia_figura_ignora_figuras_de_linguagem():
    assert referencia_figura("A figura ilustra um trilho de ar.")
    assert referencia_figura("Conforme a figura a seguir, calcule a área.")
    assert referencia_figura("No gráfico mostrado, a reta corta o eixo y.")
    assert not referencia_figura("O oximoro é uma figura de retórica.")
    assert not referencia_figura("A figura do índio no Brasil de hoje.")


def test_markdown_latex_escapa_cifrao_e_converte_delimitadores():
    assert markdown_latex(r"Seja \(x^2\) e R$ 10") == r"Seja $x^2$ e R\$ 10"


def test_para_inline_converte_dolares_sem_tocar_em_moeda():
    assert para_inline(r"Seja $x^2$ e R$ 10, calcule $$y$$ e \[z\].") == (
        r"Seja \(x^2\) e R$ 10, calcule \(y\) e \(z\)."
    )


def test_latex_valido_detecta_delimitadores_quebrados():
    assert latex_valido(r"Custa R$ \(5{,}00\) e \(x^2\).")
    assert latex_valido("Custa R$ 5,00.")
    assert not latex_valido(r"\(R\\)\ 5{,}00$")
    assert not latex_valido(r"\(x^2")


def test_loader_enem_challenge_converte_mathml_e_marca_figura(monkeypatch):
    linha = {
        "id": "2010_1",
        "nullified": None,
        "exam_year": "2010",
        "question": f"A figura ilustra o sistema: {SATELITE}",
        "choices": {"text": [SATELITE, "b", "c", "d", "e"]},
        "answerKey": "A",
    }
    monkeypatch.setattr(sources, "_load", lambda *a, **k: [linha])

    [bruta] = list(sources.carregar_enem_challenge())

    assert bruta.depende_imagem_sem_descricao
    assert "<math" not in bruta.enunciado and "<math" not in bruta.alternativas[0]
    assert r"\(" in bruta.enunciado and r"\(" in bruta.alternativas[0]
    assert not bruta.texto_invalido


def test_loader_enem_2022_2024_insere_descricao_bruta(monkeypatch):
    linha = {
        "id": "questao_01",
        "question": "[[placeholder]]\nO gráfico mostra as vendas.",
        "description": ["Descrição da imagem: ## Gráfico\nde vendas por dia"],
        "alternatives": ["a", "b", "c", "d", "e"],
        "label": "A",
    }
    monkeypatch.setattr(sources, "_load", lambda *a, **k: [linha])

    bruta = next(iter(sources.carregar_enem_2022_2024()))

    # o classificador é quem limpa prefixo/títulos; o loader só materializa a figura
    assert "[Figura: Descrição da imagem: ## Gráfico\nde vendas por dia]" in bruta.enunciado
    assert not bruta.depende_imagem_sem_descricao


def test_loader_bluex_troca_marcador_e_normaliza(monkeypatch):
    linha = {
        "id": "USP_2020_1",
        "subject": ["mathematics"],
        "alternatives_type": "string",
        "answer": "a",
        "question": "Veja o gráfico:\n[IMAGE 0]",
        "alternatives": ["a)  1", "b) 2", "c) 3", "d) 4", "e) 5"],
        "has_associated_images": True,
        "context_captions": ["Descrição da imagem: gráfico de f"],
        "blind_captions": [],
    }
    monkeypatch.setattr(sources, "_load", lambda *a, **k: [linha])

    [bruta] = list(sources.carregar_bluex())

    assert "[Figura: Descrição da imagem: gráfico de f]" in bruta.enunciado
    assert bruta.alternativas[0] == "1"
    assert not bruta.depende_imagem_sem_descricao and not bruta.texto_invalido


class _ClassificadorFixo:
    def classificar(self, questao: QuestaoBruta) -> Rotulo:
        return Rotulo(disciplina="matematica", topico="funcoes", conceitos=["funcao_afim"], dificuldade="media")


def _bruta(i: int, **kw) -> QuestaoBruta:
    dados = {
        "id": f"Q{i}",
        "fonte": "teste",
        "ano": 2020,
        "enunciado": f"Questão {i}",
        "alternativas": ["1", "2", "3", "4", "5"],
        "gabarito": "A",
    }
    dados.update(kw)
    return QuestaoBruta(**dados)


def test_ingestao_descarta_texto_invalido_e_imagem_sem_descricao(tmp_path):
    brutas = [
        _bruta(0),
        _bruta(1, texto_invalido=True),
        _bruta(2, depende_imagem_sem_descricao=True),
    ]

    questoes, stats = ingerir_questoes(
        brutas,
        _ClassificadorFixo(),
        RotuloCache(tmp_path / "rotulos.jsonl"),
        bateria_por_topico=0,
    )

    assert [q.id for q in questoes] == ["Q0"]
    assert stats["descartadas_formato"] == 1
    assert stats["descartadas_imagem"] == 1


def test_para_questao_usa_texto_normalizado_pelo_classificador():
    rotulo = Rotulo(
        disciplina="matematica",
        topico="funcoes",
        conceitos=["funcao_afim"],
        dificuldade="media",
        enunciado=r"Seja \(f(x) = x^{2}\).",
        alternativas=["1", "2", "3", "4", "5"],
    )

    questao = _para_questao(_bruta(0), rotulo)

    assert questao is not None and questao.enunciado == r"Seja \(f(x) = x^{2}\)."


def test_para_questao_descarta_normalizacao_suspeita():
    rotulo = Rotulo(
        disciplina="matematica",
        topico="funcoes",
        conceitos=["funcao_afim"],
        dificuldade="media",
        enunciado="<math>x</math>",
        alternativas=["1", "2", "3", "4", "5"],
    )

    questao = _para_questao(_bruta(0), rotulo)

    assert questao is not None and questao.enunciado == "Questão 0"
