"""Montagem dos documentos da base vetorial a partir dos trechos e do banco de questões.

Questões da bateria de avaliação NUNCA entram no índice (evita vazamento para o pós-teste),
e o gabarito das questões de referência não é indexado: elas servem como modelo de estilo.
"""

from estudai.domain.material import TipoDocumento
from estudai.domain.questoes import Particao, Questao
from estudai.infrastructure.ingestion.wikibooks import Trecho
from estudai.ports.knowledge import Document


def montar_documentos(trechos: list[Trecho], questoes: list[Questao]) -> list[Document]:
    documentos = [
        Document(
            id=t.id,
            tipo=TipoDocumento.MATERIAL,
            fonte=f"wikibooks:{t.titulo}",
            texto=t.texto,
            topico=t.topico,
            metadados={"secao": t.secao},
        )
        for t in trechos
    ]
    documentos += [
        Document(
            id=q.id,
            tipo=TipoDocumento.QUESTAO,
            fonte=f"{q.fonte}:{q.ano}",
            texto=q.formatar(com_gabarito=False),
            topico=q.topico,
            metadados={"conceitos": q.conceitos, "dificuldade": q.dificuldade.value},
        )
        for q in questoes
        if q.particao == Particao.BANCO
    ]
    return documentos
