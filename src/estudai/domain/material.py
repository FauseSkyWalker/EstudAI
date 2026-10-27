"""Material de apoio recuperado da base de conhecimento pelo Agente de Conteúdo."""

from enum import StrEnum

from pydantic import BaseModel, Field


class TipoDocumento(StrEnum):
    MATERIAL = "material"  # trecho didático (Wikibooks)
    QUESTAO = "questao"  # questão de referência do banco


class TrechoRecuperado(BaseModel):
    doc_id: str
    tipo: TipoDocumento
    fonte: str
    texto: str
    score: float


class PacoteMaterial(BaseModel):
    """Trechos ranqueados. O texto vem sempre da base (nunca reescrito pelo LLM), o que
    garante fidelidade às fontes por construção."""

    pedido: str
    trechos: list[TrechoRecuperado] = Field(default_factory=list)
    questoes_referencia: list[TrechoRecuperado] = Field(default_factory=list)

    @property
    def referencias(self) -> list[str]:
        return [t.doc_id for t in self.trechos + self.questoes_referencia]

    def como_contexto(self, max_chars: int = 6000) -> str:
        partes, total = [], 0
        for t in self.trechos + self.questoes_referencia:
            bloco = f"[{t.doc_id}] ({t.fonte})\n{t.texto}"
            if total + len(bloco) > max_chars:
                break
            partes.append(bloco)
            total += len(bloco)
        return "\n\n".join(partes) if partes else "(nenhum material recuperado)"
