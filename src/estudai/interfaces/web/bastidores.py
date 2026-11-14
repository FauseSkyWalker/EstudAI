"""Bastidores na tela: em que etapa do pipeline o ciclo está e o que cada agente está fazendo."""

import streamlit as st

from estudai.application.progresso import ETAPAS, EventoProgresso, TipoProgresso

ICONES = {
    TipoProgresso.ETAPA: "🧭",
    TipoProgresso.PENSANDO: "💭",
    TipoProgresso.RESPONDEU: "💬",
    TipoProgresso.FERRAMENTA: "🛠️",
    TipoProgresso.FIGURA: "🖼️",
    TipoProgresso.NOTA: "ℹ️",
}


def _linha(evento: EventoProgresso, inicio: float) -> str:
    quem = f"**{evento.agente}** · " if evento.agente else ""
    return f"`+{evento.instante - inicio:4.0f}s` {ICONES[evento.tipo]} {quem}{evento.descricao}"


def quadro_ao_vivo(eventos: list[EventoProgresso]) -> None:
    """Tudo o que aconteceu desde a última mensagem; redesenhado a cada segundo enquanto trabalham."""
    if not eventos:
        st.caption("⏳ Preparando o ciclo…")
        return
    atual = eventos[-1]
    etapa = ETAPAS.get(atual.etapa, "")
    rotulo = f"{etapa} — {atual.agente or 'sistema'}: {atual.descricao}" if etapa else atual.descricao
    with st.container(border=True):
        st.markdown(f"**⏳ {rotulo}**")
        for evento in eventos:
            st.markdown(_linha(evento, eventos[0].instante))


def registro_completo(
    eventos: list[EventoProgresso],
    expandido: bool = False,
    titulo: str = "Bastidores: o que os agentes fizeram",
) -> None:
    """Tudo o que os agentes fizeram num trecho (o ciclo inteiro ou uma mensagem)."""
    if not eventos:
        return
    chamadas = sum(e.tipo == TipoProgresso.PENSANDO for e in eventos)
    with st.expander(f"🔍 {titulo} ({chamadas} chamadas de modelo)", expanded=expandido):
        for evento in eventos:
            if evento.tipo == TipoProgresso.ETAPA:
                st.markdown(f"**{ICONES[evento.tipo]} {evento.descricao}**")
            else:
                st.markdown("&nbsp;&nbsp;&nbsp;" + _linha(evento, eventos[0].instante))
