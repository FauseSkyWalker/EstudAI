"""Bastidores na tela: em que etapa do pipeline o ciclo está e o que cada agente está fazendo."""

import streamlit as st

from estudai.application.progresso import ETAPAS, EventoProgresso, TipoProgresso

ICONES = {
    TipoProgresso.ETAPA: "🧭",
    TipoProgresso.PENSANDO: "💭",
    TipoProgresso.RESPONDEU: "✅",
    TipoProgresso.FERRAMENTA: "🛠️",
    TipoProgresso.FIGURA: "🖼️",
    TipoProgresso.NOTA: "ℹ️",
}


def _linha(evento: EventoProgresso, inicio: float) -> str:
    quem = f"**{evento.agente}** · " if evento.agente else ""
    return f"`+{evento.instante - inicio:4.0f}s` {ICONES[evento.tipo]} {quem}{evento.descricao}"


def faixa_pipeline(eventos: list[EventoProgresso], em_andamento: bool) -> None:
    """Etapas do grafo: ✅ concluída, ▶️ atual, ◻️ ainda não alcançada (algumas são puladas,
    ex.: o diagnóstico de quem já foi diagnosticado)."""
    vistas = [e.etapa for e in eventos if e.tipo == TipoProgresso.ETAPA]
    atual = vistas[-1] if vistas and em_andamento else None
    partes = []
    for chave, nome in ETAPAS.items():
        if chave == atual:
            partes.append(f"▶️ **{nome}**")
        elif chave in vistas:
            partes.append(f"✅ {nome}")
        else:
            partes.append(f"◻️ {nome}")
    st.caption(" → ".join(partes))


def quadro_ao_vivo(eventos: list[EventoProgresso], ultimos: int = 8) -> None:
    """O que está acontecendo agora; redesenhado a cada segundo enquanto os agentes trabalham."""
    if not eventos:
        st.status("Preparando o ciclo…", state="running")
        return
    atual = eventos[-1]
    etapa = ETAPAS.get(atual.etapa, "")
    rotulo = f"{etapa} — {atual.agente or 'sistema'}: {atual.descricao}" if etapa else atual.descricao
    with st.status(rotulo, state="running", expanded=True):
        for evento in eventos[-ultimos:]:
            st.markdown(_linha(evento, eventos[0].instante))


def registro_completo(eventos: list[EventoProgresso], expandido: bool = False) -> None:
    """Tudo o que os agentes fizeram no ciclo, agrupado por etapa."""
    if not eventos:
        return
    chamadas = sum(e.tipo == TipoProgresso.PENSANDO for e in eventos)
    with st.expander(f"🔍 Bastidores: o que os agentes fizeram ({chamadas} chamadas de modelo)", expanded=expandido):
        for evento in eventos:
            if evento.tipo == TipoProgresso.ETAPA:
                st.markdown(f"**{ICONES[evento.tipo]} {evento.descricao}**")
            else:
                st.markdown("&nbsp;&nbsp;&nbsp;" + _linha(evento, eventos[0].instante))
