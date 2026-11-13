"""Desenho de figuras na interface web via Streamlit."""

import streamlit as st

from estudai.domain.figura import Figura, FormatoFigura


def desenhar_figura(figura: Figura) -> None:
    if figura.formato == FormatoFigura.MERMAID:
        st.mermaid_chart(figura.conteudo, alt=figura.descricao)
    else:
        st.image(figura.png, caption=figura.descricao, alt=figura.descricao)
