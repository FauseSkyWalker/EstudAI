"""Interface web da sessão de estudo (Streamlit).

Rode com `uv run estudai web` (ou `uv run streamlit run src/estudai/interfaces/web/app.py`).
O ciclo roda numa thread (`SessaoWeb`); esta tela só desenha o estado e entrega as respostas.
"""

import re

import streamlit as st

from estudai.bootstrap import Container
from estudai.domain.correcao import RespostaAluno
from estudai.domain.curriculum import Topico
from estudai.domain.tutoria import TurnoTutor
from estudai.interfaces.web.sessao import EstadoSessao, Evento, Fase, Pedido, SessaoWeb, TipoEvento
from estudai.settings import Settings

TOPICOS = {
    Topico.FUNCOES: "Funções",
    Topico.PROBABILIDADE: "Probabilidade",
    Topico.CINEMATICA: "Cinemática",
    Topico.DINAMICA: "Dinâmica",
}
FASES = {Fase.DIAGNOSTICO: "🔎 Diagnóstico", Fase.TUTORIA: "🧑‍🏫 Tutoria", Fase.PRATICA: "📝 Prática"}
_FEEDBACK = re.compile(r"^\[(?P<id>[^\]]+)\] (?P<status>Correta|Incorreta)\.\s*(?P<texto>.*)", re.DOTALL)


@st.cache_resource
def container() -> Container:
    return Container(Settings())


# --- barra lateral ------------------------------------------------------------------------


def barra_lateral(c: Container, sessao: SessaoWeb | None) -> None:
    st.title("📐 EstudAI")
    st.caption("Tutoria adaptativa em Matemática e Física para vestibulares")

    if sessao is not None and sessao.estado in (EstadoSessao.TRABALHANDO, EstadoSessao.AGUARDANDO_ALUNO):
        st.info(f"Sessão em andamento: **{sessao.aluno_id}** · {TOPICOS[sessao.topico]}")
        if st.button("Encerrar sessão", type="secondary", width="stretch"):
            sessao.cancelar()
            del st.session_state["sessao"]
            st.rerun()
        return

    aluno = st.text_input("Seu nome (identifica o seu perfil)", value=st.session_state.get("aluno", ""))
    topico = st.selectbox("Tópico", list(TOPICOS), format_func=TOPICOS.get)
    faltando = c.dados_faltando()
    for caminho, comando in faltando:
        st.error(f"`{caminho}` não existe. Rode antes:\n\n`{comando}`")

    if st.button("Iniciar sessão", type="primary", width="stretch", disabled=bool(faltando) or not aluno):
        try:
            nova = SessaoWeb(c.orquestrador(), aluno.strip(), topico).iniciar()
        except Exception as exc:  # noqa: BLE001 - ex.: credencial ausente
            st.error(f"Não foi possível iniciar: {exc}")
        else:
            st.session_state.update(sessao=nova, aluno=aluno)
            st.rerun()

    if aluno:
        mostrar_perfil(c, aluno.strip())


def mostrar_perfil(c: Container, aluno: str) -> None:
    try:
        perfil = c.perfis().load(aluno)
    except ValueError:
        st.caption("Use só letras, números, `-`, `_` ou `.` no nome.")
        return
    if perfil is None:
        st.caption("Aluno novo: a primeira sessão começa com um diagnóstico.")
        return
    st.subheader("Seu perfil")
    st.caption(f"versão {perfil.versao}")
    for topico, estado in perfil.topicos.items():
        nivel = estado.nivel.value if estado.nivel else "não diagnosticado"
        st.markdown(f"**{TOPICOS.get(topico, topico)}**: {nivel}")
        if estado.lacunas:
            st.caption("Lacunas: " + ", ".join(estado.lacunas))


# --- área principal -----------------------------------------------------------------------


def tela_inicial() -> None:
    st.header("Bem-vindo(a)!")
    st.markdown(
        "Cada sessão é um ciclo completo:\n\n"
        "1. **Diagnóstico** — algumas questões de vestibular para estimar seu nível (só na primeira vez);\n"
        "2. **Tutoria** — o tutor explica os pontos que você mais precisa e faz perguntas;\n"
        "3. **Resumo** — um material de revisão do que foi ensinado;\n"
        "4. **Prática** — exercícios personalizados, corrigidos com feedback;\n"
        "5. **Perfil atualizado** — o próximo ciclo se adapta ao que você errou.\n\n"
        "Escolha seu nome e um tópico na barra lateral para começar."
    )


def desenhar_evento(evento: Evento) -> None:
    if evento.tipo == TipoEvento.TUTOR:
        turno: TurnoTutor = evento.conteudo
        with st.chat_message("assistant", avatar="🧑‍🏫"):
            if turno.avaliacao_resposta_anterior:
                st.caption(turno.avaliacao_resposta_anterior)
            st.markdown(turno.explicacao)
            st.markdown(f"**{turno.pergunta_verificacao}**")
    elif evento.tipo == TipoEvento.ALUNO:
        with st.chat_message("user"):
            st.markdown(evento.conteudo)
    elif evento.tipo == TipoEvento.ITEM:
        item_id, item, resposta = evento.conteudo
        with st.expander(f"Questão {item_id} · você marcou **{resposta.escolha or 'em branco'}**"):
            st.markdown(item.formatar())
    elif feedback := _FEEDBACK.match(evento.conteudo):
        caixa = st.success if feedback["status"] == "Correta" else st.error
        caixa(f"**{feedback['status']}** ({feedback['id']}). {feedback['texto']}")
    elif evento.conteudo.startswith("# "):  # resumo de revisão
        with st.container(border=True):
            st.markdown(evento.conteudo.replace("# ", "#### 📘 ", 1))
    elif evento.fase == Fase.TUTORIA:  # fechamento do tutor
        with st.chat_message("assistant", avatar="🧑‍🏫"):
            st.markdown(evento.conteudo)
    else:
        st.info(evento.conteudo)


def historico(sessao: SessaoWeb) -> Fase | None:
    """Desenha os eventos agrupados por fase; devolve a última fase exibida."""
    fase_atual = None
    for evento in sessao.learner.eventos():
        if evento.fase != fase_atual:
            fase_atual = evento.fase
            st.subheader(FASES[fase_atual])
        desenhar_evento(evento)
    return fase_atual


def pedir_resposta(sessao: SessaoWeb, pedido: Pedido, ultima_fase: Fase | None) -> None:
    if pedido.turno is not None:  # o turno já aparece no histórico; falta só a resposta
        if texto := st.chat_input("Responda ao tutor…"):
            sessao.learner.responder(texto)
            st.rerun()
        return

    item = pedido.item
    if pedido.fase != ultima_fase:
        st.subheader(FASES[pedido.fase])
    with st.form(key=f"item-{pedido.item_id}", border=True):
        st.markdown(f"**Questão {pedido.item_id}**")
        st.markdown(item.enunciado)
        escolha = st.radio(
            "Alternativas",
            item.letras,
            index=None,
            format_func=lambda letra: f"{letra}) {item.alternativa(letra)}",
        )
        resolucao = st.text_area("Como você resolveu? (opcional — ajuda o diagnóstico)")
        if st.form_submit_button("Enviar resposta", type="primary"):
            sessao.learner.responder(RespostaAluno(item_id=pedido.item_id, escolha=escolha, resolucao=resolucao))
            st.rerun()


@st.fragment(run_every=1.0)
def aguardar_agentes(sessao: SessaoWeb, eventos_vistos: int) -> None:
    """Enquanto os agentes trabalham, verifica o estado a cada segundo e redesenha quando muda."""
    if sessao.estado != EstadoSessao.TRABALHANDO or len(sessao.learner.eventos()) != eventos_vistos:
        st.rerun()
    st.caption("⏳ Os agentes estão trabalhando…")


def resultado(sessao: SessaoWeb) -> None:
    r = sessao.resultado
    st.success("Ciclo concluído! 🎉")
    acertos = sum(c.correta for c in r.correcoes)
    col1, col2, col3 = st.columns(3)
    col1.metric("Acertos na prática", f"{acertos}/{len(r.correcoes)}")
    col2.metric("Perfil", f"v{r.versao_perfil_inicial} → v{r.versao_perfil_final}")
    col3.metric("Custo", f"US$ {r.uso.total.cost_usd:.3f}")
    if r.diagnostico_final:
        d = r.diagnostico_final
        st.markdown(f"**Nível estimado em {TOPICOS[r.topico]}:** {d.nivel.value}")
        if d.lacunas:
            st.markdown("**Ainda vale reforçar:** " + ", ".join(d.lacunas))
    if r.relatorio:
        with st.container(border=True):
            st.markdown(f"#### 📊 Relatório\n{r.relatorio.desempenho}")
            for titulo, itens in (
                ("Pontos fortes", r.relatorio.pontos_fortes),
                ("A melhorar", r.relatorio.pontos_a_melhorar),
                ("Recomendações", r.relatorio.recomendacoes),
            ):
                if itens:
                    st.markdown(f"**{titulo}**\n" + "\n".join(f"- {i}" for i in itens))
    if r.removidos:
        st.warning(f"{len(r.removidos)} exercício(s) removido(s) por gabarito não confirmado pelo painel.")


def painel_sessao(c: Container, sessao: SessaoWeb) -> None:
    st.header(f"{TOPICOS[sessao.topico]} · {sessao.aluno_id}")
    ultima_fase = historico(sessao)
    estado = sessao.estado
    if estado == EstadoSessao.AGUARDANDO_ALUNO:
        pedir_resposta(sessao, sessao.learner.pendente, ultima_fase)
    elif estado == EstadoSessao.TRABALHANDO:
        aguardar_agentes(sessao, len(sessao.learner.eventos()))
    elif estado == EstadoSessao.CONCLUIDA:
        resultado(sessao)
        if st.button("Novo ciclo neste tópico", type="primary"):
            st.session_state["sessao"] = SessaoWeb(c.orquestrador(), sessao.aluno_id, sessao.topico).iniciar()
            st.rerun()
    elif estado == EstadoSessao.ERRO:
        st.error("A sessão falhou. Detalhes abaixo.")
        st.code(sessao.erro)
    if estado in (EstadoSessao.CONCLUIDA, EstadoSessao.ERRO, EstadoSessao.CANCELADA) and st.button("Voltar ao início"):
        del st.session_state["sessao"]
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="EstudAI", page_icon="📐", layout="centered")
    c = container()
    sessao: SessaoWeb | None = st.session_state.get("sessao")
    with st.sidebar:
        barra_lateral(c, sessao)
    sessao = st.session_state.get("sessao")
    if sessao is None:
        tela_inicial()
    else:
        painel_sessao(c, sessao)


main()
