"""Interface web da sessão de estudo (Streamlit).

Rode com `uv run estudai web` (ou `uv run streamlit run src/estudai/interfaces/web/app.py`).
O ciclo roda numa thread (`SessaoWeb`); esta tela só desenha o estado e entrega as respostas.
"""

import streamlit as st
from dotenv import load_dotenv

from estudai.application.orquestrador import ResultadoCiclo
from estudai.application.sessoes import EventoSessao
from estudai.bootstrap import Container
from estudai.domain.correcao import RespostaAluno
from estudai.domain.curriculum import Topico
from estudai.domain.notificacao import Notificacao, TipoNotificacao
from estudai.domain.perfil import PerfilAluno
from estudai.domain.tutoria import TurnoTutor
from estudai.interfaces.web.bastidores import faixa_pipeline, quadro_ao_vivo, registro_completo
from estudai.interfaces.web.figuras import desenhar_figura
from estudai.interfaces.web.markdown import markdown_latex
from estudai.interfaces.web.sessao import EstadoSessao, Evento, Fase, Pedido, SessaoWeb, TipoEvento
from estudai.settings import Settings

# telemetria opcional (ex.: LANGSMITH_*) configurada no .env precisa estar no ambiente do processo
load_dotenv()

TOPICOS = {
    Topico.FUNCOES: "Funções",
    Topico.PROBABILIDADE: "Probabilidade",
    Topico.CINEMATICA: "Cinemática",
    Topico.DINAMICA: "Dinâmica",
}
FASES = {Fase.DIAGNOSTICO: "🔎 Diagnóstico", Fase.TUTORIA: "🧑‍🏫 Tutoria", Fase.PRATICA: "📝 Prática"}


@st.cache_resource
def container() -> Container:
    return Container(Settings())


# --- barra lateral ------------------------------------------------------------------------


def barra_lateral(c: Container, sessao: SessaoWeb | None) -> None:
    st.title("📐 EstudAI")
    st.caption("Sistema de tutoria em exatas para vestibulares")

    if sessao is not None and sessao.estado in (EstadoSessao.TRABALHANDO, EstadoSessao.AGUARDANDO_ALUNO):
        st.info(f"Sessão em andamento: **{sessao.aluno_id}** · {TOPICOS[sessao.topico]}")
        if st.button("Encerrar sessão", type="secondary", width="stretch"):
            sessao.cancelar()
            del st.session_state["sessao"]
            st.rerun()
        return

    aluno = st.text_input("Nome de usuário", value=st.session_state.get("aluno", ""))
    topico = st.selectbox("Tópico", list(TOPICOS), format_func=TOPICOS.get)
    faltando = c.dados_faltando()
    for caminho, comando in faltando:
        st.error(f"`{caminho}` não existe. Rode antes:\n\n`{comando}`")

    if st.button("Iniciar sessão", type="primary", width="stretch", disabled=bool(faltando) or not aluno):
        try:
            nova = SessaoWeb(c.orquestrador(), aluno.strip(), topico, sessoes=c.sessoes()).iniciar()
        except Exception as exc:  # noqa: BLE001 - ex.: credencial ausente
            st.error(f"Não foi possível iniciar: {exc}")
        else:
            st.session_state.pop("sessao_historica", None)
            st.session_state.update(sessao=nova, aluno=aluno)
            st.rerun()

    if aluno:
        perfil = mostrar_perfil(c, aluno.strip())
        mostrar_sessoes(c, aluno.strip(), perfil)


def mostrar_sessoes(c: Container, aluno: str, perfil: PerfilAluno | None) -> None:
    try:
        sessoes = c.sessoes().list(aluno)
    except ValueError:
        return
    st.subheader("Sessões anteriores")
    if not sessoes:
        if perfil is not None and perfil.episodios:
            st.caption("Apenas resumos: as transcrições antigas não foram salvas.")
            for episodio in reversed(perfil.episodios[-8:]):
                data = episodio.registrado_em.astimezone().strftime("%d/%m %H:%M")
                with st.expander(f"{data} · {TOPICOS.get(episodio.topico, episodio.topico)}"):
                    st.caption(f"{episodio.acertos}/{episodio.respondidos} acertos na prática")
                    if episodio.pontos_chave:
                        st.markdown("**Pontos-chave**\n" + "\n".join(f"- {p}" for p in episodio.pontos_chave))
                    if episodio.proximos_passos:
                        st.markdown("**Próximos passos**\n" + "\n".join(f"- {p}" for p in episodio.proximos_passos))
        else:
            st.caption("Nenhuma sessão concluída ainda.")
        return
    for sessao in sessoes:
        data = sessao.finalizada_em.astimezone().strftime("%d/%m %H:%M")
        status = "✅" if sessao.objetivo_atingido else "◻️"
        titulo = f"{status} {data} · {TOPICOS.get(sessao.topico, sessao.topico)}"
        with st.expander(titulo):
            st.caption(f"{sessao.acertos}/{sessao.respondidos} acertos na prática")
            if st.button("Abrir sessão", key=f"abrir-sessao-{sessao.id}", width="stretch"):
                st.session_state["sessao_historica"] = {
                    "aluno_id": sessao.aluno_id,
                    "sessao_id": sessao.id,
                }
                st.rerun()


def mostrar_perfil(c: Container, aluno: str) -> PerfilAluno | None:
    try:
        perfil = c.perfis().load(aluno)
    except ValueError:
        st.caption("Use só letras, números, `-`, `_` ou `.` no nome.")
        return None
    if perfil is None:
        st.caption("Perfil não encontrado: Diagnóstico inicial será aplicado")
        return None
    st.subheader("Seu perfil")
    st.caption(f"versão {perfil.versao}")
    for topico, estado in perfil.topicos.items():
        nivel = estado.nivel.value if estado.nivel else "não diagnosticado"
        st.markdown(f"**{TOPICOS.get(topico, topico)}**: {nivel}")
        if estado.lacunas:
            st.caption("Lacunas: " + ", ".join(estado.lacunas))
    return perfil


# --- área principal -----------------------------------------------------------------------


def tela_inicial() -> None:
    st.header("Bem-vindo(a)!")
    st.markdown(
        "A aplicação será conduzida em ciclos de aprendizado:\n\n"
        "1. **Diagnóstico** — algumas questões de vestibular para estimar seu nível inicial\n"
        "2. **Tutoria** — o tutor explica os pontos que você mais precisa e faz perguntas para testar seu aprendizado\n"
        "3. **Resumo** — um material de revisão do que foi ensinado\n"
        "4. **Prática** — exercícios personalizados, corrigidos com feedback dado sua resposta\n"
        "5. **Perfil atualizado** — o próximo ciclo se adapta ao que você errou\n\n"
        "Escolha seu nome e um tópico na barra lateral para começar."
    )


def desenhar_evento(evento: Evento) -> None:
    if evento.tipo == TipoEvento.TUTOR:
        turno: TurnoTutor = evento.conteudo
        with st.chat_message("assistant", avatar="🧑‍🏫"):
            if turno.avaliacao_resposta_anterior:
                st.caption(markdown_latex(turno.avaliacao_resposta_anterior))
            st.markdown(markdown_latex(turno.explicacao))
            for figura in turno.imagens:
                desenhar_figura(figura)
            st.markdown(f"**{markdown_latex(turno.pergunta_verificacao)}**")
    elif evento.tipo == TipoEvento.ALUNO:
        with st.chat_message("user"):
            st.markdown(evento.conteudo)
    elif evento.tipo == TipoEvento.ITEM:
        item_id, item, resposta = evento.conteudo
        with st.expander(f"Questão {item_id} · você marcou **{resposta.escolha or 'em branco'}**"):
            st.markdown(markdown_latex(item.formatar()))
            for figura in getattr(item, "imagens", []):
                desenhar_figura(figura)
    elif evento.tipo == TipoEvento.NOTIFICACAO:
        _desenhar_notificacao(evento.conteudo)
    else:
        st.info(str(evento.conteudo))


def _desenhar_notificacao(notificacao: Notificacao) -> None:
    texto = markdown_latex(notificacao.texto)
    if notificacao.tipo == TipoNotificacao.FEEDBACK_EXERCICIO:
        caixa = st.success if notificacao.correta else st.error
        status = "Correta" if notificacao.correta else "Incorreta"
        caixa(f"**{status}** ({notificacao.item_id}). {texto}")
    elif notificacao.tipo == TipoNotificacao.RESUMO:
        with st.container(border=True):
            st.markdown(texto.replace("# ", "#### 📘 ", 1))
    elif notificacao.tipo == TipoNotificacao.RELATORIO:
        with st.container(border=True):
            st.markdown(f"#### 📊 Relatório\n{texto}")
    elif notificacao.tipo == TipoNotificacao.ITEM_REMOVIDO:
        st.warning(texto)
    elif notificacao.tipo == TipoNotificacao.TUTORIA:  # fechamento do tutor
        with st.chat_message("assistant", avatar="🧑‍🏫"):
            st.markdown(texto)
            for figura in notificacao.figuras:
                desenhar_figura(figura)
    else:
        st.info(texto)


def _titulo_trilha(evento: Evento) -> str:
    if evento.tipo == TipoEvento.ITEM:
        return "Bastidores desta questão"
    return "Bastidores desta mensagem"


def historico(sessao: SessaoWeb) -> Fase | None:
    eventos = sessao.learner.eventos()
    transicao = sessao.learner.pedido_em_transicao if sessao.estado == EstadoSessao.TRABALHANDO else None
    if transicao is not None and transicao.item_id is not None:
        # a questão respondida ainda aparece como formulário desabilitado no rodapé
        eventos = [
            evento
            for evento in eventos
            if not (evento.tipo == TipoEvento.ITEM and evento.conteudo[0] == transicao.item_id)
        ]
    return historico_eventos(eventos)


def historico_eventos(eventos: list[Evento]) -> Fase | None:
    """Desenha os eventos agrupados por fase. Devolve a última fase exibida."""
    fase_atual = None
    for evento in eventos:
        if evento.fase != fase_atual:
            fase_atual = evento.fase
            st.subheader(FASES[fase_atual])
        registro_completo(list(evento.trilha), titulo=_titulo_trilha(evento))
        desenhar_evento(evento)
    return fase_atual


def _form_item(sessao: SessaoWeb, pedido: Pedido, *, disabled: bool = False) -> None:
    item = pedido.item
    if item is None:
        return
    chave_escolha = f"escolha-{sessao.id}-{pedido.item_id}"
    chave_resolucao = f"resolucao-{sessao.id}-{pedido.item_id}"

    def _responder_item() -> None:
        if sessao.learner.pendente is None:
            return
        sessao.learner.responder(
            RespostaAluno(
                item_id=pedido.item_id,
                escolha=st.session_state.get(chave_escolha),
                resolucao=st.session_state.get(chave_resolucao, ""),
            )
        )

    with st.form(key=f"item-{sessao.id}-{pedido.item_id}", border=True):
        st.markdown(f"**Questão {pedido.item_id}**")
        st.markdown(markdown_latex(item.enunciado))
        for figura in getattr(item, "imagens", []):
            desenhar_figura(figura)
        st.radio(
            "Alternativas",
            item.letras,
            index=None,
            key=chave_escolha,
            disabled=disabled,
            format_func=lambda letra: markdown_latex(f"{letra}) {item.alternativa(letra)}"),
        )
        st.text_area(
            "Como você resolveu? (Opcional para melhor diagnóstico)",
            key=chave_resolucao,
            disabled=disabled,
        )
        st.form_submit_button(
            "Enviar resposta",
            type="primary",
            on_click=None if disabled else _responder_item,
            disabled=disabled,
        )


def pedir_resposta(sessao: SessaoWeb, pedido: Pedido, ultima_fase: Fase | None) -> None:
    if pedido.turno is not None:
        chave = f"resposta-tutor-{sessao.id}"

        def _responder_tutor() -> None:
            if sessao.learner.pendente is None:
                return
            if texto := st.session_state.get(chave):
                sessao.learner.responder(texto)

        st.chat_input("Responda ao tutor…", key=chave, on_submit=_responder_tutor)
        return

    if pedido.fase != ultima_fase:
        st.subheader(FASES[pedido.fase])

    registro_completo(
        list(pedido.trilha),
        titulo="Bastidores desta questão",
        expandido=not sessao.learner.eventos(),
    )
    _form_item(sessao, pedido)


@st.fragment(run_every=1.0)
def area_agentes(sessao: SessaoWeb, eventos_vistos: int, ultima_fase: Fase | None) -> None:
    if sessao.estado == EstadoSessao.TRABALHANDO:
        if len(sessao.learner.eventos()) != eventos_vistos:
            st.rerun()
        quadro_ao_vivo(sessao.progresso_pendente())
        transicao = sessao.learner.pedido_em_transicao
        if transicao is not None and transicao.item is not None:
            st.caption("Resposta enviada; aguardando correção…")
            _form_item(sessao, transicao, disabled=True)
        else:
            st.chat_input("Aguarde os agentes…", disabled=True, key=f"resposta-tutor-{sessao.id}")
    elif sessao.estado == EstadoSessao.AGUARDANDO_ALUNO:
        pedir_resposta(sessao, sessao.learner.pendente, ultima_fase)
    else:
        st.rerun()


def resultado(sessao: SessaoWeb) -> None:
    _mostrar_resultado(sessao.resultado)
    if sessao.aviso_historico:
        st.warning(sessao.aviso_historico)


def _mostrar_resultado(r: ResultadoCiclo, *, concluido: bool = True) -> None:
    if concluido:
        st.success("Ciclo concluído! 🎉")
    acertos = sum(c.correta for c in r.correcoes)
    col1, col2, col3 = st.columns(3)
    col1.metric("Acertos na prática", f"{acertos}/{len(r.correcoes)}")
    col2.metric("Perfil", f"{r.versao_perfil_final} Avaliações Acumuladas")
    col3.metric("Custo", f"US$ {r.uso.total.cost_usd:.3f}")
    if r.diagnostico_final:
        d = r.diagnostico_final
        st.markdown(f"**Nível estimado em {TOPICOS[r.topico]}:** {d.nivel.value}")
        if d.lacunas:
            st.markdown("**Ainda vale reforçar:** " + ", ".join(d.lacunas))
    if r.relatorio:
        with st.container(border=True):
            st.markdown(f"#### 📊 Relatório\n{markdown_latex(r.relatorio.desempenho)}")
            for titulo, itens in (
                ("Pontos fortes", r.relatorio.pontos_fortes),
                ("A melhorar", r.relatorio.pontos_a_melhorar),
                ("Recomendações", r.relatorio.recomendacoes),
            ):
                if itens:
                    st.markdown(f"**{titulo}**\n" + "\n".join(f"- {markdown_latex(i)}" for i in itens))
    if r.removidos:
        st.warning(f"{len(r.removidos)} exercício(s) removido(s) por gabarito não confirmado pelo painel.")


def _evento_de_sessao(evento: EventoSessao) -> Evento:
    if evento.tipo == TipoEvento.TUTOR:
        if evento.turno is None:
            raise ValueError("evento de tutor sem turno")
        turno = evento.turno.model_copy(update={"imagens": list(evento.turno_imagens)})
        conteudo = turno
    elif evento.tipo == TipoEvento.ALUNO:
        conteudo = evento.aluno or ""
    elif evento.tipo == TipoEvento.ITEM:
        if evento.item is None or evento.resposta_item is None:
            raise ValueError("evento de item incompleto")
        conteudo = (evento.item.id, evento.item, evento.resposta_item)
    elif evento.tipo == TipoEvento.NOTIFICACAO:
        if evento.notificacao is None:
            raise ValueError("evento de notificação incompleto")
        conteudo = evento.notificacao.como_notificacao()
    else:
        raise ValueError(f"tipo de evento desconhecido: {evento.tipo}")
    return Evento(tipo=evento.tipo, fase=evento.fase, conteudo=conteudo, trilha=tuple(evento.trilha))


def tela_sessao_historica(c: Container, aluno_id: str, sessao_id: str) -> None:
    try:
        sessao = c.sessoes().load(aluno_id, sessao_id)
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Não foi possível abrir a sessão: {exc}")
        if st.button("Voltar ao início"):
            st.session_state.pop("sessao_historica", None)
            st.rerun()
        return

    if st.button("← Voltar ao início"):
        st.session_state.pop("sessao_historica", None)
        st.rerun()

    st.header(f"{TOPICOS.get(sessao.topico, sessao.topico)} · {sessao.aluno_id}")
    st.caption(f"Sessão concluída em {sessao.finalizada_em.astimezone():%d/%m/%Y %H:%M}")
    try:
        eventos = [_evento_de_sessao(evento) for evento in sessao.eventos]
    except ValueError as exc:
        st.error(f"O histórico desta sessão está incompleto: {exc}")
        return
    historico_eventos(eventos)
    _mostrar_resultado(sessao.resultado, concluido=False)


def painel_sessao(c: Container, sessao: SessaoWeb) -> None:
    st.header(f"{TOPICOS[sessao.topico]} · {sessao.aluno_id}")
    estado = sessao.estado
    em_andamento = estado in (EstadoSessao.TRABALHANDO, EstadoSessao.AGUARDANDO_ALUNO)
    faixa_pipeline(sessao.progresso(), em_andamento)
    ultima_fase = historico(sessao)
    if estado in (EstadoSessao.CONCLUIDA, EstadoSessao.ERRO, EstadoSessao.CANCELADA):
        registro_completo(sessao.progresso_pendente(), titulo="Bastidores finais do ciclo")
    if estado in (EstadoSessao.TRABALHANDO, EstadoSessao.AGUARDANDO_ALUNO):
        area_agentes(sessao, len(sessao.learner.eventos()), ultima_fase)
    elif estado == EstadoSessao.CONCLUIDA:
        resultado(sessao)
        if st.button("Novo ciclo neste tópico", type="primary"):
            st.session_state.pop("sessao_historica", None)
            st.session_state["sessao"] = SessaoWeb(
                c.orquestrador(), sessao.aluno_id, sessao.topico, sessoes=c.sessoes()
            ).iniciar()
            st.rerun()
    elif estado == EstadoSessao.ERRO:
        st.error("A sessão falhou. Detalhes abaixo.")
        st.code(sessao.erro)
    if estado in (EstadoSessao.CONCLUIDA, EstadoSessao.ERRO, EstadoSessao.CANCELADA) and st.button("Voltar ao início"):
        del st.session_state["sessao"]
        st.session_state.pop("sessao_historica", None)
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="EstudAI", page_icon="📐", layout="centered")
    c = container()
    sessao: SessaoWeb | None = st.session_state.get("sessao")
    with st.sidebar:
        barra_lateral(c, sessao)
    sessao = st.session_state.get("sessao")
    historica = st.session_state.get("sessao_historica")
    if sessao is None and historica is not None:
        tela_sessao_historica(c, historica["aluno_id"], historica["sessao_id"])
    elif sessao is None:
        tela_inicial()
    else:
        painel_sessao(c, sessao)


if __name__ == "__main__":
    main()
