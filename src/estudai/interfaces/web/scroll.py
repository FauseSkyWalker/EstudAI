"""Auto-scroll opcional para a página inteira. Usa JS puro devido a limitações do Streamlit"""

import streamlit as st

# LLM-generated below
_SCRIPT = """
<script>
(() => {
  // tick __TICK__
  const KEY = "estudai:atBottom";
  try {
    const win = window.parent;
    const doc = win.document;
    const scroller = () => {
      const main = doc.querySelector("main#stMain") || doc.querySelector("section.main") || doc.querySelector("main");
      return main && main.scrollHeight > main.clientHeight + 4
        ? main
        : (doc.scrollingElement || doc.documentElement);
    };
    const save = () => {
      const el = scroller();
      const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 80;
      win.sessionStorage.setItem(KEY, atBottom ? "1" : "0");
    };
    if (!win.__estudaiScrollBound) {
      win.addEventListener("scroll", save, {passive: true});
      const main = doc.querySelector("main#stMain") || doc.querySelector("section.main") || doc.querySelector("main");
      if (main) main.addEventListener("scroll", save, {passive: true});
      win.__estudaiScrollBound = true;
    }
    if (win.sessionStorage.getItem(KEY) === null) win.sessionStorage.setItem(KEY, "1");
    if (__NOVO__ && win.sessionStorage.getItem(KEY) !== "0") {
      const el = scroller();
      el.scrollTo({top: el.scrollHeight, behavior: "smooth"});
      setTimeout(save, 700);
    }
  } catch (e) {
    // iframe cross-origin ou DOM ainda não pronto
  }
})();
</script>
"""


def controle_scroll(novo: bool) -> None:
    tick = st.session_state.get("_scroll_tick", 0) + 1
    st.session_state["_scroll_tick"] = tick
    script = _SCRIPT.replace("__NOVO__", "true" if novo else "false").replace("__TICK__", str(tick))
    st.iframe(script, height=1)
