"""Pipeline de ingestão do banco de questões.

fontes -> exclui dependentes de imagem sem descrição -> rotula por LLM (com cache, para
retomar execuções interrompidas) -> mantém Matemática/Física dos 4 tópicos -> separa a
bateria fixa de pré/pós-teste -> grava `questoes.jsonl`.
"""

import json
import random
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import ValidationError

from estudai.agents.base import AgentOutputError
from estudai.domain.curriculum import Disciplina, Topico
from estudai.domain.questoes import Particao, Questao
from estudai.infrastructure.ingestion.classificador import ClassificadorQuestoes, Rotulo
from estudai.infrastructure.ingestion.sources import QuestaoBruta
from estudai.ports.llm import LLMError


class RotuloCache:
    def __init__(self, caminho: Path) -> None:
        self._caminho = caminho
        self._dados: dict[str, Rotulo] = {}
        if caminho.exists():
            for linha in caminho.read_text(encoding="utf-8").splitlines():
                registro = json.loads(linha)
                self._dados[registro["id"]] = Rotulo.model_validate(registro["rotulo"])

    def get(self, questao_id: str) -> Rotulo | None:
        return self._dados.get(questao_id)

    def put(self, questao_id: str, rotulo: Rotulo) -> None:
        self._dados[questao_id] = rotulo
        self._caminho.parent.mkdir(parents=True, exist_ok=True)
        with self._caminho.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": questao_id, "rotulo": rotulo.model_dump(mode="json")}, ensure_ascii=False) + "\n")


def _para_questao(bruta: QuestaoBruta, rotulo: Rotulo) -> Questao | None:
    if rotulo.disciplina == "outra" or rotulo.topico == "outro":
        return None
    topico = Topico(rotulo.topico)
    if topico.disciplina != Disciplina(rotulo.disciplina):
        return None  # rótulo incoerente
    try:
        return Questao(
            id=bruta.id,
            fonte=bruta.fonte,
            ano=bruta.ano,
            disciplina=topico.disciplina,
            topico=topico,
            conceitos=rotulo.conceitos,
            dificuldade=rotulo.dificuldade,
            enunciado=bruta.enunciado,
            alternativas=bruta.alternativas,
            gabarito=bruta.gabarito,
            descricoes_imagens=bruta.descricoes_imagens,
        )
    except ValidationError:
        return None  # p.ex. número de alternativas fora do padrão


def separar_bateria(questoes: list[Questao], por_topico: int, seed: int = 0) -> list[Questao]:
    """Marca `por_topico` questões de cada tópico como bateria, maximizando a cobertura de conceitos."""
    rng = random.Random(seed)
    por_topico_q: dict[Topico, list[Questao]] = defaultdict(list)
    for q in questoes:
        if q.conceitos:
            por_topico_q[q.topico].append(q)
    bateria: set[str] = set()
    for lista in por_topico_q.values():
        rng.shuffle(lista)
        cobertos: set[str] = set()
        for _ in range(min(por_topico, len(lista))):
            melhor = max((q for q in lista if q.id not in bateria), key=lambda q: len(set(q.conceitos) - cobertos))
            bateria.add(melhor.id)
            cobertos |= set(melhor.conceitos)
    return [q.model_copy(update={"particao": Particao.BATERIA}) if q.id in bateria else q for q in questoes]


def ingerir_questoes(
    brutas: Iterable[QuestaoBruta],
    classificador: ClassificadorQuestoes,
    cache: RotuloCache,
    *,
    bateria_por_topico: int = 8,
    seed: int = 0,
    workers: int = 8,
    progresso: Callable[[str], None] = lambda _: None,
) -> tuple[list[Questao], dict[str, int]]:
    stats: dict[str, int] = defaultdict(int)
    elegiveis = []
    for b in brutas:
        stats["lidas"] += 1
        if b.depende_imagem_sem_descricao:
            stats["descartadas_imagem"] += 1
        else:
            elegiveis.append(b)

    def rotular(b: QuestaoBruta) -> tuple[QuestaoBruta, Rotulo | None]:
        if (rotulo := cache.get(b.id)) is not None:
            return b, rotulo
        try:
            return b, classificador.classificar(b)
        except (AgentOutputError, LLMError):
            return b, None

    questoes = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (bruta, rotulo) in enumerate(pool.map(rotular, elegiveis), start=1):
            if rotulo is None:
                stats["falha_rotulagem"] += 1
                continue
            if cache.get(bruta.id) is None:
                cache.put(bruta.id, rotulo)
            if (questao := _para_questao(bruta, rotulo)) is None:
                stats["fora_do_escopo"] += 1
            else:
                questoes.append(questao)
            if i % 50 == 0:
                progresso(f"{i}/{len(elegiveis)} rotuladas")

    questoes = separar_bateria(questoes, bateria_por_topico, seed)
    for q in questoes:
        stats[f"{q.topico.value}:{q.particao.value}"] += 1
    return questoes, dict(stats)
