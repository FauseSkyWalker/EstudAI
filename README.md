# EstudAI

**Diagnóstico, Ensino e Prática: um Pipeline Multiagente de Tutoria Adaptativa em Matemática e Física**
MO810A/MC959A — Sistemas Multiagentes com Modelos de Linguagem (IC/UNICAMP, 2026)
Fause Mascarenhas · Vinícius Faria

Implementação da metodologia da Fase 2: seis agentes especializados coordenados por um
orquestrador centralizado e sequencial sobre um perfil de aluno único, persistente e versionado.

## Arquitetura

O código segue Clean Architecture: as dependências apontam sempre para dentro, e só a raiz de
composição (`bootstrap.py`) conhece as implementações concretas.

```
interfaces/ ──► bootstrap.py ──► infrastructure/
     │                               │
     ▼                               ▼
application/ ──► agents/ ──► ports/ ──► domain/
```

| Camada | Pasta | Responsabilidade | Depende de |
|---|---|---|---|
| Domínio | `domain/` | Entidades e regras puras: currículo, perfil versionado, itens, planejamento determinístico, protocolo de mensagens | — |
| Portas | `ports/` | Contratos: `LLMClient`, `Embedder`, `KnowledgeBase`, `QuestionBank`, `ProfileRepository`, `CodeExecutor`, `Learner` | domínio |
| Agentes | `agents/` | Os 6 agentes da Tabela 2 + resolutor do painel; saída estruturada com reparo; laço ReAct | domínio, portas |
| Aplicação | `application/` | Orquestrador, validação por consenso, estratégias de prática, telemetria, log de mensagens | agentes, domínio, portas |
| Infraestrutura | `infrastructure/` | Gemini, OpenAI-compatível, retry/telemetria, BM25 + vetorial, JSON, sandbox, ingestão | portas (implementa) |
| Avaliação | `evaluation/` | Personas, aluno simulado (DINA + LLM), juízes, experimento C1/C2/C3, estatística | aplicação, portas |
| Interfaces | `interfaces/` | CLI (`typer`) e aluno no terminal | bootstrap |

Consequências práticas:
- **Trocar de fornecedor de LLM** = um adaptador novo em `infrastructure/llm/` + uma linha em `bootstrap.py`.
- **Mesmo pipeline em produção e na avaliação**: o orquestrador só conversa com a porta `Learner`,
  implementada pelo aluno real (`interfaces/console.py`) e pelo simulado (`evaluation/aluno_simulado.py`).
- **Condições experimentais sem `if` no orquestrador**: C2 é `ConfigSessao(adaptativo=False)`;
  C3 troca a estratégia `ExerciseProvider` (gerados → banco real).
- **Testável offline**: todos os agentes recebem `LLMClient` por injeção; os testes usam um LLM roteirizado.

### Mapeamento metodologia → código

| Metodologia | Código |
|---|---|
| Agente Diagnosticador | `agents/diagnosticador.py` (sondagem por cobertura de conceitos + estimativa LLM com fallback determinístico) |
| Agente Tutor por Disciplina | `agents/tutor.py` (uma instância por `Disciplina`) |
| Agente de Conteúdo (ReAct, RAG) | `agents/conteudo.py` (busca semântica/textual/híbrida; só seleciona ids → fidelidade por construção) |
| Agente de Resumo | `agents/resumo.py` (resumo de revisão + relatório da prática) |
| Agente Gerador de Exercícios (ReAct) | `agents/gerador.py` (ferramentas `consultar_conteudo` e `executar_python`) |
| Agente Corretor | `agents/corretor.py` (corretude determinística; LLM só diagnostica o raciocínio) |
| Painel de resolutores + verificação simbólica | `agents/resolutor.py`, `application/validacao.py` |
| Orquestração centralizada e sequencial | `application/orquestrador.py` |
| Perfil persistente versionado | `domain/perfil.py`, `infrastructure/persistence/json_profiles.py` |
| Protocolo de mensagens JSON (Apêndice A) | `domain/mensagens.py`, `application/mensageria.py` |
| Retry / reenvio com instrução de correção | `infrastructure/llm/decorators.py`, `agents/base.py` |
| Níveis "Fronteira" / "Econômico" | `settings.py` (`frontier`, `economic`), `bootstrap.py` |
| Sandbox Python (sympy, sem rede, timeout) | `infrastructure/sandbox/python_executor.py` |
| Bases BLUEX / ENEM Challenge / ENEM 2022–24 | `infrastructure/ingestion/sources.py` |
| Wikibooks (curadoria manual) | `config/wikibooks.toml`, `infrastructure/ingestion/wikibooks.py` |
| Estudantes simulados com estado externo | `evaluation/personas.py`, `evaluation/aluno_simulado.py` |
| Juízes LLM (rubrica de tutoria, qualidade de itens, transição de conceito) | `evaluation/juizes.py` |
| Condições C1/C2/C3, Wilcoxon, bootstrap, κ | `evaluation/experimento.py`, `evaluation/estatistica.py`, `evaluation/relatorio.py` |

### Ciclo de uma sessão

```
[diagnóstico inicial]  Diagnosticador escolhe questões do banco → aluno responde → nível/lacunas → perfil v+1
tutoria                objetivo derivado do perfil → Conteúdo (RAG) → Tutor ⇄ aluno (até objetivo ou limite)
resumo                 Agente de Resumo → material de revisão
prática                Gerador (ReAct) → verificação simbólica + painel → itens aprovados
correção               aluno resolve → Corretor (gabarito suspeito → painel → item removido e notificado)
atualização            Diagnosticador usa as correções → perfil v+1 → relatório da prática
```

## Instalação

```bash
uv sync --all-extras          # ou: uv sync --extra gemini --extra openai --extra ingest
cp .env.example .env          # preencha chaves e modelos
```

O modelo de fronteira (Muse Spark 1.3) é acessado por um endpoint compatível com a API da
OpenAI: preencha `ESTUDAI_FRONTIER__BASE_URL` e `ESTUDAI_FRONTIER__MODEL`. Para testar o
pipeline só com Gemini, use `ESTUDAI_FRONTIER__PROVIDER=gemini` e um modelo Gemini.

## Uso

```bash
uv run estudai ingest-questions               # baixa, filtra e rotula (modelo econômico; retomável via cache)
uv run estudai ingest-wikibooks               # baixa e fatia o material curado
uv run estudai build-index                    # base vetorial (sem a bateria de avaliação)

uv run estudai estudar --aluno ana --topico funcoes     # sessão interativa
uv run estudai perfil ana                               # histórico de versões do perfil

uv run estudai experimento --saida results/exp1.jsonl   # 3 condições × 3 personas × 4 tópicos
uv run estudai relatorio results/exp1.jsonl --orcamento-usd 0.50
```

O experimento grava cada execução assim que termina; rodar de novo o mesmo comando retoma de onde parou.

## Testes

```bash
uv run pytest
```

Os testes não fazem chamadas de rede: usam um LLM roteirizado (`tests/fakes.py`), o embedder
por hashing e o executor de sandbox real.

## Limitações conhecidas (além das da Seção 5.4)

- A sandbox usa audit hooks (PEP 578), que não são fronteira de segurança contra código
  adversarial; em produção, rode o executor em contêiner sem rede.
- A verificação simbólica só confirma itens cuja resposta é um número; os demais dependem
  apenas do consenso do painel (com limiar de unanimidade, configurável).
- O embedder por hashing (`ESTUDAI_EMBEDDING_PROVIDER=hashing`) é só para testes/uso offline.
