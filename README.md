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
| Portas | `ports/` | Contratos: `Embedder`, `KnowledgeBase`, `QuestionBank`, `ProfileRepository`, `CodeExecutor`, `Learner` | domínio |
| Agentes | `agents/` | Os 6 agentes da Tabela 2 + resolutor do painel; saída estruturada com reparo; ferramentas nativas (LangChain) | domínio, portas |
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
- **Testável offline**: todos os agentes recebem um `BaseChatModel` por injeção; os testes usam um LLM roteirizado.

### Mapeamento metodologia → código

| Metodologia | Código |
|---|---|
| Agente Diagnosticador | `agents/diagnosticador.py` (sondagem por cobertura de conceitos + estimativa LLM com fallback determinístico) |
| Agente Tutor por Disciplina | `agents/tutor.py` (uma instância por `Disciplina`) |
| Agente de Conteúdo (LangChain + ferramentas, RAG) | `agents/conteudo.py` (busca semântica/textual/híbrida; só seleciona ids → fidelidade por construção) |
| Agente de Resumo | `agents/resumo.py` (resumo de revisão + relatório da prática) |
| Agente Gerador de Exercícios (LangChain + ferramentas) | `agents/gerador.py` (ferramentas `consultar_conteudo` e `executar_python`) |
| Agente Corretor | `agents/corretor.py` (corretude determinística; LLM só diagnostica o raciocínio) |
| Painel de resolutores + verificação simbólica | `agents/resolutor.py`, `application/validacao.py` |
| Orquestração centralizada e sequencial | `application/orquestrador.py` |
| Perfil persistente versionado | `domain/perfil.py`, `infrastructure/persistence/json_profiles.py` |
| Protocolo de mensagens JSON (Apêndice A) | `domain/mensagens.py`, `application/mensageria.py` |
| Retry / reenvio com instrução de correção | `agents/base.py` |
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
prática                Gerador (LangChain + ferramentas) → verificação simbólica + painel → itens aprovados
correção               aluno resolve → Corretor (gabarito suspeito → painel → item removido e notificado)
atualização            Diagnosticador usa as correções → perfil v+1 → relatório da prática
```

## Instalação

```bash
uv sync --extra ingest --extra web   # liste todos os extras: o uv sync remove os que ficarem de fora
                                     # (--all-extras também instala torch/sentence-transformers, pesado)
cp .env.example .env          # preencha chaves e modelos
```

### Modelos

Cada nível (`ECONOMIC`, `FRONTIER`, `JUDGES`) escolhe um provedor no `.env` (exemplos comentados
no `.env.example`):

| `PROVIDER` | Quando usar | Chave |
|---|---|---|
| `openrouter` | Mais simples: a configuração da metodologia (Gemini 3 Flash + Muse Spark 1.3) com uma chave só | `ESTUDAI_OPENROUTER_API_KEY` |
| `gemini` | Direto na API do Google (o plano gratuito tem cotas muito baixas) | `ESTUDAI_GEMINI_API_KEY` |
| `openai_compat` | Qualquer endpoint compatível com a OpenAI (preencha `BASE_URL`) | `ESTUDAI_OPENAI_API_KEY` |
| `claude_cli` | Usa o Claude Code CLI instalado e logado na máquina | nenhuma |

Os embeddings da busca usam o Gemini (`ESTUDAI_GEMINI_API_KEY`) qualquer que seja o provedor de chat.
Depois de configurar, confira se todos os modelos respondem:

```bash
uv run estudai verificar-modelos
```

## Preparar os dados (uma vez)

```bash
uv run estudai ingest-wikibooks               # baixa e fatia o material didático curado
uv run estudai ingest-questions               # baixa, filtra e rotula as questões (retomável via cache)
uv run estudai build-index                    # base vetorial (sem a bateria de avaliação)
```

Para um teste rápido e barato, use `ingest-questions --limite 80`. As interfaces abaixo se recusam
a iniciar (e dizem qual comando falta) enquanto o banco de questões ou o índice não existirem.

## Interfaces

As duas interfaces rodam o mesmo ciclo — diagnóstico (só na primeira sessão do aluno), tutoria,
resumo, exercícios com correção e perfil atualizado — e salvam o mesmo perfil: dá para começar
numa e continuar na outra usando o mesmo nome de aluno.

### Navegador (Streamlit)

```bash
uv run estudai web                 # abre http://localhost:8501
uv run estudai web --porta 8502    # se a porta estiver ocupada
```

1. Na barra lateral, digite seu nome (identifica o perfil) e escolha o tópico.
2. Clique em **Iniciar sessão**. As questões aparecem como formulário (alternativas + resolução
   opcional) e a tutoria como chat; enquanto os agentes trabalham, a tela mostra
   "⏳ Os agentes estão trabalhando…" e se atualiza sozinha.
3. No fim do ciclo aparecem acertos, nível estimado, relatório e custo; **Novo ciclo** continua no
   mesmo tópico, já adaptado ao que você errou.

Para parar o servidor, use **Ctrl+C** no terminal. **Encerrar sessão** (barra lateral) interrompe o
ciclo em andamento; recarregar a página (F5) também descarta a sessão em andamento, mas o que já
foi salvo no perfil permanece.

No Windows, feche o Streamlit antes de rodar `uv sync`: com ele aberto, o `uv` não consegue
substituir os arquivos do ambiente ("arquivo já está sendo usado por outro processo").

### Terminal

```bash
uv run estudai estudar --aluno ana --topico funcoes   # tópicos: funcoes, probabilidade, cinematica, dinamica
uv run estudai estudar --aluno ana --topico funcoes --ciclos 2
uv run estudai perfil ana                             # histórico de versões do perfil
```

Responda às perguntas do tutor digitando e às questões com a letra da alternativa. **Ctrl+C**
interrompe (no PowerShell, Ctrl+Z não funciona).

## Experimento (Seção 5.3)

```bash
uv run estudai experimento --saida results/exp1.jsonl   # 3 condições × 3 personas × 4 tópicos
uv run estudai relatorio results/exp1.jsonl --orcamento-usd 0.50
```

O experimento grava cada execução assim que termina; rodar de novo o mesmo comando retoma de onde parou.

## Testes

```bash
uv run pytest
```

Os testes não fazem chamadas de rede: usam um LLM roteirizado (`tests/fakes.py`), a base BM25
do LanceDB (sem embeddings) e o executor de sandbox real.

## Limitações conhecidas (além das da Seção 5.4)

- A sandbox usa audit hooks (PEP 578), que não são fronteira de segurança contra código
  adversarial; em produção, rode o executor em contêiner sem rede.
- A verificação simbólica só confirma itens cuja resposta é um número; os demais dependem
  apenas do consenso do painel (com limiar de unanimidade, configurável).
- A base de conhecimento funciona só com BM25 quando não recebe um embedder; nesse caso os
  modos semântico e híbrido caem para a busca textual.
- O banco guarda fórmulas em LaTeX (`\(...\)`). A interface web renderiza via Streamlit/KaTeX;
  no terminal (Rich) os delimitadores aparecem como texto.
