# EstudAI

Sistema MultiAgente de Tutoria Adaptativa para Exatas em Vestibulares Brasileiros
MO810A - IC/UNICAMP - 2026

Este repositório inclui o trabalho prático desenvolvido para a discplina MO810A, envolvendo um sistema de tutoria utilizando arquiteturas multiagentes. As principais capacidades do sistema são, a partir de material didática e questões base, sintetizar material de ensino com diagramas, imagens e exercícios existentes de vestibulares sob demanda, e, dado a tutoria realizada, gerar questões com base em questões anteriores de vestibulares, que são verificadas por LLM juízes e verificação via Python. Os resultados sobre cada ciclo de tutoria e prática são armazenados como memória episódica, o que personaliza ciclos futuros e permite adaptação ao nível do aluno.

## Instalação e execução

```bash
uv sync --extra ingest --extra web   # adicione --extra embeddings se usar sentence-transformers para geração dos embeddings local
cp .env.example .env                 # preencha chaves e modelos
uv run estudai build-index           # constrói base vetorial para busca
uv run estudai web
```

(OPCIONAL) Para economizar chamadas de LLM, a rotulação já foi realizada e está inclusa no repositório. Caso deseja re-aplicar o pré-processamento, execute:

```bash
uv run estudai ingest-wikibooks      # baixa o material didático
uv run estudai ingest-questions      # baixa, filtra e rotula as questões (utiliza chamadas LLM)
uv run estudai build-index
```

### Modelos

O `.env.example` já vem com a configuração utilizada para desenvolvimento do projeto: **DeepSeek Flash** para os níveis econômico e de fronteira, e **Gemini 3 Flash** para os juízes.

| `PROVIDER` | Quando usar | Chave |
|---|---|---|
| `openai_compat` | DeepSeek ou qualquer endpoint compatível com a OpenAI (preencha `BASE_URL`) | `ESTUDAI_OPENAI_API_KEY` |
| `gemini` | API do Google (juízes e embeddings) | `ESTUDAI_GEMINI_API_KEY` |
| `openrouter` | Plataforma OpenRouter | `ESTUDAI_OPENROUTER_API_KEY` |

Os embeddings da busca usam o Gemini por padrão. Porém também é possível utilizar modelos de embedding locais, utilizando `ESTUDAI_EMBEDDING_PROVIDER=sentencetransformers` e instalando com `--extra embeddings` (utiliza o torch).

## Uso

Recomendamos a interface web para interagir com o sistema. Um ciclo de tutoria é composto por:

- Diagnóstico: algumas questões de vestibular para estimar seu nível inicial
- Tutoria: o tutor explica os pontos que você mais precisa e faz perguntas para testar seu aprendizado
- Resumo: um material de revisão do que foi ensinado
- Prática: exercícios personalizados, corrigidos com feedback dado sua resposta
- Atualização do perfil: Preenchimento de memória episódica para o próximo ciclo, que adapta ao que você errou

O repositório foi desenvolvido e testado utilizando o modelo `DeepSeek-V4.1-Flash` para o modelo de fronteira e econômico, providenciado pela API oficial do DeepSeek.

### Features

- **Tutoria Adaptativa**: Seus erros e dúvidas são acompanhados durante a tutoria e questões, e adaptados conformes os ciclos de ensino. Há também a possibilidade de justificar as questões elaboradas para diagnósticos parciais. É possível manualmente inspecionar o perfil persistente de cada aluno em `data/profiles`.
- **Imagens/Diagramas**: A tutoria e geração de imagens são acompanhadas com a possibilidade para imagens (via matplotlib) e diagramas (via Mermaid), sujeito à capacidade da LLM selecionada para tal. Em caso de erro, a legenda gerada pela LLM é disponibilizada.
- **Material Didático/Questões**: O sistema parte a partir de material didático existente (via Wikibooks) e questões de vestibulares reais, permitindo que o aluno peça diretamente examplos de ambas as bases de conhecimento. Os geradores de exercícios possuem acesso ao mesmo material, o que permite questões mais similares aos vestibulares reais.
- **Acompanhamento do sistema**: A interface web permite analisar as ações de cada agente em tempo real.

## Limitações

- A sandbox usa _audit hooks_, ainda possuindo possíveis vulnerabilidades.
- A verificação simbólica de exercícios confirma apenas itens cuja resposta é um número. Os demais dependem apenas do consenso do painel. Ambas verificações estão sujeita a erros
- Dado o escopo prático do trabalho, ainda há a necessidade de formalização de rúbricas de ensino e estimação do nível de ensino do aluno.
- A capacidade de geração de imagens/diagramas e a formatação LaTeX depende da capacidade da LLM utilizada. Observamos que mesmo com LLMs fortes, ainda pode haver erros na formatação.

## Arquitetura

| Camada | Pasta | Responsabilidade |
|---|---|---|
| Domínio | `domain/` | Entidades e regras: currículo, perfil versionado, itens, planejamento determinístico, protocolo de mensagens |
| Portas | `ports/` | Contratos: `Embedder`, `KnowledgeBase`, etc. |
| Agentes | `agents/` | Implementação dos agentes, saída estruturada com reparo e ferramentas nativas (LangChain) |
| Aplicação | `application/` | Ciclo de estudo (fases), validação por consenso, estratégias de prática, telemetria, log de mensagens, bastidores (progresso) |
| Orquestração | `orchestration/` | Grafo LangGraph que ordena as fases do ciclo |
| Infraestrutura | `infrastructure/` | Gemini, OpenAI-compatível, retry/telemetria, BM25 + vetorial, JSON, sandbox (código e figuras matplotlib), Mermaid, ingestão |
| Avaliação | `evaluation/` | Personas, aluno simulado, juízes, experimentos C1/C2/C3 da metolodogia |
| Interfaces | `interfaces/` | Interfaces CLI (`typer`) e web (Streamlit) |