"""Imita `claude -p --output-format json` nos testes (sem rede, sem custo).

Controlado por variáveis de ambiente: FAKE_CLAUDE_MODE (ok | lixo | erro_definitivo | erro_transiente),
FAKE_CLAUDE_STRUCTURED (JSON devolvido em `structured_output`) e FAKE_CLAUDE_LOG (arquivo onde cada
chamada registra argumentos, prompt e system prompt recebidos).
"""

import json
import os
import sys

args = sys.argv[1:]
prompt = sys.stdin.read()


def valor(flag: str) -> str | None:
    return args[args.index(flag) + 1] if flag in args else None


system = None
if arquivo := valor("--system-prompt-file"):
    with open(arquivo, encoding="utf-8") as f:
        system = f.read()

if log := os.environ.get("FAKE_CLAUDE_LOG"):
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"args": args, "prompt": prompt, "system": system, "cwd": os.getcwd()}) + "\n")

modo = os.environ.get("FAKE_CLAUDE_MODE", "ok")
if modo == "lixo":
    print("isto não é JSON")
    sys.exit(1)
if modo == "erro_definitivo":
    saida = {"is_error": True, "api_error_status": 400, "result": "API Error: 400 modelo inválido"}
elif modo == "erro_transiente":
    saida = {"is_error": True, "api_error_status": 529, "result": "Overloaded"}
else:
    estruturado = json.loads(os.environ.get("FAKE_CLAUDE_STRUCTURED", '{"resultado": 42}'))
    saida = {
        "is_error": False,
        "result": "resposta livre",
        "structured_output": estruturado if valor("--json-schema") else None,
        "total_cost_usd": 0.0123,
        "modelUsage": {
            valor("--model"): {"inputTokens": 100, "cacheReadInputTokens": 50, "outputTokens": 20},
        },
    }
print(json.dumps(saida))
