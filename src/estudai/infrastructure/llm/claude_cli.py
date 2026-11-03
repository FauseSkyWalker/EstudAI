"""Motor alternativo: o Claude Code CLI (`claude -p`) como modelo de chat do LangChain.

Cada chamada roda `claude -p --output-format json` num diretório temporário, sem ferramentas,
sem MCP, sem sessão persistida e com o system prompt do agente (não o do Claude Code). A saída
estruturada usa o `--json-schema` nativo do CLI. Usa a autenticação já configurada no CLI
(assinatura ou chave), sem chave no `.env`.

Limitações em relação a uma API: não há `temperature` (o CLI não expõe; o modelo usa o padrão,
o que mantém os resolutores do painel independentes, mas os juízes deixam de ser determinísticos),
e cada chamada abre um processo (alguns segundos de latência extra).
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field, ValidationError

# Erros da API que valem nova tentativa (rate limit, sobrecarga, falhas do servidor)
_TRANSIENTES = {408, 429, 500, 502, 503, 504, 529}


class ClaudeCLIError(RuntimeError):
    """O CLI falhou ou devolveu `is_error` depois das novas tentativas."""


def localizar_claude_cli() -> list[str] | None:
    """Comando para executar o CLI. No Windows, o `claude` do npm é um `.cmd`, e argumentos
    passados a um `.cmd` atravessam o `cmd.exe`, que corrompe o JSON do `--json-schema`;
    por isso chamamos o executável nativo que o `.cmd` envolve."""
    exe = shutil.which("claude")
    if exe is None:
        return None
    if os.name == "nt" and exe.lower().endswith((".cmd", ".bat")):
        nativo = Path(exe).parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
        return [str(nativo)] if nativo.exists() else None
    return [exe]


def _texto(content: str | list) -> str:
    if isinstance(content, str):
        return content
    return "".join(p if isinstance(p, str) else p.get("text", "") for p in content)


def _serializar(conversa: list[BaseMessage]) -> str:
    """O CLI recebe um único prompt; históricos (reparo de saída, laço de ferramentas) viram uma transcrição."""
    if len(conversa) == 1:
        return _texto(conversa[0].content)
    rotulos = {"human": "USUÁRIO", "ai": "ASSISTENTE"}
    partes = ["Histórico da conversa. Responda apenas à última mensagem do USUÁRIO, no formato pedido.\n"]
    partes += [f"[{rotulos.get(m.type, m.type.upper())}]\n{_texto(m.content)}\n" for m in conversa]
    return "\n".join(partes)


class ClaudeCLIChatModel(BaseChatModel):
    model: str = "claude-haiku-5-5"
    command: list[str] = Field(default_factory=list, description="Vazio = localizar o CLI automaticamente.")
    timeout_s: float = 300.0
    max_retries: int = 2
    effort: str | None = None  # low | medium | high | xhigh | max

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    def with_structured_output(self, schema, *, include_raw: bool = False, **kwargs: Any):
        """Usa o `--json-schema` do CLI e valida o resultado com o próprio schema pydantic."""
        json_schema = schema.model_json_schema()

        def run(entrada, config=None):
            mensagem = self.invoke(entrada, config=config, json_schema=json_schema)
            try:
                parsed, erro = schema.model_validate_json(_texto(mensagem.content)), None
            except (ValidationError, ValueError) as exc:
                parsed, erro = None, exc
            if include_raw:
                return {"raw": mensagem, "parsed": parsed, "parsing_error": erro}
            if erro is not None:
                raise erro
            return parsed

        return RunnableLambda(run)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        json_schema: dict | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = "\n\n".join(_texto(m.content) for m in messages if isinstance(m, SystemMessage))
        conversa = [m for m in messages if not isinstance(m, SystemMessage)]
        dados = self._executar(system, _serializar(conversa), json_schema)

        estruturado = dados.get("structured_output")
        conteudo = (
            json.dumps(estruturado, ensure_ascii=False)
            if json_schema is not None and estruturado is not None
            else dados.get("result") or ""
        )
        # `modelUsage` soma todos os turnos internos (a saída estruturada usa um turno extra)
        uso_por_modelo = dados.get("modelUsage") or {}
        entrada = sum(
            u.get("inputTokens", 0) + u.get("cacheReadInputTokens", 0) + u.get("cacheCreationInputTokens", 0)
            for u in uso_por_modelo.values()
        )
        saida = sum(u.get("outputTokens", 0) for u in uso_por_modelo.values())
        modelo = next(iter(uso_por_modelo), self.model)
        mensagem = AIMessage(
            content=conteudo,
            usage_metadata={"input_tokens": entrada, "output_tokens": saida, "total_tokens": entrada + saida},
            response_metadata={"model_name": modelo, "cost_usd": dados.get("total_cost_usd")},
        )
        return ChatResult(generations=[ChatGeneration(message=mensagem)], llm_output={"model_name": modelo})

    def _executar(self, system: str, prompt: str, json_schema: dict | None) -> dict:
        comando = self.command or localizar_claude_cli()
        if not comando:
            raise ClaudeCLIError("Claude CLI não encontrado no PATH; instale o Claude Code ou troque o provider.")
        args = [
            *comando, "-p", "--output-format", "json", "--model", self.model,
            "--tools", "", "--no-session-persistence", "--strict-mcp-config", "--disable-slash-commands",
        ]  # fmt: skip
        if self.effort:
            args += ["--effort", self.effort]
        if json_schema is not None:
            args += ["--json-schema", json.dumps(json_schema, ensure_ascii=False)]

        erro = "sem tentativas"
        # diretório temporário: o CLI não descobre o CLAUDE.md nem as configurações do projeto
        with tempfile.TemporaryDirectory(prefix="estudai-claude-") as cwd:
            if system:  # por arquivo: evita o limite de tamanho da linha de comando do Windows
                arquivo = Path(cwd) / "system.md"
                arquivo.write_text(system, encoding="utf-8")
                args += ["--system-prompt-file", str(arquivo)]
            for tentativa in range(self.max_retries + 1):
                if tentativa:
                    time.sleep(min(2**tentativa, 30))
                try:
                    proc = subprocess.run(
                        args, input=prompt, capture_output=True, text=True, encoding="utf-8",
                        errors="replace", cwd=cwd, timeout=self.timeout_s,
                    )  # fmt: skip
                except subprocess.TimeoutExpired:
                    erro = f"tempo limite de {self.timeout_s:.0f}s excedido"
                    continue
                try:
                    dados = json.loads(proc.stdout)
                except json.JSONDecodeError:
                    erro = f"saída não-JSON (código {proc.returncode}): {(proc.stderr or proc.stdout)[-400:]}"
                    continue
                if not dados.get("is_error"):
                    return dados
                status = dados.get("api_error_status")
                erro = f"erro da API ({status}): {str(dados.get('result'))[:400]}"
                if status not in _TRANSIENTES:
                    break  # erro definitivo (modelo inválido, requisição malformada...)
        raise ClaudeCLIError(f"Claude CLI ({self.model}): {erro}")
