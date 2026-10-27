"""Executor Python isolado em subprocesso (Seção 4).

- processo separado em modo isolado (`-I`), sem bytecode (`-B`), em diretório temporário;
- limite de tempo (o processo é encerrado ao estourar);
- audit hook (PEP 578) que bloqueia rede, criação de processos, ctypes e escrita em arquivos;
- semente fixa para `random`, tornando a execução determinística.

Atenção: audit hooks não são uma fronteira de segurança contra código adversarial. Para o
escopo do projeto (código de verificação gerado pelo próprio sistema) são suficientes; em
produção, rode o executor em um contêiner sem rede.
"""

import os
import subprocess
import sys
import tempfile

from estudai.ports.tools import ExecutionResult

_PRELUDIO = r"""
import sys as _sys, random as _random
_random.seed(0)
# bibliotecas confiáveis são importadas ANTES do hook (o sympy usa ctypes ao carregar no Windows)
import math, fractions, sympy

_BLOQUEADOS = ("socket.", "subprocess.", "os.system", "os.exec", "os.spawn", "os.posix_spawn",
               "os.fork", "os.remove", "os.unlink", "os.rmdir", "os.rename", "shutil.",
               "ctypes.", "urllib.", "http.", "ftplib.", "smtplib.", "webbrowser.", "winreg.")
def _guarda(evento, args):
    if evento.startswith(_BLOQUEADOS):
        raise PermissionError(f"operação bloqueada na sandbox: {evento}")
    if evento == "open" and isinstance(args[1], str) and any(m in args[1] for m in "wax+"):
        raise PermissionError("escrita em arquivo bloqueada na sandbox")
_sys.addaudithook(_guarda)
del _guarda
"""

_MAX_SAIDA = 20_000


class SubprocessPythonExecutor:
    def __init__(self, *, timeout_s: float = 10.0, python: str = sys.executable) -> None:
        self._timeout_s = timeout_s
        self._python = python

    def run(self, code: str, timeout_s: float | None = None) -> ExecutionResult:
        env = {k: v for k, v in os.environ.items() if k.upper() in {"SYSTEMROOT", "PATH", "TEMP", "TMP"}}
        with tempfile.TemporaryDirectory(prefix="estudai-sandbox-") as cwd:
            try:
                proc = subprocess.run(
                    [self._python, "-I", "-B", "-"],
                    input=_PRELUDIO + "\n" + code,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_s or self._timeout_s,
                    cwd=cwd,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                return ExecutionResult(ok=False, timed_out=True, stderr="tempo limite excedido")
        return ExecutionResult(
            ok=proc.returncode == 0,
            stdout=proc.stdout[-_MAX_SAIDA:],
            stderr=proc.stderr[-_MAX_SAIDA:],
        )
