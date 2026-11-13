"""Executor Python isolado em subprocesso (Seção 4).

- processo separado em modo isolado (`-I`), sem bytecode (`-B`), em diretório temporário;
- limite de tempo (o processo é encerrado ao estourar);
- audit hook (PEP 578) que bloqueia rede, criação de processos, ctypes e escrita em arquivos;
- semente fixa para `random`, tornando a execução determinística.

A leitura de arquivos não é bloqueada. Atenção: audit hooks não são uma fronteira de segurança
contra código adversarial. Para o escopo do projeto (código gerado pelo próprio sistema) são
suficientes; em produção, rode o executor em um contêiner sem rede.
"""

import os
import subprocess
import sys
import tempfile

from estudai.ports.tools import ExecutionResult

_PRELUDIO_INICIO = r"""
import sys as _sys, random as _random, os as _os
_random.seed(0)
import math, fractions
"""

_PRELUDIO_SYMPY = r"""
# bibliotecas confiáveis são importadas ANTES do hook (o sympy usa ctypes ao carregar no Windows)
import sympy
"""

_PRELUDIO_GUARDA = r"""
_BLOQUEADOS = ("socket.", "subprocess.", "os.system", "os.exec", "os.spawn", "os.posix_spawn",
               "os.fork", "os.remove", "os.unlink", "os.rmdir", "os.rename", "os.mkdir",
               "os.link", "os.symlink", "os.chmod", "os.chown", "os.truncate", "shutil.",
               "ctypes.", "urllib.", "http.", "ftplib.", "smtplib.", "webbrowser.", "winreg.")
_FLAGS_ESCRITA = _os.O_WRONLY | _os.O_RDWR | _os.O_CREAT | _os.O_APPEND | _os.O_TRUNC
def _guarda(evento, args):
    if evento.startswith(_BLOQUEADOS):
        raise PermissionError(f"operação bloqueada na sandbox: {evento}")
    if evento == "open":
        modo = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        if isinstance(modo, str) and any(m in modo for m in "wax+"):
            raise PermissionError("escrita em arquivo bloqueada na sandbox")
        if isinstance(flags, int) and flags & _FLAGS_ESCRITA:
            raise PermissionError("escrita em arquivo bloqueada na sandbox")
_sys.addaudithook(_guarda)
del _guarda
"""

_MAX_SAIDA = 20_000


class SubprocessPythonExecutor:
    def __init__(
        self,
        *,
        timeout_s: float = 10.0,
        python: str = sys.executable,
        preludio_confiavel: str = "",
        env_extra: dict[str, str] | None = None,
        max_saida: int = _MAX_SAIDA,
        importar_sympy: bool = True,
    ) -> None:
        """Configura o sandbox do executor de python"""
        self._timeout_s = timeout_s
        self._python = python
        self._preludio = (
            _PRELUDIO_INICIO
            + (_PRELUDIO_SYMPY if importar_sympy else "")
            + preludio_confiavel
            + "\n"
            + _PRELUDIO_GUARDA
        )
        self._env_extra = env_extra or {}
        self._max_saida = max_saida

    def run(self, code: str, timeout_s: float | None = None) -> ExecutionResult:
        env = {k: v for k, v in os.environ.items() if k.upper() in {"SYSTEMROOT", "PATH", "TEMP", "TMP"}}
        env.update(self._env_extra)
        with tempfile.TemporaryDirectory(prefix="estudai-sandbox-") as cwd:
            try:
                proc = subprocess.run(
                    [self._python, "-I", "-B", "-"],
                    input=self._preludio + "\n" + code,
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
            stdout=proc.stdout[-self._max_saida :],
            stderr=proc.stderr[-self._max_saida :],
        )
