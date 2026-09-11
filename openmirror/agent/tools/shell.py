"""Running commands on the machine.

The risk classifier below is the part that matters. It exists so that the
approval prompt means something: if every command needs a human, the human
stops reading, and the one command that deletes their home directory gets the
same reflexive yes as the forty `git status` calls before it. So a read is a
read, and `rm -rf` is never anything but destructive.

It is a heuristic and it is deliberately biased. Anything it cannot parse
confidently is escalated, never waved through — a command it does not
understand is exactly the sort it should be asking about. It is not a
sandbox, and it is not a substitute for one: it decides what to *ask*, and
containment is the container's job.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import subprocess
import sys
from typing import Any

from openmirror.agent.tools.base import Assessment, Output, Tool, ToolContext, ToolError, truncate
from openmirror.protocol.agent import Risk

WINDOWS = sys.platform == 'win32'


def _spawn_kwargs() -> dict[str, Any]:
    """Start the command in its own group, whatever the platform calls that.

    The point is the same on both: a timeout or an interrupt has to reach the
    command's children, not just the shell that launched them. Killing only
    the shell leaves an orphaned dev server holding its port, which is the
    failure people actually hit.
    """
    if WINDOWS:
        # CREATE_NEW_PROCESS_GROUP is the closest Windows analogue, and the
        # flag `taskkill /T` needs to walk the tree later.
        return {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
    # POSIX: setsid, so the whole group can be signalled by negative pid.
    return {'start_new_session': True}


async def _kill_tree(proc: Any, hard: bool = True) -> None:
    """Kill a process and everything it started.

    Windows has no process groups in the POSIX sense and no SIGTERM, so the
    only reliable way to get the children is to shell out to taskkill. It is
    ugly and it is what works.
    """
    if proc.returncode is not None:
        return
    if WINDOWS:
        try:
            killer = await asyncio.create_subprocess_exec(
                'taskkill', '/F', '/T', '/PID', str(proc.pid),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.wait(), timeout=10)
        except (TimeoutError, OSError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        return

    try:
        os.killpg(os.getpgid(proc.pid), 9 if hard else 15)
    except (ProcessLookupError, PermissionError):
        pass

# Commands that only look. Anything not on this list is at least `execute`.
READ_ONLY = {
    'ls', 'cat', 'head', 'tail', 'wc', 'file', 'stat', 'pwd', 'whoami', 'date',
    'echo', 'printf', 'which', 'type', 'env', 'printenv', 'uname', 'hostname',
    'df', 'du', 'free', 'ps', 'top', 'uptime', 'id', 'groups', 'tree',
    'grep', 'egrep', 'fgrep', 'rg', 'ag', 'find', 'fd', 'locate',
    'diff', 'cmp', 'md5sum', 'sha256sum', 'basename', 'dirname', 'realpath',
    'sort', 'uniq', 'cut', 'awk', 'sed', 'jq', 'column', 'nl', 'tr',
    # Windows and PowerShell equivalents. Without these every `dir` on Windows
    # asks for approval, and a prompt that fires constantly is a prompt nobody
    # reads — which is the whole failure this classifier exists to avoid.
    'dir', 'where', 'findstr', 'more', 'tasklist', 'systeminfo', 'ver',
    'get-content', 'get-childitem', 'get-location', 'get-process', 'get-item',
    'select-string', 'measure-object', 'select-object', 'sort-object',
    'format-list', 'format-table', 'out-string', 'write-output', 'write-host',
    'test-path', 'get-date', 'get-help', 'gci', 'gc',
}

# Read-only subcommands of tools that are otherwise not read-only.
READ_ONLY_SUB = {
    'git': {'status', 'log', 'diff', 'show', 'branch', 'remote', 'blame', 'describe', 'ls-files', 'rev-parse'},
    'docker': {'ps', 'images', 'logs', 'inspect', 'version'},
    'podman': {'ps', 'images', 'logs', 'inspect', 'version'},
    'npm': {'ls', 'list', 'view', 'outdated'},
    'kubectl': {'get', 'describe', 'logs', 'version'},
    'systemctl': {'status', 'show', 'list-units', 'is-active', 'is-enabled'},
}

# Matched against the whole command line, after the token scan. These are the
# ones that are worth being blunt about.
#
# Both shells are here regardless of platform, and deliberately. `bash` on
# Windows is a real thing (Git Bash, WSL) and PowerShell runs on Linux and
# macOS; matching only the host's native shell would leave a hole exactly
# where somebody is being clever.
DESTRUCTIVE_PATTERNS = [
    (re.compile(r'\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+'), 'recursive or forced delete'),
    (re.compile(r'\bdd\s+.*\bof=/dev/'), 'writing directly to a device'),
    (re.compile(r'\bmkfs(\.\w+)?\b'), 'formatting a filesystem'),
    (re.compile(r'>\s*/dev/(sd|nvme|hd|vd)'), 'writing to a block device'),
    (re.compile(r'\bgit\s+push\b.*(--force\b|--force-with-lease\b|\s-f\b)'), 'force push'),
    (re.compile(r'\bgit\s+(reset\s+--hard|clean\s+-[a-zA-Z]*f)'), 'discarding uncommitted work'),
    (re.compile(r'\b(shutdown|reboot|halt|poweroff)\b'), 'powering the machine down'),
    (re.compile(r'\b(DROP|TRUNCATE)\s+(TABLE|DATABASE|SCHEMA)\b', re.I), 'dropping a database object'),
    (re.compile(r'\bchmod\s+(-R\s+)?777\b'), 'making files world-writable'),
    (re.compile(r'\b(kill|pkill|killall)\s+(-9|-KILL)\b'), 'force-killing processes'),
    (re.compile(r':\(\)\s*\{.*\|.*&.*\}\s*;?\s*:'), 'fork bomb'),
    (re.compile(r'\bcrontab\s+-r\b'), 'deleting the crontab'),

    # --- Windows: cmd.exe ---
    (re.compile(r'\bdel\s+(?:/[a-z]\s+)*/[sq]\b', re.I), 'recursive delete'),
    (re.compile(r'\b(?:rd|rmdir)\s+(?:/[a-z]\s+)*/s\b', re.I), 'recursive directory delete'),
    (re.compile(r'\bformat\s+[a-z]:', re.I), 'formatting a drive'),
    (re.compile(r'\bdiskpart\b', re.I), 'partitioning a disk'),
    (re.compile(r'\bvssadmin\s+delete\s+shadows', re.I), 'deleting shadow copies'),
    (re.compile(r'\bcipher\s+/w', re.I), 'wiping free space'),
    (re.compile(r'\breg\s+delete\b', re.I), 'deleting a registry key'),
    (re.compile(r'\bbcdedit\b', re.I), 'changing the boot configuration'),
    (re.compile(r'\bshutdown\s+/[rs]\b', re.I), 'shutting the machine down'),
    (re.compile(r'\btaskkill\s+.*\/f\b', re.I), 'force-killing processes'),

    # --- Windows: PowerShell ---
    (re.compile(r'\bRemove-Item\b.*-(?:Recurse|Force)\b', re.I), 'recursive or forced delete'),
    (re.compile(r'\bRemove-Item\b.*\*', re.I), 'wildcard delete'),
    (re.compile(r'\b(?:Format-Volume|Clear-Disk|Initialize-Disk)\b', re.I), 'formatting a disk'),
    (re.compile(r'\bStop-Computer\b|\bRestart-Computer\b', re.I), 'powering the machine down'),
    (re.compile(r'\bRemove-ItemProperty\b|\bRemove-LocalUser\b', re.I), 'removing a registry value or user'),
    (re.compile(r'\bSet-ExecutionPolicy\s+(?:Unrestricted|Bypass)\b', re.I), 'disabling script signing checks'),
    (re.compile(r'\bInvoke-Expression\b|\biex\b', re.I), 'executing a constructed string'),
]

NETWORK_COMMANDS = {
    'curl', 'wget', 'ssh', 'scp', 'rsync', 'sftp', 'ftp', 'nc', 'ncat', 'telnet',
    'pip', 'pip3', 'npm', 'npx', 'yarn', 'pnpm', 'apt', 'apt-get', 'dnf', 'yum',
    'brew', 'cargo', 'go', 'gem', 'composer', 'uv', 'poetry',
    'winget', 'choco', 'scoop', 'nuget',
    'invoke-webrequest', 'invoke-restmethod', 'iwr', 'curl.exe', 'start-bitstransfer',
}

# Shell metacharacters that make a token scan unreliable, because what runs is
# decided at runtime rather than visible in the text.
OPAQUE = re.compile(r'\$\(|`|\beval\b|\bexec\b|\|\s*(sh|bash|zsh)\b')


def _command_name(token: str) -> str:
    """The comparable name of a command, across shells.

    Two normalisations, both narrow on purpose:

    `.exe`, `.cmd` and `.bat` are stripped, so `curl.exe` is `curl`.

    A hyphenated token is lower-cased, because that is the shape of a
    PowerShell cmdlet and PowerShell is case-insensitive — `Get-ChildItem` and
    `get-childitem` are one command. POSIX command names are *not*
    case-insensitive, so everything else keeps its case: lower-casing `LS`
    into `ls` would grade an unknown binary as a known-safe read, which is the
    one direction this must never fail in.
    """
    name = os.path.basename(token)
    for suffix in ('.exe', '.cmd', '.bat', '.ps1'):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.lower() if '-' in name else name


def _segments(command: str) -> list[list[str]]:
    """Split a command line into the individual commands it will run.

    Best-effort by construction: `shlex` understands quoting but not shell
    grammar, so a line it cannot lex is reported as unparseable and the caller
    escalates rather than guessing.
    """
    try:
        tokens = shlex.split(command, comments=True)
    except ValueError as exc:
        raise ToolError(f'unbalanced quoting: {exc}') from exc

    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in ('&&', '||', ';', '|', '&'):
            if current:
                segments.append(current)
            current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    return segments


# Paths that are worth escalating for on sight, because a command touching
# them is leaving the working directory whatever else it is doing.
_PATH_LIKE = re.compile(r'(?:^|[\s=\'"(])((?:~|/|\.\./)[^\s\'"();|&]*)')


def paths_outside(command: str, ctx: object) -> list[str]:
    """Absolute or parent-relative paths in a command that leave the root.

    The shell cannot be confined — what a command touches is decided at
    runtime, and no amount of string inspection changes that. What it *can*
    do is stop grading `cat /etc/shadow` as a plain read, so that under a
    policy that auto-runs reads it still stops and asks.

    Real containment is the OS's job: a container, a namespace, seccomp.
    This is honesty about the boundary, not enforcement of it.
    """
    from openmirror.agent.tools.base import escapes_root

    found = []
    for match in _PATH_LIKE.finditer(command):
        candidate = match.group(1)
        if escapes_root(candidate, ctx):
            found.append(candidate)
    return found


def classify(command: str) -> tuple[Risk, str]:
    """The risk of this specific command line, and why."""
    for pattern, why in DESTRUCTIVE_PATTERNS:
        if pattern.search(command):
            return Risk.DESTRUCTIVE, why

    if OPAQUE.search(command):
        # Command substitution, eval, or a pipe into a shell: what actually
        # runs is not in the text, so no token scan can be trusted.
        return Risk.EXECUTE, 'builds the command at runtime'

    try:
        segments = _segments(command)
    except ToolError:
        return Risk.EXECUTE, 'could not be parsed'

    if not segments:
        return Risk.READ, 'empty'

    risk = Risk.READ
    why = 'reads only'

    for seg in segments:
        # Skip a leading VAR=value assignment so `FOO=1 ls` still reads as ls.
        idx = 0
        while idx < len(seg) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*=.*', seg[idx]):
            idx += 1
        if idx >= len(seg):
            continue

        argv = seg[idx:]
        name = _command_name(argv[0])
        if name in ('sudo', 'doas', 'env', 'nice', 'nohup', 'time', 'xargs') and len(argv) > 1:
            # sudo raises the floor: whatever follows now runs as root.
            escalated = name in ('sudo', 'doas')
            argv = argv[1:]
            name = _command_name(argv[0])
            if escalated and risk is Risk.READ:
                risk, why = Risk.EXECUTE, 'runs as root'

        if name in NETWORK_COMMANDS:
            if risk in (Risk.READ, Risk.EXECUTE):
                risk, why = Risk.NETWORK, f'{name} reaches the network'
            continue

        sub = READ_ONLY_SUB.get(name)
        if sub is not None:
            args = [a for a in argv[1:] if not a.startswith('-')]
            if args and args[0] in sub:
                continue
            if risk is Risk.READ:
                risk, why = Risk.EXECUTE, f'{name} {args[0] if args else ""}'.strip()
            continue

        if name in READ_ONLY:
            # A read-only command with its output redirected is a write.
            continue

        if risk is Risk.READ:
            risk, why = Risk.EXECUTE, f'runs {name}'

    # Redirection writes files whatever the command was.
    if re.search(r'(?<![0-9<>])>{1,2}(?!&)', command) and risk is Risk.READ:
        risk, why = Risk.WRITE, 'redirects output to a file'

    return risk, why


class ShellTool(Tool):
    name = 'shell'
    description = (
        'Run a shell command on the machine and return its output. '
        'Use this for builds, tests, git, package managers and anything else with a CLI. '
        'Prefer the dedicated file tools for reading and editing files: they are cheaper and '
        'their results are easier to act on.'
    )
    input_schema = {
        'type': 'object',
        'properties': {
            'command': {'type': 'string', 'description': 'The command line to run.'},
            'cwd': {'type': 'string', 'description': 'Working directory, relative to the session root.'},
            'timeout': {
                'type': 'integer',
                'description': 'Seconds before the command is killed. Default 120, maximum 1800.',
            },
        },
        'required': ['command'],
    }

    def __init__(self, default_timeout: int = 120, max_output: int = 30_000) -> None:
        self.default_timeout = default_timeout
        self.max_output = max_output

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        command = (args.get('command') or '').strip()
        if not command:
            return Assessment(risk=Risk.READ, summary='', invalid='command is required')

        risk, why = classify(command)

        # A command naming a path outside the root is never a plain read,
        # however innocent the verb. Without this, `cat /etc/shadow` grades as
        # `read` and runs silently under the default policy — the file tools
        # would have refused the same path outright.
        if risk in (Risk.READ, Risk.WRITE):
            outside = paths_outside(command, ctx)
            if outside:
                risk = Risk.EXECUTE
                why = f'reaches outside the working root ({outside[0]})'

        # One line, the command itself first, because that is what a person
        # reads. The reason follows for the cases where it is not obvious.
        shown = command if len(command) <= 120 else command[:117] + '...'
        return Assessment(risk=risk, summary=shown if risk is Risk.READ else f'{shown}   ({why})')

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        from openmirror.agent.tools.base import resolve_in_root

        command = args['command']
        cwd = resolve_in_root(args['cwd'], ctx) if args.get('cwd') else ctx.cwd
        if not cwd.is_dir():
            raise ToolError(f'{cwd}: not a directory')

        timeout = min(int(args.get('timeout') or self.default_timeout), 1800)

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
            env={**os.environ, **ctx.env},
            **_spawn_kwargs(),
        )

        chunks: list[str] = []
        size = 0


        async def pump(stream: asyncio.StreamReader, which: str) -> None:
            nonlocal size
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode('utf-8', 'replace')
                # Streamed to the watcher in full; kept for the model only up
                # to the cap, so a runaway loop cannot exhaust the context.
                await ctx.emit(text, which)
                if size < self.max_output * 2:
                    chunks.append(text)
                    size += len(text)

        pumps = asyncio.gather(pump(proc.stdout, 'stdout'), pump(proc.stderr, 'stderr'))
        timed_out = False
        try:
            await asyncio.wait_for(asyncio.gather(proc.wait(), pumps), timeout=timeout)
        except TimeoutError:
            timed_out = True
            pumps.cancel()
            # Politely first, so the command can clean up after itself; then
            # not politely, for anything that ignores it.
            await _kill_tree(proc, hard=False)
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except TimeoutError:
                await _kill_tree(proc, hard=True)
                await proc.wait()
        except asyncio.CancelledError:
            # The turn was interrupted. Nothing else will ever reap this
            # process, so it is killed here and waited for — without the
            # wait, asyncio closes the event loop while the transport is
            # still open and complains about it at garbage-collection time.
            pumps.cancel()
            await _kill_tree(proc, hard=True)
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except TimeoutError:
                pass
            raise

        code = proc.returncode
        body, truncated = truncate(''.join(chunks), self.max_output, keep='both')

        if timed_out:
            content = f'Command timed out after {timeout}s and was killed.\n\n{body}'
        elif code == 0:
            content = body or '(no output)'
        else:
            # The exit code stated plainly: models otherwise assume success
            # whenever a failing command happened to print nothing to stderr.
            content = f'Exit code {code}\n\n{body or "(no output)"}'

        return Output(
            content=content,
            display={'exit_code': code, 'timed_out': timed_out, 'cwd': str(cwd), 'command': command},
            truncated=truncated,
        )
