"""Frontmatter: the few lines of YAML at the top of a markdown file.

Skills and agent definitions are both markdown files that open with a block
like this, and whose body is the instructions:

    ---
    name: review
    description: Review the uncommitted changes for bugs.
    tools: read_file, grep, shell
    ---

The format is the one the rest of the ecosystem already writes, so a skill or
an agent made for another client works here unchanged — which is the whole
reason to parse it rather than invent something simpler.

Parsed by hand rather than with a YAML library. What these files contain in
practice is flat keys, plain or quoted strings, block scalars for a long
description, and lists written inline or one item to a line. That is a page
of parser, against a dependency whose full grammar — anchors, tags, `no`
meaning false — is mostly ways for a description to mean something other than
what it says. Anything this does not understand is skipped, not fatal: a
skill with one odd line should still load.
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)

_KEY = re.compile(r'^([A-Za-z_][\w-]*)\s*:\s*(.*)$')


def split(text: str) -> tuple[dict[str, Any], str]:
    """(fields, body). A file with no frontmatter is all body."""
    text = text.lstrip('﻿')
    lines = text.splitlines()
    if not lines or lines[0].strip() != '---':
        return {}, text

    for end in range(1, len(lines)):
        if lines[end].strip() in ('---', '...'):
            break
    else:
        # An opening fence with no closing one is a document that happens to
        # start with a horizontal rule, not a broken header.
        return {}, text

    fields = _parse(lines[1:end])
    body = '\n'.join(lines[end + 1 :]).strip('\n')
    return fields, body


def _parse(lines: list[str]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        if line[0].isspace():
            # Indented, but not under a key that wanted it: nothing to attach to.
            continue
        match = _KEY.match(line)
        if not match:
            log.debug('frontmatter: skipping %r', line)
            continue
        key, raw = match.group(1), match.group(2).strip()

        if raw in ('|', '>', '|-', '>-', '|+', '>+'):
            block, i = _indented(lines, i)
            if raw.startswith('|'):
                fields[key] = '\n'.join(block).strip('\n')
            else:
                # Folded: lines join with spaces, and a blank line is a break.
                paragraphs = '\n'.join(block).strip('\n').split('\n\n')
                fields[key] = '\n'.join(' '.join(p.split()) for p in paragraphs)
            continue

        if raw == '':
            block, j = _indented(lines, i)
            items = [b.strip()[1:].strip() for b in block if b.strip().startswith('-')]
            if items:
                fields[key] = [_scalar(item) for item in items]
                i = j
            else:
                fields[key] = ''
            continue

        fields[key] = _value(raw)
    return fields


def _indented(lines: list[str], i: int) -> tuple[list[str], int]:
    """The indented (or blank) lines from `i`, dedented, and where they stop."""
    block: list[str] = []
    while i < len(lines) and (not lines[i].strip() or lines[i][0].isspace()):
        block.append(lines[i])
        i += 1
    while block and not block[-1].strip():
        block.pop()
    indents = [len(b) - len(b.lstrip()) for b in block if b.strip()]
    cut = min(indents) if indents else 0
    return [b[cut:] for b in block], i


def _value(raw: str) -> Any:
    if raw.startswith('[') and raw.endswith(']'):
        inner = raw[1:-1].strip()
        return [_scalar(part) for part in _split_commas(inner)] if inner else []
    return _scalar(raw)


def _split_commas(text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    quote = ''
    for ch in text:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ''
        elif ch in '\'"':
            quote = ch
            buf.append(ch)
        elif ch == ',':
            parts.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if ''.join(buf).strip():
        parts.append(''.join(buf).strip())
    return parts


def _scalar(raw: str) -> Any:
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] == '"':
        return raw[1:-1].replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
    if len(raw) >= 2 and raw[0] == raw[-1] == "'":
        return raw[1:-1].replace("''", "'")
    # A trailing comment, but only after whitespace: `#` inside a value is text.
    raw = re.split(r'\s+#', raw, maxsplit=1)[0].strip()
    lowered = raw.lower()
    if lowered == 'true':
        return True
    if lowered == 'false':
        return False
    return raw


def as_list(value: Any) -> list[str]:
    """A list of names, however it was written.

    `tools: read_file, grep`, `tools: [read_file, grep]` and a dash list all
    mean the same thing, and all three turn up in files people have written.
    """
    if value is None or value == '':
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value)
    # Commas when there are any, and spaces only when there are none: a name
    # can carry a pattern with a space in it — `Bash(git diff:*)` — and
    # splitting that on the space leaves two halves that name nothing.
    parts = text.split(',') if ',' in text else text.split()
    return [part.strip() for part in parts if part.strip()]


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == '':
        return default
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')
