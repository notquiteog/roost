"""Where the page and the server have to agree.

The browser client is data-driven in a few places — a table of tool verbs, a
list of approval modes in a `<select>` — and each of those is a copy of
something the Python side owns. A copy that drifts does not crash: the UI
just starts saying the wrong thing, or offering a choice the server refuses.
Both are the sort of quiet wrong that only a test notices.

These read the page and the client as text rather than running them, so every
check asserts that it found something to check. A rewrite that defeats the
parser fails loudly here instead of quietly passing.
"""

import re
from pathlib import Path

from roost.agent.approval import Mode

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / 'roost' / 'static'
TOOLS = ROOT / 'roost' / 'agent' / 'tools'


def _tool_names() -> set[str]:
    declared: set[str] = set()
    for path in sorted(TOOLS.glob('*.py')):
        declared |= set(re.findall(r"(?m)^    name = '(\w+)'", path.read_text()))
    assert len(declared) > 15, 'the tool scan found almost nothing — has the layout changed?'
    return declared


def test_every_tool_has_a_verb():
    """The transcript says "Ran npm test", not "shell".

    A tool with no verb falls back to its own name, which is honest but reads
    as machinery in the middle of prose. Nobody notices the day a tool is
    added; this does.
    """
    app = (STATIC / 'app.js').read_text()
    at = app.find('const VERBS = {')
    assert at != -1, 'could not find VERBS in app.js'
    verbs = dict(re.findall(r"(?m)^  (\w+): '([^']+)',", app[at : app.index('};', at)]))
    assert verbs, 'no verbs parsed'

    declared = _tool_names()
    missing = declared - set(verbs)
    assert not missing, f'tools the transcript has no plain word for: {sorted(missing)}'

    unreal = set(verbs) - declared
    assert not unreal, f'verbs for tools that no longer exist: {sorted(unreal)}'


def test_the_approval_modes_the_page_offers_are_real():
    """The approval control now *sets* the mode rather than describing it.

    A `<option>` whose value is not a `Mode` used to be a cosmetic mistake.
    Since the composer sends it, it is now an error message on a click — and
    the two selects in the page have to stay in step with the enum and with
    each other.
    """
    html = (STATIC / 'index.html').read_text()
    selects = re.findall(r'<select[^>]*>(.*?)</select>', html, re.S)
    offered = [
        set(re.findall(r'<option value="([\w_]+)"', block))
        for block in selects
        if re.search(r'<option value="(read_only|ask)"', block)
    ]
    assert len(offered) == 2, 'expected the composer chip and the new-session dialog to offer modes'

    real = {m.value for m in Mode}
    for block in offered:
        assert block == real, f'the page offers {sorted(block)}, the server knows {sorted(real)}'
