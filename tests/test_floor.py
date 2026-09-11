"""The floor, and the qualifier that has to travel with it.

openmirror's features are built against a minimum — `qwen3.5:9b` or `gemma4:12b` for
chat, `Qwen3-Embedding-4B` for memory. That much is an ordinary constant and
needs little guarding.

The part that needs guarding is the *qualifier*. The evidence for the chat
floor is a single measurement — gemma4:12b completing a five-step browser task
in twenty-six seconds, first try — and that measurement is only true with the
toolset narrowed to about ten tools. Given the full set of roughly thirty, the
same model on the same task failed outright. Quoted without its qualifier the
number becomes a promise the harness cannot keep, and it is exactly the kind of
detail that gets trimmed when a paragraph is tightened later.

So: wherever the measurement appears, a word about narrowing has to appear in
the same paragraph. That is a property of the prose, so it is checked against
the prose.

Two ways a check like this goes vacuous, and fixing one does not fix the other:
the scan can be anchored where the documents are not (asserted — the root is
found or it raises), and the pattern can stop matching a rewritten paragraph
(asserted — a positive control fixture carrying both a bare claim and a
properly qualified one, asserting exactly which comes back). An empty result is
what "clean", "wrong directory" and "pattern died" all look like.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from openmirror.agent.runtime import TOOLSETS
from openmirror.providers import catalog


def repo_root() -> Path:
    """The repository, found rather than assumed."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / 'README.md').exists() and (parent / 'docs' / 'PROVIDERS.md').exists():
            return parent
    raise AssertionError(f'cannot find the repository to scan, starting from {here}')


def read_real(rel: str) -> str:
    text = (repo_root() / rel).read_text(encoding='utf-8')
    assert len(text) > 500, f'{rel} is too short to have been read properly'
    return text


DOCS = ('README.md', 'docs/PROVIDERS.md', 'docs/INSTALL.md')


# -- the constants ----------------------------------------------------------


def test_the_chat_floor_is_stated_and_plausible():
    assert catalog.FLOOR_CHAT, 'FLOOR_CHAT is empty'
    for tag in catalog.FLOOR_CHAT:
        assert re.fullmatch(r'[A-Za-z0-9._-]+(:[A-Za-z0-9._-]+)?', tag), tag


def test_exactly_one_embedding_model_is_the_floor():
    floors = [m.id for m in catalog.EMBEDDING_MODELS if m.floor]
    assert floors == ['qwen3-embedding-4b'], floors


def test_the_floor_embedder_still_carries_what_makes_it_usable():
    # A floor that loses its dimensions or its query prefix is a floor that
    # silently retrieves worse — the exact failure it was chosen to avoid.
    m = next(m for m in catalog.EMBEDDING_MODELS if m.floor)
    assert m.dimensions == 2560, m.dimensions
    assert m.instruct_queries, 'the Qwen3 family wants the instruction on queries'
    assert m.served_by.get('ollama') == 'qwen3-embedding:4b'


def test_the_tool_budget_is_below_the_full_set_or_it_says_nothing():
    # The budget exists to be smaller than everything. If the full set ever
    # shrinks below it the advice has quietly become a no-op.
    full = len({name for names in TOOLSETS.values() for name in names})
    assert catalog.FLOOR_TOOL_BUDGET < full, (catalog.FLOOR_TOOL_BUDGET, full)
    # And it must be reachable: a real, useful combination has to fit inside it.
    narrowed = len(set(TOOLSETS['browser']) | set(TOOLSETS['web']) | set(TOOLSETS['ask']))
    assert narrowed <= catalog.FLOOR_TOOL_BUDGET, (narrowed, catalog.FLOOR_TOOL_BUDGET)


# -- the qualifier ----------------------------------------------------------

#: The measurement that must never travel alone.
CLAIM = re.compile(r'twenty-six seconds|26\s*(?:s\b|seconds)', re.I)

#: Any of these in the same paragraph means the claim is properly qualified.
QUALIFIER = re.compile(
    r'narrow\w*|only the browser tools|tool list|toolset|tools list|about ten tools|ten tools',
    re.I,
)


def unqualified_claims(text: str) -> list[str]:
    """Paragraphs that quote the measurement without saying it was narrowed."""
    out = []
    for para in re.split(r'\n\s*\n', text):
        if CLAIM.search(para) and not QUALIFIER.search(para):
            out.append(' '.join(para.split())[:120])
    return out


def test_the_scanner_can_tell_a_qualified_claim_from_a_bare_one():
    # Positive control, kept beside the real scan rather than in a sibling
    # test: both halves in one fixture, asserting exactly which comes back.
    # The qualified paragraph is the important one — a scanner reading too
    # loosely would flag it, and a scanner reading nothing reports both clean.
    fixture = (
        'gemma4:12b did the task in twenty-six seconds, correct, first try.\n'
        '\n'
        'The same model, narrowed to the browser tools, did it in 26 seconds.\n'
        '\n'
        'Some unrelated paragraph about approval modes.\n'
    )
    caught = unqualified_claims(fixture)
    assert len(caught) == 1, caught
    assert 'first try' in caught[0], caught


@pytest.mark.parametrize('doc', DOCS)
def test_the_measurement_never_appears_without_its_qualifier(doc):
    text = read_real(doc)
    found = unqualified_claims(text)
    assert not found, (
        f'{doc} quotes the twenty-six-second result without saying the toolset '
        f'was narrowed:\n' + '\n'.join(found)
    )


# -- the documents say it at all --------------------------------------------


def test_every_document_that_should_state_the_floor_does():
    for doc in DOCS:
        text = read_real(doc)
        assert re.search(r'\bfloor\b', text, re.I), f'{doc} never mentions the floor'
        for tag in catalog.FLOOR_CHAT:
            assert tag in text, f'{doc} does not name {tag}'


def test_the_floor_is_documented_as_a_warning_rather_than_a_wall():
    # The floor is the easy half to over-apply. Without this said plainly,
    # "tested against" turns into "requires" the first time someone summarises
    # it, and an operator with a small box is told no by a project that never
    # decided to say no.
    for doc in ('README.md', 'docs/PROVIDERS.md'):
        text = read_real(doc)
        assert re.search(r'never a wall|warning, never|yours to (?:decide|try)', text, re.I), (
            f'{doc} states a floor without saying a smaller model is still allowed'
        )


def test_the_ceiling_is_documented_so_the_floor_is_not_read_as_a_target():
    for doc in ('README.md', 'docs/PROVIDERS.md'):
        text = read_real(doc)
        assert re.search(r'frontier', text, re.I), f'{doc} does not mention frontier models'
        assert re.search(r'thinking|reasoning', text, re.I), f'{doc} does not mention thinking'


def test_the_embedding_migration_warning_is_where_someone_will_hit_it():
    # Moving UP to the floor is the likeliest moment to lose a memory store,
    # because a floor is aimed at installs sitting on an older default.
    for doc in ('docs/PROVIDERS.md', 'docs/INSTALL.md'):
        text = read_real(doc)
        assert re.search(r'not comparable|unsearchable|unfindable', text, re.I), (
            f'{doc} does not warn that changing the embedding model strands existing vectors'
        )
