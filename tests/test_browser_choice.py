"""Choosing which browser the app drives.

Most of this is about refusing a choice at the moment it is made. A browser
setting that saves anything and fails later fails from inside a tool, halfway
through a task, as a Playwright traceback — and the person reading it is trying
to book a flight, not debug a browser installation.
"""

from __future__ import annotations

import json
import os
import stat

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openmirror.agent.browsers import (
    CATALOG,
    CHANNELS,
    ENGINES,
    BrowserChoice,
    BrowserSettings,
    BrowserUnavailable,
    from_id,
)
from openmirror.routers import browser as browser_router


@pytest.fixture
def binary(tmp_path):
    """Something that exists and is executable. Not a browser — the settings
    layer checks that a path could be launched, not that it is Chromium, which
    is a claim only a launch can make."""
    path = tmp_path / 'brave-browser'
    path.write_text('#!/bin/sh\nexit 0\n')
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


# -- what a choice means -----------------------------------------------------


def test_an_id_becomes_the_right_playwright_arguments(binary):
    assert from_id('chromium').launch_kwargs() == {}
    assert from_id('chrome').launch_kwargs() == {'channel': 'chrome'}
    assert from_id('firefox').engine == 'firefox'
    assert from_id('custom', str(binary)).launch_kwargs() == {'executable_path': str(binary)}


def test_a_binary_wins_over_a_channel():
    """Playwright refuses both together, and the explicit path is the more
    specific of the two."""
    both = BrowserChoice(engine='chromium', channel='chrome', executable='/usr/bin/brave')
    assert both.launch_kwargs() == {'executable_path': '/usr/bin/brave'}
    assert both.id == 'custom'


def test_the_catalog_only_offers_things_the_code_understands():
    for entry in CATALOG:
        assert entry['id'] == 'custom' or entry['id'] in ENGINES or entry['id'] in CHANNELS
        # Every entry says when you would want it. A list of browser names with
        # no reason attached is a list nobody can choose from.
        assert entry['note'] and entry['label']


# -- refusing what cannot launch ---------------------------------------------


def test_a_path_that_is_not_there_is_refused(tmp_path):
    with pytest.raises(BrowserUnavailable) as caught:
        BrowserChoice(executable=str(tmp_path / 'nope')).validate()
    assert 'not a file' in str(caught.value)


def test_a_path_that_is_not_executable_is_refused(tmp_path):
    """Otherwise this fails inside Playwright with an error about a missing
    shared library, which names nothing a person can act on."""
    path = tmp_path / 'text'
    path.write_text('not a program')
    path.chmod(0o644)
    with pytest.raises(BrowserUnavailable) as caught:
        BrowserChoice(executable=str(path)).validate()
    assert 'not executable' in str(caught.value)


def test_a_channel_on_firefox_is_refused():
    with pytest.raises(BrowserUnavailable):
        BrowserChoice(engine='firefox', channel='chrome').validate()


def test_an_engine_that_is_not_an_engine_is_refused():
    with pytest.raises(BrowserUnavailable):
        BrowserChoice(engine='netscape').validate()


def test_choosing_a_custom_build_without_a_path_is_refused():
    with pytest.raises(BrowserUnavailable) as caught:
        from_id('custom', '')
    assert 'path' in str(caught.value)


# -- the saved choice --------------------------------------------------------


def test_nothing_saved_means_the_environment_default(tmp_path):
    default = BrowserChoice(engine='firefox')
    store = BrowserSettings(tmp_path / 'browser.json', default)
    assert not store.saved
    assert store.load().engine == 'firefox'


def test_saving_and_clearing(tmp_path, binary):
    store = BrowserSettings(tmp_path / 'browser.json', BrowserChoice())
    store.save(from_id('custom', str(binary)))
    assert store.saved
    assert store.load().executable == str(binary)

    assert json.loads((tmp_path / 'browser.json').read_text())['executable'] == str(binary)

    store.clear()
    assert not store.saved
    assert store.load().id == 'chromium'


def test_a_saved_choice_that_has_stopped_working_falls_back(tmp_path, binary):
    """Chrome uninstalled, a path that moved. The default is used and the log
    says why — silently launching something else would be worse, and refusing
    to browse at all would be worse still."""
    store = BrowserSettings(tmp_path / 'browser.json', BrowserChoice())
    store.save(from_id('custom', str(binary)))
    binary.unlink()
    assert store.load().id == 'chromium'


def test_a_corrupt_file_is_not_fatal(tmp_path):
    path = tmp_path / 'browser.json'
    path.write_text('{ this is not json')
    assert BrowserSettings(path, BrowserChoice()).load().id == 'chromium'


def test_a_saved_file_cannot_smuggle_in_an_engine_that_does_not_exist(tmp_path):
    path = tmp_path / 'browser.json'
    path.write_text(json.dumps({'engine': 'netscape'}))
    assert BrowserSettings(path, BrowserChoice()).load().engine == 'chromium'


# -- the API -----------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    from openmirror.config import config

    monkeypatch.setattr(config, 'data_dir', tmp_path)
    monkeypatch.setattr(config, 'browser_engine', 'chromium')
    monkeypatch.setattr(config, 'browser_channel', '')
    monkeypatch.setattr(config, 'browser_executable', '')
    app = FastAPI()
    app.include_router(browser_router.router)
    with TestClient(app) as live:
        yield live


def test_reading_says_what_is_in_force_and_where_it_came_from(client):
    body = client.get('/api/browser').json()
    assert body['browser'] == 'chromium'
    assert body['source'] == 'environment'
    assert {c['id'] for c in body['choices']} == {e['id'] for e in CATALOG}
    # Installed state is reported rather than the entry being hidden: somebody
    # whose Chrome is missing needs to see that it is missing.
    assert all('installed' in c for c in body['choices'])


def test_saving_a_choice_makes_it_the_one_in_force(client, binary):
    reply = client.put('/api/browser', json={'browser': 'custom', 'executable': str(binary)})
    assert reply.status_code == 200
    body = reply.json()
    assert body['browser'] == 'custom'
    assert body['source'] == 'saved'
    # Said out loud: a session already running keeps the browser it started.
    assert 'already running' in body['note']

    assert client.get('/api/browser').json()['executable'] == str(binary)


def test_a_refusal_explains_itself_and_changes_nothing(client):
    reply = client.put('/api/browser', json={'browser': 'custom', 'executable': '/nope/nothing'})
    assert reply.status_code == 400
    assert 'not a file' in reply.json()['detail']
    assert client.get('/api/browser').json()['source'] == 'environment'


def test_an_unknown_browser_is_refused(client):
    assert client.put('/api/browser', json={'browser': 'netscape'}).status_code == 400


def test_clearing_goes_back_to_the_environment(client, binary):
    client.put('/api/browser', json={'browser': 'custom', 'executable': str(binary)})
    body = client.delete('/api/browser').json()
    assert body['source'] == 'environment'
    assert body['browser'] == 'chromium'


def test_the_environment_provides_the_default_until_something_is_saved(client, monkeypatch):
    from openmirror.config import config

    monkeypatch.setattr(config, 'browser_engine', 'webkit')
    body = client.get('/api/browser').json()
    assert body['browser'] == 'webkit'
    assert body['default'] == 'webkit'

    client.put('/api/browser', json={'browser': 'chromium'})
    after = client.get('/api/browser').json()
    # Saved wins, and the default it is overriding is still reported — a person
    # who set the variable needs to know which of the two is in force.
    assert (after['browser'], after['source'], after['default']) == ('chromium', 'saved', 'webkit')


def test_the_search_browser_and_the_agent_browser_read_the_same_choice(tmp_path, monkeypatch, binary):
    """One setting, not two. "Which browser does this use" having two answers is
    how one of them quietly stops being maintained."""
    from openmirror.agent import browsers
    from openmirror.config import config

    monkeypatch.setattr(config, 'data_dir', tmp_path)
    monkeypatch.setattr(config, 'browser_engine', 'chromium')
    monkeypatch.setattr(config, 'browser_channel', '')
    monkeypatch.setattr(config, 'browser_executable', '')
    browsers.settings().save(from_id('custom', str(binary)))

    assert browsers.current().executable == str(binary)
    assert browsers.current().launch_kwargs() == {'executable_path': str(binary)}
    assert os.path.exists(browsers.current().executable)
