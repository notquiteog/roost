"""The page has to be allowed to notice that it changed.

Static assets went out with an ETag and a Last-Modified and no `Cache-Control`
at all, which does not mean "do not cache" to a browser — it means "invent a
freshness window", and Chromium invents a tenth of the file's age. An interface
updated on the server then goes on running the copy already in the browser
until somebody hard-refreshes, and the change gets blamed for doing nothing.
That happened here: a fix was live and the page in front of it still held the
old handler.

`no-cache` keeps the copy and revalidates it, so the cost of being right is a
304 with no body.
"""

from __future__ import annotations

from starlette.applications import Starlette
from starlette.testclient import TestClient

from openmirror.main import STATIC, Revalidated


def _client() -> TestClient:
    app = Starlette()
    app.mount('/static', Revalidated(directory=STATIC), name='static')
    return TestClient(app)


def test_an_asset_is_revalidated_rather_than_assumed_fresh():
    response = _client().get('/static/app.js')
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-cache'


def test_an_unchanged_asset_still_costs_nothing_to_recheck():
    """The point of `no-cache` over `no-store`: it is asked for, not resent."""
    client = _client()
    first = client.get('/static/app.js')
    again = client.get('/static/app.js', headers={'If-None-Match': first.headers['etag']})
    assert again.status_code == 304
    assert not again.content
