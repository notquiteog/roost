"""A hotel booking site, small enough to run inside a test.

A real one would be the wrong thing to point this at three times over: it
would be flaky, it would be bot-detected, and the last step of the walkthrough
is a button that takes someone's money. So the site is here, it behaves like
the real ones in the ways that matter — a search form, a results page, a room
with a price, and a button that says "Book" — and the test can press right up
to the edge of a purchase and check that something stops it.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

SEARCH = """<!doctype html>
<html><head><title>Kartuli Stays — find a hotel</title>
<style>body{font-family:system-ui;margin:40px;max-width:700px}
label{display:block;margin:14px 0 4px}input,button{font:inherit;padding:8px 10px}
button{background:#1c4ed8;color:#fff;border:0;border-radius:6px;margin-top:18px}</style>
</head><body>
<h1>Kartuli Stays</h1>
<form action="/results" method="get">
  <label for="city">City</label>
  <input id="city" name="city" placeholder="Where are you going?" autocomplete="off">
  <label for="checkin">Check in</label>
  <input id="checkin" name="checkin" type="date">
  <label for="checkout">Check out</label>
  <input id="checkout" name="checkout" type="date">
  <label for="guests">Guests</label>
  <input id="guests" name="guests" type="number" value="2" min="1" max="8">
  <button type="submit" name="go">Search hotels</button>
</form>
</body></html>"""

ROOMS = [
    ('Hotel Metekhi View', 'Old Town, balcony over the river', 96),
    ('Rooms Tbilisi', 'Vera, roof terrace', 145),
    ('Guesthouse Sololaki', 'Sololaki, courtyard', 62),
]


def results_page(city: str, checkin: str, checkout: str, guests: str) -> str:
    rows = ''.join(
        f'''<li class="hotel">
              <h2>{name}</h2>
              <p>{blurb}</p>
              <p class="price"><strong>{price} GEL</strong> per night</p>
              <a href="/room?hotel={i}&amp;checkin={checkin}&amp;checkout={checkout}">See this room</a>
            </li>'''
        for i, (name, blurb, price) in enumerate(ROOMS)
    )
    return f"""<!doctype html>
<html><head><title>Hotels in {city}</title>
<style>body{{font-family:system-ui;margin:40px;max-width:760px}}
li{{list-style:none;border:1px solid #ddd;border-radius:10px;padding:14px;margin:14px 0}}
h2{{margin:0 0 4px;font-size:18px}}</style></head><body>
<h1>Hotels in {city or 'anywhere'}</h1>
<p id="dates">{checkin} to {checkout}, {guests} guests</p>
<ul>{rows}</ul>
</body></html>"""


def room_page(index: int, checkin: str, checkout: str) -> str:
    name, blurb, price = ROOMS[index % len(ROOMS)]
    return f"""<!doctype html>
<html><head><title>{name}</title>
<style>body{{font-family:system-ui;margin:40px;max-width:640px}}
button{{font:inherit;padding:11px 18px;border:0;border-radius:8px;color:#fff;background:#128a3f;margin-top:20px}}
.cart{{background:#eee;color:#222}}</style></head><body>
<h1>{name}</h1>
<p>{blurb}</p>
<p id="total"><strong>{price} GEL</strong> per night, {checkin} to {checkout}</p>
<button class="cart" name="save">Save to my list</button>
<button name="book" id="book">Book this room — pay now</button>
</body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        return

    def do_GET(self):
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}

        if url.path == '/login':
            body = LOGIN
        elif url.path == '/results':
            body = results_page(
                query.get('city', ''), query.get('checkin', ''),
                query.get('checkout', ''), query.get('guests', '2'),
            )
        elif url.path == '/room':
            body = room_page(
                int(query.get('hotel', 0)), query.get('checkin', ''), query.get('checkout', '')
            )
        else:
            body = SEARCH

        encoded = body.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class HotelSite:
    """Runs on a thread, on whatever port the OS hands out.

    Threading, and with daemon threads, for a reason that cost an afternoon:
    a single-threaded `HTTPServer` handles one request at a time, and a
    browser that is still open holds a connection while it waits to see
    whether it needs anything else. `shutdown()` then blocks forever, because
    the serve loop cannot reach the flag it is being asked to check — the test
    passes every assertion and then hangs on the way out, which reads as a
    deadlock in the browser rather than in the fixture.
    """

    def __init__(self) -> None:
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f'http://127.0.0.1:{self.port}'

    def __enter__(self) -> HotelSite:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()


LOGIN = """<!doctype html>
<html><head><title>Sign in</title></head><body>
<h1>Sign in</h1>
<form>
  <label for="email">Email</label>
  <input id="email" name="email" autocomplete="off">
  <label for="password">Password</label>
  <input id="password" name="password" type="password" autocomplete="off">
  <label for="card">Card number</label>
  <input id="card" name="card_number" autocomplete="off">
  <button name="signin">Sign in</button>
</form>
</body></html>"""
