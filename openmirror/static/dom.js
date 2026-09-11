/* The four things every part of this page needs.
 *
 * Extracted when the client grew from one screen to six. They were duplicated
 * for about an hour first, which was long enough to notice that `api` in
 * particular must not be: it is the one place that knows the daemon has gone,
 * and two copies of that knowledge means one of them showing "live" while the
 * other has been failing for a minute.
 */

export const $ = (sel, root = document) => root.querySelector(sel);

export const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

/* One of the symbols defined at the top of the page. */
export function icon(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'ic');
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  use.setAttribute('href', `#i-${name}`);
  svg.appendChild(use);
  return svg;
}

/* Whoever wants to know when the daemon comes and goes. */
const reachability = new Set();
let reachable = true;

export function onReachable(fn) {
  reachability.add(fn);
}

function setReachable(ok) {
  if (ok === reachable) return;
  reachable = ok;
  for (const fn of reachability) fn(ok);
}

/* Every call to the daemon goes through here.
 *
 * The page is expected to outlive the process it talks to: the daemon gets
 * restarted, the laptop sleeps, the tab sits open overnight. So a failed fetch
 * is an ordinary condition rather than an exception, and callers get null
 * instead of a rejection — an unhandled rejection in a five-second poll is
 * particularly bad, because that poll is the thing that notices the daemon
 * came back. */
export async function api(path, options) {
  try {
    const res = await fetch(path, options);
    setReachable(true);
    return res;
  } catch {
    setReachable(false);
    return null;
  }
}

/* The common case: a JSON body, or null if anything at all went wrong. The
 * status is deliberately not exposed here — a caller that needs to tell 404
 * from 500 should use `api` and look. */
export async function json(path, options) {
  const res = await api(path, options);
  if (!res || !res.ok) return null;
  try {
    return await res.json();
  } catch {
    return null;
  }
}

export async function post(path, body) {
  return api(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

/* The token, if this page was opened with one. Every socket needs it and
 * every one of them was reading it out of the query string separately. */
export function token() {
  return new URLSearchParams(location.search).get('token');
}

/* A websocket URL on this origin, with the token attached. */
export function socket(path, params = {}) {
  const url = new URL(path, location.href);
  url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  for (const [key, value] of Object.entries(params)) {
    if (value !== null && value !== undefined && value !== '') url.searchParams.set(key, String(value));
  }
  const t = token();
  if (t) url.searchParams.set('token', t);
  return url;
}

export function took(ms) {
  // Rounded: a /command that answers at once takes a few microseconds, and
  // the raw figure printed as "0.0000419464111328ms".
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(ms < 10000 ? 1 : 0)}s`;
  const mins = Math.floor(ms / 60000);
  return `${mins}m ${Math.round((ms % 60000) / 1000)}s`;
}
