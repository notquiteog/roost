/* Searching the web yourself.
 *
 * Worth having for a reason that is not obvious: the agent's search backend
 * is configured once, with a key, and is usually better than a browser tab —
 * a self-hosted SearxNG, or a Brave key that does not track you. Making it
 * reachable directly means "look this up" does not have to cost a model turn,
 * and someone who only wanted the links gets the links.
 *
 * The reader is the other half. A result you can read here rather than in a
 * new tab is a result you can read without leaving what you were doing, and
 * it goes through the same fetch the agent uses — so the private-address
 * guard applies to a person clicking a link, not only to a model following
 * one.
 */

import { $, el, json } from './dom.js';

async function run(event) {
  event.preventDefault();
  const query = $('#search-q').value.trim();
  if (!query) return;

  $('#search-meta').textContent = 'searching…';
  $('#search-results').textContent = '';

  const data = await json(`/api/search?q=${encodeURIComponent(query)}&count=12`);
  if (!data) {
    $('#search-meta').textContent =
      'the search backend did not answer. The keyless fallback is blocked in practice — '
      + 'set ROOST_SEARCH_BACKEND to brave or tavily with a key, or to your own searxng.';
    return;
  }

  $('#search-meta').textContent = `${data.results.length} results via ${data.backend}`;
  const box = $('#search-results');
  for (const hit of data.results) {
    const row = el('div', 'hit');
    const title = el('a', 'title', hit.title || hit.url);
    title.href = hit.url;
    title.target = '_blank';
    title.rel = 'noreferrer';
    row.appendChild(title);
    row.appendChild(el('div', 'url', hit.url));
    if (hit.snippet) row.appendChild(el('div', 'snippet', hit.snippet));

    const read = el('button', 'ghost small read', 'Read it here');
    read.onclick = () => reader(hit.url, hit.title || hit.url);
    row.appendChild(read);
    box.appendChild(row);
  }
}

async function reader(url, title) {
  $('#search-reader').hidden = false;
  $('#reader-title').textContent = title;
  $('#reader-body').textContent = 'fetching…';

  const data = await json(`/api/search/fetch?url=${encodeURIComponent(url)}`);
  $('#reader-body').textContent = data
    ? data.text
    : 'that page could not be fetched. Private and link-local addresses are refused '
      + 'on purpose — a page reached through here can otherwise reach services never '
      + 'exposed to the internet.';
}

export function wireSearch() {
  $('#search-form').onsubmit = run;
  $('#reader-close').onclick = () => { $('#search-reader').hidden = true; };
}

export function focusSearch() {
  $('#search-q').focus();
}
