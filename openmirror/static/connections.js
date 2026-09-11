/* Adding, editing and routing the things that answer.
 *
 * Two panels in one dialog, and the split is the point of the whole feature.
 * The top half is *routing* — which connection answers chat, which answers
 * embedding, which makes pictures — and every modality is its own row. The
 * bottom half is the connections themselves.
 *
 * Chat and embedding being separate rows is not a nicety. They send different
 * things to different places: your questions go to one and your entire
 * remembered history goes to the other, and a UI that offers "your AI
 * provider" as one setting is a UI that will eventually ship someone's memory
 * to a vendor they picked to write code with.
 *
 * The key field is write-only, here and in the API. It is never sent back to
 * this page, so editing a connection's Tor setting cannot accidentally blank
 * its key — the server keeps what it has when this sends nothing.
 */

import { $, api, el, json } from './dom.js';

let catalog = { hosts: [], embedding_models: [] };

const MODALITIES = [
  ['chat', 'Chat', 'The model that thinks. Everything you type goes here.'],
  ['embedding', 'Memory', 'Turns what you tell it into vectors. Your whole remembered '
    + 'history goes through this one, which is why it is chosen separately.'],
  ['stt', 'Dictation', 'Turns your speech into text.'],
  ['tts', 'Speech', 'Reads replies aloud.'],
  ['image', 'Images', ''],
  ['video', 'Video', ''],
];

function host(id) {
  return catalog.hosts.find((h) => h.id === id);
}

async function loadCatalog() {
  const data = await json('/api/providers/catalog');
  if (data) catalog = data;

  const select = $('#conn-host');
  select.textContent = '';
  for (const h of catalog.hosts) {
    const option = el('option', '', `${h.label}${h.local ? ' (your hardware)' : ''}`);
    option.value = h.id;
    select.appendChild(option);
  }
  fillHost();
}

function fillHost() {
  const chosen = host($('#conn-host').value);
  if (!chosen) return;
  $('#conn-url').value = chosen.base_url;
  $('#conn-local').checked = chosen.local;
  // Where the key comes from, for the one host whose key is not from a
  // billing page. Everything else keeps the generic hint.
  $('#conn-key').placeholder = chosen.id === 'perch'
    ? "perch_… — from Perch's console, Connect page"
    : 'leave blank if it needs none';

  const bits = [];
  if (chosen.note) bits.push(chosen.note);
  if (!chosen.lists_models) {
    bits.push('This service does not publish a model list, so the names offered come from '
      + 'this build rather than from the server.');
  }
  if (chosen.env_key) bits.push(`Leave the key blank to use ${chosen.env_key} from the environment.`);
  const embeds = catalog.embedding_models.filter((m) => m.served_by[chosen.id]);
  if (embeds.length) {
    bits.push('Embedding models here: ' + embeds.map((m) => `${m.label} (${m.dimensions}d)`).join(', ') + '.');
  }
  $('#conn-note').textContent = bits.join(' ');
}

async function loadRoutes(providers, capabilities, routes) {
  const box = $('#conn-routes');
  box.textContent = '';

  for (const [id, label, why] of MODALITIES) {
    const able = capabilities[id] || [];
    const row = el('div', 'row');
    row.appendChild(el('span', '', label));

    const select = el('select');
    const auto = el('option', '', able.length ? 'first that can' : 'nothing can do this yet');
    auto.value = '';
    select.appendChild(auto);
    for (const providerId of able) {
      const info = providers.find((p) => p.id === providerId);
      const option = el('option', '', `${providerId}${info && info.local ? ' (local)' : ''}`);
      option.value = providerId;
      select.appendChild(option);
    }
    select.disabled = !able.length;
    if (routes[id]) select.value = routes[id].provider;

    const model = el('input');
    model.placeholder = 'model (optional)';
    model.value = (routes[id] && routes[id].model) || '';
    model.style.width = '150px';

    const save = async () => {
      if (!select.value) return;
      const res = await api('/api/providers/routes', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ modality: id, provider: select.value, model: model.value.trim() }),
      });
      if (res && !res.ok) {
        $('#conn-status').textContent = (await res.json().catch(() => ({}))).detail || 'that route was refused';
      } else {
        $('#conn-status').textContent = `${label} now goes to ${select.value}.`;
      }
    };
    select.onchange = save;
    model.onchange = save;

    row.appendChild(select);
    row.appendChild(model);
    if (why) row.title = why;
    box.appendChild(row);
  }
}

async function loadList(providers) {
  const box = $('#conn-list');
  box.textContent = '';
  if (!providers.length) {
    box.appendChild(el('li', 'meta', 'Nothing connected yet.'));
    return;
  }

  for (const p of providers) {
    const li = el('li');
    const who = el('div', 'who');
    who.appendChild(el('span', '', p.label || p.id));
    who.appendChild(el('small', '', `${p.base_url || ''} · ${p.modalities.join(', ')}`));
    li.appendChild(who);

    if (p.local) li.appendChild(el('span', 'tag local', 'your hardware'));
    if (p.tor) li.appendChild(el('span', 'tag tor', 'tor'));
    if (!p.editable) li.appendChild(el('span', 'tag', 'from .env'));

    // The connection behind this row. Usually the row's own id; for Perch it
    // is one connection behind up to five rows, none of which share its id.
    const cid = p.connection_id || p.id;

    const test = el('button', 'ghost small', 'Test');
    test.onclick = async () => {
      test.textContent = 'testing…';
      const result = await json(`/api/providers/connections/${cid}/test`, { method: 'POST' });
      // A count and a few real names, rather than a tick: it proves the
      // address, the key and the route in one, which a tick does not.
      test.textContent = result
        ? (result.ok ? `${result.models} models` : 'failed')
        : 'no answer';
      if (result && !result.ok) test.title = result.error;
      setTimeout(() => { test.textContent = 'Test'; }, 6000);
    };
    li.appendChild(test);

    if (p.editable) {
      const remove = el('button', 'ghost small', 'Remove');
      remove.onclick = async () => {
        // Said, because it is not what the row suggests: every Perch row is the
        // same connection, and removing one removes all of them.
        const what = cid !== p.id
          ? `Disconnect ${cid}? Every one of its services goes with it.`
          : `Disconnect ${p.label || p.id}?`;
        if (!confirm(what)) return;
        await api(`/api/providers/connections/${cid}`, { method: 'DELETE' });
        refresh();
      };
      li.appendChild(remove);
    }
    box.appendChild(li);
  }
}

export async function refresh() {
  const data = await json('/api/providers');
  if (!data) return;
  await loadRoutes(data.providers, data.capabilities || {}, data.routes || {});
  await loadList(data.providers);

  const tor = await json('/api/providers/tor');
  if (tor) {
    $('#conn-tor-note').textContent = tor.installed
      ? `— the proxy at ${tor.proxy} ${tor.listening ? 'is answering' : 'is not answering'}. `
        + 'Checked before saving; if it is not there, this is refused rather than saved '
        + 'and quietly ignored.'
      : '— needs the socks extra: pip install "openmirror[tor]"';
  }
}

async function add(event) {
  event.preventDefault();
  $('#conn-status').textContent = 'connecting…';

  const res = await api('/api/providers/connections', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      host_id: $('#conn-host').value,
      base_url: $('#conn-url').value.trim(),
      // Undefined rather than '' when blank, so the server keeps whatever key
      // it already has instead of clearing it.
      api_key: $('#conn-key').value || undefined,
      local: $('#conn-local').checked,
      tor: $('#conn-tor').checked,
    }),
  });

  if (!res) {
    $('#conn-status').textContent = 'the daemon is not reachable';
    return;
  }
  if (!res.ok) {
    $('#conn-status').textContent = (await res.json().catch(() => ({}))).detail || 'that did not connect';
    return;
  }

  // Which services came up, for a host that is several. "connected" alone
  // would hide that Perch has video switched off, which is exactly the thing
  // somebody about to route video at it needs to know.
  const body = await res.json().catch(() => ({}));
  const off = Object.entries(body.services || {}).filter(([, up]) => !up).map(([s]) => s);
  $('#conn-key').value = '';
  $('#conn-status').textContent = body.registered && body.registered.length
    ? `connected — ${body.registered.map((id) => id.split(':').pop()).join(', ')}`
      + (off.length ? `; ${off.join(', ')} switched off` : '')
    : 'connected';
  refresh();
}

/* Fill the panel in. Exported rather than bound to a button of its own,
   because connections live in a tab of the settings dialog now: what opens
   them is that dialog, and this is what it calls when the tab is shown. */
export async function openConnections() {
  await loadCatalog();
  await refresh();
}

export function wireConnections() {
  $('#conn-host').onchange = fillHost;
  $('#conn-add').onsubmit = add;
}
