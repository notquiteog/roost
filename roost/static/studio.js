/* Making pictures and video, with every knob the backend actually has.
 *
 * The whole panel is drawn from what the server says. Nothing here knows what
 * a sampler is, or that A1111 has schedulers and OpenAI does not — it asks
 * `/api/media/describe`, gets a list of parameters with types, ranges and a
 * sentence each, and renders it. So installing a new sampler on the diffusion
 * box adds it to this dropdown, and adding a backend adds a form, with no
 * release in between.
 *
 * Two decisions about how much to show.
 *
 * **Advanced is a fold, not a lock.** Everything is present and settable; the
 * ones a person needs most days are open and the other twenty are one click
 * away. Hiding a parameter behind a preference would mean the person who
 * wants it has to find out it exists first.
 *
 * **Every control carries its own sentence.** They come from the provider,
 * they are the same words the model is given, and they are the difference
 * between setting a CFG scale and moving a slider called CFG.
 */

import { $, api, el, json, post } from './dom.js';

let kind = 'image';
let schema = [];
const values = new Map();
let polling = null;

const GROUPS = ['prompt', 'shape', 'motion', 'sampling', 'quality', 'output'];

function status(text) {
  $('#studio-status').textContent = text;
}

/* One control, chosen by what the parameter is. A range for anything with
   both ends known, because dragging tells you the shape of the setting in a
   way a number field never does; a number field otherwise, because a slider
   with no bounds is a lie about what is allowed. */
function control(param) {
  const row = el('div', 'param');
  const top = el('div', 'top');
  const label = el('label', '', param.label);
  label.htmlFor = `p-${param.name}`;
  top.appendChild(label);

  const shown = el('span', 'value');
  top.appendChild(shown);
  row.appendChild(top);

  const current = values.has(param.name) ? values.get(param.name) : param.default;
  let input;

  if (param.kind === 'bool') {
    input = el('input');
    input.type = 'checkbox';
    input.checked = Boolean(current);
    top.insertBefore(input, shown);
    label.remove();
    top.appendChild(el('label', '', param.label));
  } else if (param.kind === 'enum' && param.options.length) {
    input = el('select');
    for (const option of param.options) {
      const node = el('option', '', option);
      node.value = option;
      input.appendChild(node);
    }
    if (current) input.value = current;
    row.appendChild(input);
  } else if (param.kind === 'text') {
    input = el('textarea');
    input.rows = param.name === 'prompt' ? 3 : 2;
    input.value = current || '';
    row.appendChild(input);
  } else if ((param.kind === 'int' || param.kind === 'float') && param.min !== null && param.max !== null) {
    input = el('input');
    input.type = 'range';
    input.min = param.min;
    input.max = param.max;
    input.step = param.step || (param.kind === 'int' ? 1 : 0.1);
    input.value = current ?? param.min;
    shown.textContent = input.value;
    row.appendChild(input);
  } else {
    input = el('input');
    input.type = param.kind === 'int' || param.kind === 'float' || param.kind === 'seed' ? 'number' : 'text';
    if (param.min !== null) input.min = param.min;
    if (param.max !== null) input.max = param.max;
    if (current !== null && current !== undefined) input.value = current;
    row.appendChild(input);
  }

  input.id = `p-${param.name}`;
  const read = () => (param.kind === 'bool' ? input.checked : input.value);
  input.oninput = () => {
    values.set(param.name, read());
    if (input.type === 'range') shown.textContent = input.value;
  };
  input.onchange = input.oninput;

  if (param.help) row.appendChild(el('div', 'why', param.help));
  return row;
}

function render() {
  const basic = $('#studio-params');
  const advanced = $('#studio-advanced-params');
  basic.textContent = '';
  advanced.textContent = '';

  // The prompt has its own box at the top of the form, so it is not repeated
  // as a parameter — two prompt fields is the sort of thing that gets typed
  // into the wrong one.
  const shown = schema.filter((p) => p.name !== 'prompt');
  const byGroup = (list) => {
    const out = [];
    for (const group of GROUPS) {
      const rows = list.filter((p) => p.group === group);
      if (!rows.length) continue;
      out.push({ group, rows });
    }
    const rest = list.filter((p) => !GROUPS.includes(p.group));
    if (rest.length) out.push({ group: 'other', rows: rest });
    return out;
  };

  for (const [target, list] of [
    [basic, shown.filter((p) => !p.advanced)],
    [advanced, shown.filter((p) => p.advanced)],
  ]) {
    for (const { group, rows } of byGroup(list)) {
      target.appendChild(el('div', 'param group-head', group));
      for (const param of rows) target.appendChild(control(param));
    }
  }

  const count = shown.filter((p) => p.advanced).length;
  $('#studio-advanced').textContent = count ? `Everything else — ${count} more ›` : 'Nothing else to tune';
  $('#studio-advanced').disabled = !count;
}

async function describe() {
  status('asking what it can do…');
  const provider = $('#studio-provider').value;
  const model = $('#studio-model').value;
  const params = new URLSearchParams({ kind });
  if (provider) params.set('provider', provider);
  if (model) params.set('model', model);

  const res = await api(`/api/media/describe?${params}`);
  if (!res) {
    status('the daemon is not reachable');
    return;
  }
  if (!res.ok) {
    const detail = (await res.json().catch(() => ({}))).detail || 'nothing is configured for this';
    status(detail);
    schema = [];
    render();
    return;
  }

  const info = await res.json();
  schema = info.params || [];
  values.clear();

  const models = $('#studio-model');
  models.textContent = '';
  for (const m of info.models || []) {
    const option = el('option', '', String(m.label || m.id));
    option.value = m.id;
    models.appendChild(option);
  }
  if (info.model) models.value = info.model;
  if (!models.options.length) {
    const option = el('option', '', kind === 'video' ? 'no workflow templates installed' : 'no models');
    option.value = '';
    models.appendChild(option);
  }

  status(`${info.provider}${info.local ? ' · on this machine' : ''}`);
  render();
}

function card(item) {
  const box = el('div', 'shot-card');
  if (item.kind === 'video') {
    const video = document.createElement('video');
    video.src = item.url;
    video.controls = true;
    video.loop = true;
    video.muted = true;
    box.appendChild(video);
  } else {
    const img = document.createElement('img');
    img.src = item.url;
    img.loading = 'lazy';
    img.alt = item.prompt || 'generated image';
    img.onclick = () => window.open(item.url, '_blank');
    box.appendChild(img);
  }

  const under = el('div', 'under');
  if (item.prompt) under.appendChild(el('span', 'prompt', item.prompt));
  // The recipe, not decoration: seed and settings are what make a result
  // reproducible, and a picture you cannot make again is one you cannot
  // change one thing about.
  const bits = [item.model];
  // The seed the backend actually used, not the -1 that was asked for: -1 is
  // a request for a random one and is worthless for reproducing anything.
  if (item.seed !== null && item.seed !== undefined) bits.push(`seed ${item.seed}`);
  const redundant = new Set([
    'prompt', 'negative_prompt', 'n', 'seed',
    // `size` is the same fact as width and height, and both are stored
    // because the two families of backend take one or the other.
    ...(item.params && item.params.width ? ['size'] : []),
  ]);
  for (const [key, value] of Object.entries(item.params || {})) {
    if (redundant.has(key)) continue;
    // A setting left at its default is not information. Booleans that are off
    // and numbers that are zero are the bulk of a long recipe and none of them
    // did anything.
    if (value === false || value === '' || value === null) continue;
    bits.push(`${key} ${value}`);
  }
  under.appendChild(el('span', 'recipe', bits.filter(Boolean).join(' · ')));
  box.appendChild(under);
  return box;
}

async function gallery() {
  const data = await json('/api/media?limit=40');
  const box = $('#studio-gallery');
  box.textContent = '';
  if (!data || !data.media.length) {
    box.appendChild(el('p', 'meta', 'Nothing made yet.'));
    return;
  }
  for (const item of data.media) box.appendChild(card(item));
}

function collect() {
  const out = {};
  for (const param of schema) {
    if (param.name === 'prompt') continue;
    const value = values.has(param.name) ? values.get(param.name) : param.default;
    if (value === null || value === undefined || value === '') continue;
    out[param.name] = value;
  }
  return out;
}

async function make(event) {
  event.preventDefault();
  const prompt = $('#studio-prompt').value.trim();
  if (!prompt) {
    status('say what to make first');
    return;
  }

  const body = {
    prompt,
    provider: $('#studio-provider').value || null,
    model: $('#studio-model').value || null,
    params: collect(),
  };

  $('#studio-make').disabled = true;
  status(kind === 'video' ? 'starting — video takes minutes' : 'making it…');

  const res = await post(`/api/media/${kind}`, body);
  $('#studio-make').disabled = false;

  if (!res) {
    status('the daemon is not reachable');
    return;
  }
  if (!res.ok) {
    status((await res.json().catch(() => ({}))).detail || 'it failed');
    return;
  }

  const result = await res.json();
  if (kind === 'video') {
    watchJob(result.id);
    return;
  }

  const ignored = result.ignored && result.ignored.length
    ? ` — ignored ${result.ignored.join(', ')}, which this backend has no setting for`
    : '';
  status(`${result.media.length} made on ${result.provider}${ignored}`);
  gallery();
}

/* Video is a job. Polled rather than pushed, because ComfyUI behind Perch
   exposes polling and nothing else, so a progress bar here would be a
   fiction — what it can honestly say is how long it has been going. */
function watchJob(id) {
  clearInterval(polling);
  polling = setInterval(async () => {
    const job = await json(`/api/media/jobs/${id}`);
    if (!job) return;
    if (job.state === 'running') {
      status(`rendering — ${job.elapsed}s so far`);
      return;
    }
    clearInterval(polling);
    polling = null;
    if (job.state === 'done') {
      status(`done in ${job.elapsed}s`);
      gallery();
    } else {
      status(`${job.state}: ${job.error || 'no reason given'}`);
    }
  }, 4000);
}

export function stopPolling() {
  clearInterval(polling);
  polling = null;
}

export async function openStudio() {
  await gallery();
  if (!schema.length) await describe();
}

export function wireStudio(providers = []) {
  const select = $('#studio-provider');
  const previous = select.value;
  select.textContent = '';
  const auto = el('option', '', 'whatever is configured');
  auto.value = '';
  select.appendChild(auto);
  for (const p of providers) {
    const option = el('option', '', `${p.label}${p.local ? ' (local)' : ''}`);
    option.value = p.id;
    select.appendChild(option);
  }

  for (const button of document.querySelectorAll('.studio-kind .seg')) {
    button.onclick = () => {
      kind = button.dataset.kind;
      for (const other of document.querySelectorAll('.studio-kind .seg')) {
        other.classList.toggle('on', other === button);
      }
      describe();
    };
  }

  select.onchange = describe;
  $('#studio-model').onchange = describe;

  // This runs again every time a connection is added or removed, and the form
  // on screen was built from what the old set of providers could do. Rebuilt
  // rather than left: connecting a diffusion server and finding the panel
  // still saying nothing is configured is the exact moment someone concludes
  // the connection did not work.
  if (previous && [...select.options].some((o) => o.value === previous)) select.value = previous;
  if (!$('#pane-studio').hidden) {
    schema = [];
    describe();
  }
  $('#studio-form').onsubmit = make;
  $('#studio-advanced').onclick = function () {
    const open = this.getAttribute('aria-expanded') !== 'true';
    this.setAttribute('aria-expanded', String(open));
    $('#studio-advanced-params').hidden = !open;
  };
}
