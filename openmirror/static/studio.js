/* Making pictures and video, with every knob the backend actually has.
 *
 * The whole panel is drawn from what the server says. Nothing here knows what
 * a sampler is, or that A1111 has schedulers and OpenAI does not — it asks
 * `/api/media/describe`, gets a list of parameters with types, ranges and a
 * sentence each, and renders it. So installing a new sampler on the diffusion
 * box adds it to this dropdown, and a Replicate model published this morning
 * arrives with its real controls, because Replicate publishes the schema and
 * the form is built from that.
 *
 * Five decisions shape this file.
 *
 * **Advanced is a fold, not a lock.** Everything is present and settable; the
 * ones a person needs most days are open and the other twenty are one click
 * away. Hiding a parameter behind a preference would mean the person who
 * wants it has to find out it exists first.
 *
 * **Every control carries its own sentence.** They come from the provider,
 * they are the same words the model is given, and they are the difference
 * between setting a CFG scale and moving a slider called CFG.
 *
 * **The recipe is a control, not a caption.** A result stores the seed and
 * every setting that made it, and the only reason to store that is to change
 * one thing and go again — so it is a button. Before this, the recipe was
 * printed under each picture and there was no way to act on it, which is a
 * museum label on something that should have been a workbench.
 *
 * **A model box takes typing.** For Replicate and fal the interesting model
 * is the one that came out yesterday, so the picker is a combo box over a
 * datalist rather than a closed dropdown: everything known is offered and
 * anything at all can be typed.
 *
 * **Settings survive a model change.** `values` is keyed by parameter name and
 * is not cleared when the form is rebuilt, so switching from one FLUX endpoint
 * to another keeps the prompt, the size and the seed — the names match, so the
 * values carry. Anything the new model does not have simply is not sent.
 */

import { $, api, el, json, post } from './dom.js';

let kind = 'image';
let schema = [];
let known = { providers: [], models: [] };
const values = new Map();

/* The gallery, held here so the lightbox can step through it with the arrow
   keys without asking the server for a page it already has. */
let shots = [];
let filter = '';
let search = '';
let exhausted = false;
let shown = -1;

let jobTimer = null;
const PAGE = 40;

const GROUPS = ['prompt', 'shape', 'motion', 'sampling', 'quality', 'output'];

/* Named sizes, offered as buttons when a backend sizes by pixels. Dragging two
   sliders to 1344×768 to get 16:9 is arithmetic nobody should be doing in a
   form; the sliders stay, because a backend that takes any size should not be
   narrowed to six of them. */
const RATIOS = [
  ['1:1', 1, 1], ['4:3', 4, 3], ['3:4', 3, 4],
  ['16:9', 16, 9], ['9:16', 9, 16], ['3:2', 3, 2], ['2:3', 2, 3],
];

function status(text, tone = '') {
  const node = $('#studio-status');
  node.textContent = text;
  node.className = `meta ${tone}`;
}

/* -------------------------------------------------------------- controls */

/* One control, chosen by what the parameter is. A range for anything with
   both ends known, because dragging tells you the shape of the setting in a
   way a number field never does; a number field otherwise, because a slider
   with no bounds is a lie about what is allowed. */
function control(param) {
  if (param.kind === 'image') return imageControl(param);

  const row = el('div', 'param');
  const top = el('div', 'top');
  const label = el('label', '', param.label);
  label.htmlFor = `p-${param.name}`;
  top.appendChild(label);

  const shownValue = el('span', 'value');
  top.appendChild(shownValue);
  row.appendChild(top);

  const current = values.has(param.name) ? values.get(param.name) : param.default;
  let input;

  if (param.kind === 'bool') {
    input = el('input');
    input.type = 'checkbox';
    input.checked = Boolean(current);
    top.insertBefore(input, shownValue);
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
    shownValue.textContent = input.value;
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
    if (input.type === 'range') shownValue.textContent = input.value;
    markChanged();
  };
  input.onchange = input.oninput;

  // A seed is the one number people want to throw away deliberately, and
  // typing -1 to mean "surprise me" is a piece of folklore rather than a
  // control.
  if (param.kind === 'seed') {
    const reroll = el('button', 'ghost tiny', 'New seed');
    reroll.type = 'button';
    reroll.onclick = () => {
      input.value = '-1';
      values.set(param.name, '-1');
      markChanged();
    };
    top.appendChild(reroll);
  }

  if (param.help) row.appendChild(el('div', 'why', param.help));
  return row;
}

/* A picture, which is a parameter like any other and looks nothing like one.
 *
 * The file goes to the store and what is kept here is its id, so the value
 * that travels with the rest of the form is a short string the server put
 * there itself — never a path from the client and never a megabyte of base64
 * riding along in the JSON. */
function imageControl(param) {
  const row = el('div', 'param param-image');
  const top = el('div', 'top');
  top.appendChild(el('label', '', param.label));
  row.appendChild(top);

  const drop = el('div', 'dropzone');
  const strip = el('div', 'thumbs');
  const hint = el('span', 'hint', 'Drop a picture, or click to choose');

  const picker = el('input');
  picker.type = 'file';
  picker.accept = 'image/png,image/jpeg,image/webp,image/gif';
  picker.multiple = true;
  picker.hidden = true;

  const ids = () => String(values.get(param.name) || '').split(',').filter(Boolean);

  function paint() {
    strip.textContent = '';
    const current = ids();
    for (const id of current) {
      const thumb = el('div', 'thumb');
      const img = document.createElement('img');
      img.src = `/api/media/${id}/file`;
      img.alt = '';
      thumb.appendChild(img);

      const remove = el('button', 'x', '×');
      remove.type = 'button';
      remove.title = 'Use a different picture';
      remove.onclick = (event) => {
        event.stopPropagation();
        values.set(param.name, current.filter((other) => other !== id).join(','));
        paint();
        markChanged();
      };
      thumb.appendChild(remove);
      strip.appendChild(thumb);
    }
    hint.textContent = current.length
      ? 'Drop another, or click to add'
      : 'Drop a picture, or click to choose';
    drop.classList.toggle('filled', current.length > 0);
  }

  async function take(files) {
    const accepted = [...files].filter((f) => f.type.startsWith('image/'));
    if (!accepted.length) return;
    drop.classList.add('busy');
    const added = [];
    for (const file of accepted) {
      // The body is the file. No multipart, no form — see the note on the
      // upload endpoint about why this is the whole request.
      const res = await api('/api/media/upload', {
        method: 'POST',
        headers: { 'Content-Type': file.type, 'X-Openmirror-Filename': encodeURIComponent(file.name) },
        body: file,
      });
      if (!res || !res.ok) {
        status((res && (await res.json().catch(() => ({}))).detail) || 'that picture would not upload', 'bad');
        continue;
      }
      added.push((await res.json()).id);
    }
    drop.classList.remove('busy');
    if (!added.length) return;
    values.set(param.name, [...ids(), ...added].join(','));
    paint();
    markChanged();
  }

  drop.onclick = () => picker.click();
  picker.onchange = () => take(picker.files);
  drop.ondragover = (event) => {
    event.preventDefault();
    drop.classList.add('over');
  };
  drop.ondragleave = () => drop.classList.remove('over');
  drop.ondrop = (event) => {
    event.preventDefault();
    drop.classList.remove('over');
    take(event.dataTransfer.files);
  };

  drop.appendChild(strip);
  drop.appendChild(hint);
  drop.appendChild(picker);
  row.appendChild(drop);
  if (param.help) row.appendChild(el('div', 'why', param.help));
  paint();
  return row;
}

/* The named sizes, when the backend sizes by pixels rather than by ratio. */
function ratios(width, height) {
  const row = el('div', 'param ratios');
  row.appendChild(el('span', 'ratio-label', 'Shape'));
  const base = 1024;
  for (const [name, w, h] of RATIOS) {
    const button = el('button', 'chip-button', name);
    button.type = 'button';
    button.onclick = () => {
      // Snapped to a multiple of 64: every diffusion backend here wants one,
      // and the arithmetic that produces 1365 is the arithmetic that produces
      // a 400 from the server.
      const scale = Math.sqrt((base * base) / (w * h));
      const round = (v) => Math.max(width.min || 64, Math.min(width.max || 4096, Math.round(v / 64) * 64));
      values.set('width', round(w * scale));
      values.set('height', round(h * scale));
      render();
      markChanged();
    };
    row.appendChild(button);
  }
  void height;
  return row;
}

function markChanged() {
  // The button says what pressing it will actually do, which changes with the
  // kind and with whether anything is in flight.
  const make = $('#studio-make');
  make.textContent = kind === 'video' ? 'Render it' : 'Make it';
}

/* ----------------------------------------------------------------- form */

function render() {
  const basic = $('#studio-params');
  const advanced = $('#studio-advanced-params');
  basic.textContent = '';
  advanced.textContent = '';

  // The prompt has its own box at the top of the form, so it is not repeated
  // as a parameter — two prompt fields is the sort of thing that gets typed
  // into the wrong one.
  const rest = schema.filter((p) => p.name !== 'prompt');
  const byGroup = (list) => {
    const out = [];
    for (const group of GROUPS) {
      const rows = list.filter((p) => p.group === group);
      if (rows.length) out.push({ group, rows });
    }
    const other = list.filter((p) => !GROUPS.includes(p.group));
    if (other.length) out.push({ group: 'other', rows: other });
    return out;
  };

  const width = rest.find((p) => p.name === 'width');
  const height = rest.find((p) => p.name === 'height');

  for (const [target, list] of [
    [basic, rest.filter((p) => !p.advanced)],
    [advanced, rest.filter((p) => p.advanced)],
  ]) {
    for (const { group, rows } of byGroup(list)) {
      target.appendChild(el('div', 'param group-head', group));
      if (group === 'shape' && width && height && target === basic) target.appendChild(ratios(width, height));
      for (const param of rows) target.appendChild(control(param));
    }
  }

  const count = rest.filter((p) => p.advanced).length;
  $('#studio-advanced').textContent = count ? `Everything else — ${count} more ›` : 'Nothing else to tune';
  $('#studio-advanced').disabled = !count;
  markChanged();
}

async function describe() {
  status('asking what it can do…');
  const provider = $('#studio-provider').value;
  const model = $('#studio-model').value.trim();
  const params = new URLSearchParams({ kind });
  if (provider) params.set('provider', provider);
  if (model) params.set('model', model);

  const res = await api(`/api/media/describe?${params}`);
  if (!res) {
    status('the daemon is not reachable', 'bad');
    return;
  }
  if (!res.ok) {
    const detail = (await res.json().catch(() => ({}))).detail || 'nothing is configured for this';
    // The two kinds genuinely have different provider lists: six things may
    // make pictures and two of them video. Switching to Video with an
    // image-only backend selected used to dead-end on "provider X is not
    // registered for it" — true, and not something anyone should have to
    // recover from by hand. So the list is refetched for this kind and, if
    // what was chosen cannot do it, that choice is dropped and it asks again.
    // It cannot loop: the second attempt asks with no provider at all.
    const list = await json(`/api/media/providers?kind=${kind}`);
    known.providers = (list && list.providers) || [];
    if (provider && !known.providers.some((p) => p.id === provider)) {
      paintProviders('');
      $('#studio-provider').value = '';
      return describe();
    }
    paintProviders(provider);
    status(detail, 'bad');
    schema = [];
    known.models = [];
    paintModels('');
    render();
    return;
  }

  const info = await res.json();
  schema = info.params || [];
  known = { providers: info.providers || [], models: info.models || [] };

  // `values` is deliberately NOT cleared. Parameter names are shared across
  // backends — prompt, width, seed, steps — so switching model keeps what was
  // set, and what the new model does not declare is simply never sent. Wiping
  // the form on every model change is how a careful set of settings gets lost
  // to a mistaken click.
  paintProviders(info.provider);
  paintModels(info.model);
  status(`${info.provider}${info.local ? ' · on this machine' : ''}`, info.local ? 'good' : '');
  render();
}

function paintProviders(current) {
  const select = $('#studio-provider');
  select.textContent = '';
  const auto = el('option', '', 'whatever is configured');
  auto.value = '';
  select.appendChild(auto);
  for (const p of known.providers) {
    const option = el('option', '', `${p.label}${p.local ? ' · local' : ''}`);
    option.value = p.id;
    select.appendChild(option);
  }
  if (current && [...select.options].some((o) => o.value === current)) select.value = current;
}

function paintModels(current) {
  const list = $('#studio-models');
  list.textContent = '';
  for (const m of known.models) {
    const option = el('option');
    option.value = m.id;
    // The datalist label is where a one-line note about the model goes, which
    // is the only place anyone reads it: a separate panel of model
    // descriptions is a panel nobody opens.
    if (m.note || m.label) option.label = m.note || m.label;
    list.appendChild(option);
  }
  const box = $('#studio-model');
  if (current && !box.value) box.value = current;
  const note = known.models.find((m) => m.id === box.value);
  $('#studio-model-note').textContent = note ? note.note || '' : '';
  $('#studio-model').placeholder = known.models.length
    ? 'pick one, or type any id this backend knows'
    : kind === 'video'
      ? 'no workflow templates installed'
      : 'no models';
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

/* ------------------------------------------------------------- the jobs */

/* Every generation is a job now, image or video, and this is the strip of
 * them. It exists because the previous version could watch exactly one and
 * had no stop button: starting a second render made the first invisible, and
 * a four-minute job you cannot cancel is a GPU working for nobody. */
function paintJobs(jobs) {
  const box = $('#studio-jobs');
  const live = jobs.filter((j) => j.state === 'queued' || j.state === 'running');
  const recent = jobs
    .filter((j) => j.state === 'failed' || j.state === 'cancelled')
    .slice(-3);

  box.textContent = '';
  for (const job of [...live, ...recent]) {
    const row = el('div', `job job-${job.state}`);

    const head = el('div', 'job-head');
    head.appendChild(el('span', 'job-kind', job.kind));
    head.appendChild(el('span', 'job-prompt', job.prompt || '(no prompt)'));
    head.appendChild(el('div', 'spacer'));
    head.appendChild(el('span', 'job-time', `${job.elapsed}s`));

    if (job.state === 'queued' || job.state === 'running') {
      const stop = el('button', 'ghost tiny', 'Stop');
      stop.type = 'button';
      stop.title = 'Cancels it at the provider too, so it stops costing';
      stop.onclick = async () => {
        stop.disabled = true;
        await api(`/api/media/jobs/${job.id}`, { method: 'DELETE' });
        tickJobs();
      };
      head.appendChild(stop);
    }
    row.appendChild(head);

    if (job.state === 'failed' || job.state === 'cancelled') {
      row.appendChild(el('div', 'job-note', job.error || 'stopped'));
    } else {
      // A determinate bar only when something downstream actually reported a
      // number. Everything else gets a moving stripe and a sentence, because
      // a bar that stops at ninety teaches people to distrust the real ones.
      const track = el('div', `bar ${job.progress === null ? 'indeterminate' : ''}`);
      const fill = el('div', 'fill');
      if (job.progress !== null) fill.style.width = `${Math.round(job.progress * 100)}%`;
      track.appendChild(fill);
      row.appendChild(track);
      row.appendChild(el('div', 'job-note',
        job.note + (job.progress !== null ? ` · ${Math.round(job.progress * 100)}%` : '')));
    }
    box.appendChild(row);
  }
  box.hidden = !box.children.length;
  return live.length;
}

let lastDone = 0;
//: A finished job arrived while the lightbox was open, so the gallery is out
//: of date and will be rebuilt when it closes. See `tickJobs`.
let galleryStale = false;

async function tickJobs() {
  const data = await json('/api/media/jobs');
  if (!data) return;
  const jobs = data.jobs || [];
  const live = paintJobs(jobs);

  // Refresh the gallery when the number of finished jobs goes up, rather than
  // on a timer: polling the library every four seconds for a change that
  // happens twice an hour is a request nobody needed.
  const done = jobs.filter((j) => j.state === 'done').length;
  if (done > lastDone) {
    lastDone = done;
    // Never while the lightbox is open: a refresh rebuilds every card, and
    // swapping the one being watched out mid-frame is a flash with no
    // explanation attached to it. It waits for the lightbox to close.
    if ($('#shot-dialog').open) galleryStale = true;
    else await loadGallery(true);
  }

  if (!live && jobTimer) {
    clearInterval(jobTimer);
    jobTimer = null;
  }
}

function watchJobs() {
  lastDone = Math.max(lastDone, 0);
  if (jobTimer) return;
  // Two seconds while something is in flight, and nothing at all when the
  // queue is empty.
  jobTimer = setInterval(tickJobs, 2000);
  tickJobs();
}

/* --------------------------------------------------------------- gallery */

function recipeOf(item) {
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
  return bits.filter(Boolean).join(' · ');
}

function card(item, index) {
  const box = el('figure', 'shot-card');
  box.tabIndex = 0;

  if (item.kind === 'video') {
    const video = document.createElement('video');
    video.src = item.url;
    video.loop = true;
    video.muted = true;
    video.playsInline = true;
    video.preload = 'metadata';
    // Plays on hover and stops on the way out: a grid of twelve autoplaying
    // videos is a grid that pins a laptop fan, and a grid of twelve still
    // frames tells you nothing about which one moved correctly.
    box.onmouseenter = () => video.play().catch(() => {});
    // Paused where it got to, not rewound. Rewinding made a pointer crossing
    // the grid restart every clip it passed over, and a row of clips all
    // snapping back to their first frame reads as the video being broken
    // rather than as a preview stopping.
    box.onmouseleave = () => video.pause();
    box.appendChild(video);
    box.appendChild(el('span', 'badge', 'video'));
  } else {
    const img = document.createElement('img');
    img.src = item.url;
    img.loading = 'lazy';
    img.decoding = 'async';
    img.alt = item.prompt || 'generated image';
    box.appendChild(img);
  }

  const open = () => openShot(index);
  box.onclick = open;
  box.onkeydown = (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      open();
    }
  };

  // The actions live on the card rather than only in the lightbox, because
  // the common one — make this again with one thing changed — should not need
  // a click to get to.
  const tools = el('div', 'shot-tools');
  for (const [label, title, fn] of [
    ['↺', 'Load these settings back into the form', () => reuse(item)],
    ['↓', 'Download', () => download(item)],
    ['✕', 'Delete', () => remove(item)],
  ]) {
    const button = el('button', 'shot-tool', label);
    button.type = 'button';
    button.title = title;
    button.onclick = (event) => {
      event.stopPropagation();
      fn();
    };
    tools.appendChild(button);
  }
  box.appendChild(tools);

  const under = el('figcaption', 'under');
  if (item.prompt) under.appendChild(el('span', 'prompt', item.prompt));
  under.appendChild(el('span', 'recipe', recipeOf(item)));
  box.appendChild(under);
  return box;
}

function skeletons(count) {
  const box = $('#studio-gallery');
  // "Nothing made yet" and a placeholder for the thing being made are a
  // contradiction, and it is the empty message people read.
  const empty = box.querySelector('.empty');
  if (empty) empty.remove();
  for (let i = 0; i < count; i += 1) box.prepend(el('div', 'shot-card skeleton'));
}

async function loadGallery(reset = false) {
  if (reset) {
    shots = [];
    exhausted = false;
  }
  const params = new URLSearchParams({ limit: String(PAGE), offset: String(shots.length) });
  if (filter) params.set('kind', filter);
  if (search) params.set('q', search);

  const data = await json(`/api/media?${params}`);
  if (!data) return;
  shots = shots.concat(data.media);
  exhausted = !data.more;
  paintGallery();
}

function paintGallery() {
  const box = $('#studio-gallery');
  box.textContent = '';
  if (!shots.length) {
    box.appendChild(el('p', 'meta empty',
      search || filter ? 'Nothing here matches that.' : 'Nothing made yet.'));
  } else {
    shots.forEach((item, index) => box.appendChild(card(item, index)));
  }
  $('#studio-more').hidden = exhausted || !shots.length;
}

/* ------------------------------------------------------ acting on a result */

function download(item) {
  const link = el('a');
  link.href = item.url;
  link.download = item.filename || item.id;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

async function remove(item) {
  if (!confirm('Delete this permanently? The file and its recipe both go.')) return;
  const res = await api(`/api/media/${item.id}`, { method: 'DELETE' });
  if (!res || !res.ok) {
    status('it would not delete', 'bad');
    return;
  }
  shots = shots.filter((other) => other.id !== item.id);
  if (shown >= shots.length) shown = shots.length - 1;
  paintGallery();
  if (shown >= 0) paintShot();
  else closeShot();
}

/* Put a result back into the form that made it.
 *
 * This is the reason the recipe is stored at all. Everything goes back —
 * kind, provider, model, prompt and every setting — because "change one thing
 * and see only that change" is the whole loop, and reconstructing nineteen
 * settings by reading them off a caption is not it. */
async function reuse(item) {
  closeShot();
  if (item.kind !== kind) {
    kind = item.kind;
    for (const button of document.querySelectorAll('.studio-kind .seg')) {
      button.classList.toggle('on', button.dataset.kind === kind);
    }
  }
  $('#studio-prompt').value = item.prompt || '';
  $('#studio-provider').value = item.provider || '';
  $('#studio-model').value = item.model || '';

  values.clear();
  for (const [key, value] of Object.entries(item.params || {})) values.set(key, value);
  // The seed it actually used, not the -1 that was asked for — reusing a
  // recipe with seed -1 reproduces nothing, which is the opposite of the point.
  if (item.seed !== null && item.seed !== undefined) values.set('seed', item.seed);

  await describe();
  $('#studio-prompt').focus();
  status('settings loaded — change one thing and go again', 'good');
}

/* ------------------------------------------------------------- lightbox */

function openShot(index) {
  shown = index;
  paintShot();
  const dialog = $('#shot-dialog');
  if (!dialog.open) dialog.showModal();
}

function closeShot() {
  const dialog = $('#shot-dialog');
  if (dialog.open) dialog.close();
  shown = -1;
}

function step(by) {
  if (shown < 0) return;
  const next = shown + by;
  if (next < 0 || next >= shots.length) return;
  shown = next;
  paintShot();
}

function paintShot() {
  const item = shots[shown];
  if (!item) return;

  const stage = $('#shot-stage');
  stage.textContent = '';
  if (item.kind === 'video') {
    const video = document.createElement('video');
    video.src = item.url;
    video.controls = true;
    video.loop = true;
    video.autoplay = true;
    stage.appendChild(video);
  } else {
    const img = document.createElement('img');
    img.src = item.url;
    img.alt = item.prompt || '';
    stage.appendChild(img);
  }

  $('#shot-prompt').textContent = item.prompt || '(no prompt)';
  $('#shot-where').textContent =
    `${item.provider} · ${item.model}${item.seed !== null && item.seed !== undefined ? ` · seed ${item.seed}` : ''}`;
  $('#shot-count').textContent = `${shown + 1} of ${shots.length}`;

  // The full recipe, one setting per row, rather than the single run-on line
  // the card shows. Same facts; this is the one you read rather than glance at.
  const table = $('#shot-recipe');
  table.textContent = '';
  const rows = Object.entries(item.params || {}).filter(([, v]) => v !== '' && v !== null);
  if (item.seed !== null && item.seed !== undefined) rows.unshift(['seed', item.seed]);
  for (const [key, value] of rows) {
    const row = el('div', 'recipe-row');
    row.appendChild(el('span', 'k', key));
    row.appendChild(el('span', 'v', String(value)));
    table.appendChild(row);
  }
  if (item.notes) {
    const row = el('div', 'recipe-row wide');
    row.appendChild(el('span', 'k', 'the model rewrote the prompt to'));
    row.appendChild(el('span', 'v', item.notes));
    table.appendChild(row);
  }

  $('#shot-prev').disabled = shown <= 0;
  $('#shot-next').disabled = shown >= shots.length - 1;
}

function wireLightbox() {
  const dialog = $('#shot-dialog');
  $('#shot-prev').onclick = () => step(-1);
  $('#shot-next').onclick = () => step(1);
  $('#shot-close').onclick = () => closeShot();
  $('#shot-reuse').onclick = () => reuse(shots[shown]);
  $('#shot-download').onclick = () => download(shots[shown]);
  $('#shot-delete').onclick = () => remove(shots[shown]);
  $('#shot-copy').onclick = async () => {
    try {
      await navigator.clipboard.writeText(shots[shown].prompt || '');
      $('#shot-copy').textContent = 'Copied';
      setTimeout(() => { $('#shot-copy').textContent = 'Copy prompt'; }, 1200);
    } catch {
      // Clipboard access is refused outside a secure context, which a plain
      // http:// install on a LAN is. Saying so beats a button that does
      // nothing.
      status('the browser will not allow copying on an insecure origin', 'bad');
    }
  };

  dialog.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowLeft') step(-1);
    if (event.key === 'ArrowRight') step(1);
  });
  dialog.addEventListener('close', () => {
    shown = -1;
    $('#shot-stage').textContent = '';
    if (galleryStale) {
      galleryStale = false;
      loadGallery(true);
    }
  });
  // Clicking the backdrop, which is the dialog element itself outside its
  // own content box.
  dialog.addEventListener('click', (event) => {
    if (event.target === dialog) closeShot();
  });
}

/* ------------------------------------------------- importing a workflow */

/* ComfyUI takes a graph, not a prompt, so a video model here is a template
 * file. The README has always said importing one is a command rather than an
 * editing job; until now there was no way to do it from the page at all. */
async function importWorkflow(file) {
  let graph;
  try {
    graph = JSON.parse(await file.text());
  } catch {
    status('that is not JSON. Export it from ComfyUI with Save (API format).', 'bad');
    return;
  }
  const name = file.name.replace(/\.json$/i, '');
  const res = await post('/api/media/workflows', { name, graph });
  if (!res || !res.ok) {
    status((res && (await res.json().catch(() => ({}))).detail) || 'that workflow would not import', 'bad');
    return;
  }
  const result = await res.json();
  const tokens = result.tokens || [];
  // The token list is the only useful confirmation: it says which controls the
  // form will have, and whether the importer found the prompt at all.
  status(
    tokens.length
      ? `imported ${name} — controls: ${tokens.join(', ')}`
      : `imported ${name}, but no inputs were recognised, so it will only take what is hard-coded in the graph`,
    tokens.includes('prompt') ? 'good' : 'bad',
  );
  if (kind === 'video') describe();
}

/* --------------------------------------------------------------- wiring */

async function make(event) {
  event.preventDefault();
  const prompt = $('#studio-prompt').value.trim();
  const params = collect();
  const hasImage = schema.some((p) => p.kind === 'image' && params[p.name]);
  if (!prompt && !hasImage) {
    status('say what to make first', 'bad');
    return;
  }

  const body = {
    kind,
    prompt,
    provider: $('#studio-provider').value || null,
    model: $('#studio-model').value.trim() || null,
    params,
  };

  $('#studio-make').disabled = true;
  const res = await post('/api/media/jobs', body);
  $('#studio-make').disabled = false;

  if (!res) {
    status('the daemon is not reachable', 'bad');
    return;
  }
  if (!res.ok) {
    status((await res.json().catch(() => ({}))).detail || 'it would not start', 'bad');
    return;
  }

  const job = await res.json();
  status(kind === 'video' ? 'started — video takes minutes' : 'started');
  // Somewhere for the result to land, so the grid does not sit unchanged for
  // ninety seconds looking like nothing happened.
  skeletons(kind === 'image' ? Number(params.n) || 1 : 1);
  void job;
  watchJobs();
}

export function stopPolling() {
  if (jobTimer) clearInterval(jobTimer);
  jobTimer = null;
}

export async function openStudio() {
  await loadGallery(true);
  if (!schema.length) await describe();
  // A job started here and left running — by switching to Code mode, or by
  // reloading the page — is still running on the server, so the strip picks it
  // back up rather than pretending the work was lost with the view of it.
  const data = await json('/api/media/jobs');
  if (data && (data.jobs || []).some((j) => j.state === 'queued' || j.state === 'running')) watchJobs();
}

export function wireStudio(providers = []) {
  known.providers = providers.filter((p) => !p.modalities || p.modalities.includes(kind));
  if (!$('#studio-provider').options.length) paintProviders('');

  for (const button of document.querySelectorAll('.studio-kind .seg')) {
    button.onclick = () => {
      kind = button.dataset.kind;
      for (const other of document.querySelectorAll('.studio-kind .seg')) {
        other.classList.toggle('on', other === button);
      }
      // The model box is the one thing that must not survive: `sora-2` is not
      // a model the image side has, and carrying it across produces a
      // confusing failure rather than an empty box.
      $('#studio-model').value = '';
      describe();
    };
  }

  $('#studio-provider').onchange = () => {
    $('#studio-model').value = '';
    describe();
  };
  $('#studio-model').onchange = describe;

  $('#studio-form').onsubmit = make;
  $('#studio-advanced').onclick = function toggle() {
    const open = this.getAttribute('aria-expanded') !== 'true';
    this.setAttribute('aria-expanded', String(open));
    $('#studio-advanced-params').hidden = !open;
  };

  for (const button of document.querySelectorAll('.studio-filters .seg')) {
    button.onclick = () => {
      filter = button.dataset.filter;
      for (const other of document.querySelectorAll('.studio-filters .seg')) {
        other.classList.toggle('on', other === button);
      }
      loadGallery(true);
    };
  }

  let typing = null;
  $('#studio-search').oninput = (event) => {
    clearTimeout(typing);
    // Debounced: the store is walked per request, and a request per keystroke
    // over a library of a few thousand is a request per keystroke too many.
    typing = setTimeout(() => {
      search = event.target.value.trim();
      loadGallery(true);
    }, 250);
  };

  $('#studio-more').onclick = () => loadGallery(false);

  const workflow = $('#studio-workflow');
  $('#studio-import').onclick = () => workflow.click();
  workflow.onchange = () => {
    if (workflow.files[0]) importWorkflow(workflow.files[0]);
    workflow.value = '';
  };

  wireLightbox();

  // This runs again every time a connection is added or removed, and the form
  // on screen was built from what the old set of providers could do. Rebuilt
  // rather than left: connecting a diffusion server and finding the panel
  // still saying nothing is configured is the exact moment someone concludes
  // the connection did not work.
  if (!$('#pane-studio').hidden) describe();
}
