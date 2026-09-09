/* Roost's browser client: the shell, and the mode you type in.
 *
 * No framework and no build step: this is served from the machine it runs on,
 * so a toolchain would be a dependency with nothing to show for it.
 *
 * This file is the session list, the agent socket and the transcript — the
 * Code mode, which is the one everything else is measured against. The other
 * five modes are their own files and know nothing about this one; they share
 * the daemon, the companion and a handful of DOM helpers, and nothing else.
 * That is why adding Talk did not touch the transcript and adding the studio
 * did not touch the socket.
 *
 * The agent socket is attached to rather than owned — closing this page
 * pauses your view of a session, it does not stop it — so on connect we send
 * the last sequence number we saw and the server replays the gap.
 */

import { companion, caption } from './companions/index.js';
import { wireConnections } from './connections.js';
import { $, api, el, icon, onReachable, socket, took } from './dom.js';
import { endLive, isLive, wireLive } from './live.js';
import { go, mode, onEnter, onLeave, wireModes } from './modes.js';
import { focusSearch, wireSearch } from './search.js';
import { openStudio, stopPolling, wireStudio } from './studio.js';
import { startTalking, stopTalking, talking, wireTalk } from './talk.js';
import { Voice } from './voice.js';
import { narrate, refreshRuns, wireWatch } from './watch.js';

function ago(seconds) {
  if (seconds < 60) return 'now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`;
  return `${Math.floor(seconds / 86400)}d`;
}

const state = {
  sessionId: null,
  seq: 0,
  agent: null,
  voice: null,
  busy: false,
  turnNode: null,
  tools: new Map(),
  // Commands typed while the socket was down, replayed on reconnect.
  outbox: [],
  // Consecutive failed reconnects, for the backoff.
  retries: 0,
  // The locally echoed user bubble, removed when the server confirms the turn.
  echo: null,
  // What the open session is, for the chips under the composer.
  info: null,
  // When the running turn started, so the transcript can say how long it took.
  turnStart: 0,
  // Files this turn has written, and by how much, from the diffs the server
  // already sends. Emptied at the start of every turn.
  turnFiles: new Map(),
  // The same counted across the whole session, for the bar.
  added: 0,
  removed: 0,
};

/* ---------------------------------------------------------------- transcript */

const transcript = $('#transcript');

function atBottom() {
  return transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 80;
}

function append(node) {
  // Only follow the tail if the reader was already at it. Yanking someone back
  // down while they are reading earlier output is the most irritating thing a
  // streaming log can do.
  const follow = atBottom();
  transcript.appendChild(node);
  if (follow) transcript.scrollTop = transcript.scrollHeight;
  settle();
  return node;
}

/* An empty session is a different window: the companion in the middle, the
   question above the box, and a few things you might ask. The moment there is
   anything to read, all of that goes and the transcript takes the room. */
function settle() {
  const empty = !transcript.querySelector('.turn, .tool, .question, .notice, .changed, .worked');
  document.body.classList.toggle('blank', empty);
  $('#hero').hidden = !empty;
  $('#starters').hidden = !empty;
}

function clearTranscript() {
  for (const node of [...transcript.children]) {
    if (node.id !== 'hero') node.remove();
  }
  settle();
}

/* The greeting names the folder the agent is actually pointed at, because
   that is the one fact about a new session worth checking before you type. */
function dressHero() {
  const info = state.info;
  const where = info && info.root ? info.root.replace(/\/+$/, '').split('/').pop() : '';
  $('#hero-ask').textContent = where ? `What should we do in ${where}?` : 'What should we do?';
  $('#hero-sub').textContent = info ? [info.model, info.policy].filter(Boolean).join('  ·  ') : '';
}

const STARTERS = [
  'Have a look around this folder and tell me what it is',
  'Find the tests and run them',
  'Explain what the most recent change does',
  'Look something up on the web for me',
];

function buildStarters() {
  const box = $('#starters');
  box.textContent = '';
  for (const text of STARTERS) {
    const row = el('button', 'starter');
    row.type = 'button';
    row.appendChild(icon('chat'));
    row.appendChild(el('span', '', text));
    // Filled in rather than sent. A suggestion you cannot edit before it runs
    // is a button that does something to your machine on one click.
    row.onclick = () => {
      const input = $('#input');
      input.value = text;
      input.focus();
      input.dispatchEvent(new Event('input'));
    };
    box.appendChild(row);
  }
}

function notice(text, kind = '') {
  // The same message twice in a row is collapsed into a count. Even with the
  // retry loop fixed, anything that can repeat should not be able to bury the
  // transcript under copies of itself.
  const last = transcript.lastElementChild;
  if (last && last.classList.contains('notice') && last.dataset.text === text) {
    const n = Number(last.dataset.count || 1) + 1;
    last.dataset.count = String(n);
    last.textContent = `${text}  (×${n})`;
    return last;
  }
  const node = el('div', `notice ${kind}`, text);
  node.dataset.text = text;
  return append(node);
}

function userTurn(text) {
  const wrap = el('div', 'turn user');
  wrap.appendChild(el('div', 'body', text));
  return append(wrap);
}

function assistantTurn() {
  const wrap = el('div', 'turn assistant');
  wrap.appendChild(el('div', 'body'));
  state.turnNode = wrap;
  return append(wrap);
}

function appendText(text) {
  if (!state.turnNode) assistantTurn();
  const body = state.turnNode.querySelector('.body');
  body.textContent += text;
  if (atBottom()) transcript.scrollTop = transcript.scrollHeight;
}

function appendThinking(text) {
  if (!state.turnNode) assistantTurn();
  let box = state.turnNode.querySelector('.thinking');
  if (!box) {
    box = el('details', 'thinking');
    box.appendChild(el('summary', '', 'reasoning'));
    box.appendChild(el('span'));
    state.turnNode.insertBefore(box, state.turnNode.firstChild);
  }
  box.querySelector('span').textContent += text;
}

function renderScreenshot(display) {
  const box = el('div', 'shot');
  const img = document.createElement('img');
  img.src = `data:${display.media_type || 'image/png'};base64,${display.image}`;
  img.alt = 'what the agent saw';
  img.loading = 'lazy';
  box.appendChild(img);

  // Click through to the full-size frame: the thumbnail is for scanning the
  // transcript, and a 2560px screenshot scaled into a card is unreadable.
  img.onclick = () => window.open(img.src, '_blank');

  if (display.width) {
    box.dataset.w = display.width;
    box.dataset.h = display.height;
  }

  const meta = [];
  if (display.width) meta.push(`${display.width}x${display.height}`);
  if (display.url) meta.push(display.url);
  if (meta.length) box.appendChild(el('div', 'shot-meta', meta.join('  ·  ')));
  return box;
}

function markClick(card, x, y, label) {
  /* Draw the click point onto the screenshot above it.
   *
   * The screenshot was taken before the click, so the marker goes on the
   * previous card rather than this one — which is exactly what someone
   * reviewing the run wants to see: the frame, and where it was about to go. */
  const shots = transcript.querySelectorAll('.tool .shot img');
  const last = shots[shots.length - 1];
  if (!last) return;

  const wrap = last.parentElement;
  wrap.style.position = 'relative';
  const dot = el('div', 'click-marker');
  // Percentages, so the marker stays put when the image is scaled to the card.
  const w = Number(wrap.dataset.w || last.naturalWidth || 0);
  const h = Number(wrap.dataset.h || last.naturalHeight || 0);
  if (!w || !h) return;
  dot.style.left = `${(x / w) * 100}%`;
  dot.style.top = `${(y / h) * 100}%`;
  dot.title = label || `${x}, ${y}`;
  wrap.appendChild(dot);
}

function renderDiff(text) {
  const box = el('div', 'diff');
  for (const line of text.split('\n')) {
    let cls = '';
    if (line.startsWith('+') && !line.startsWith('+++')) cls = 'add';
    else if (line.startsWith('-') && !line.startsWith('---')) cls = 'del';
    else if (line.startsWith('@@')) cls = 'hunk';
    const span = el('span', cls, line + '\n');
    box.appendChild(span);
  }
  return box;
}

/* --------------------------------------------------------------- tool cards */

/* What the agent did, said as a person would say it. The tool name is exact
   and useless at a glance; "Ran" followed by the command is what someone
   skimming a transcript is actually reading for. Anything not listed falls
   back to its own name in monospace, which is honest about being a tool. */
const VERBS = {
  read_file: 'Read',
  read_files: 'Read',
  outline: 'Sketched',
  list_dir: 'Listed',
  glob: 'Found',
  grep: 'Searched',
  recall: 'Recalled',
  write_file: 'Wrote',
  edit_file: 'Edited',
  multi_edit: 'Edited',
  apply_patch: 'Patched',
  plan: 'Planned',
  remember: 'Remembered',
  shell: 'Ran',
  web_search: 'Searched the web for',
  web_fetch: 'Fetched',
  research: 'Looked into',
  browser_navigate: 'Opened',
  browser_read: 'Read the page',
  browser_click: 'Clicked',
  browser_type: 'Typed into',
  browser_screenshot: 'Looked at',
  browser_hand_over: 'Handed you',
  desktop_screenshot: 'Looked at the screen',
  desktop_click: 'Clicked',
  desktop_type: 'Typed',
  desktop_key: 'Pressed',
  desktop_scroll: 'Scrolled',
  system_info: 'Checked the machine',
  package_search: 'Looked for',
  package_install: 'Installed',
  package_remove: 'Removed',
  display_info: 'Checked the displays',
  display_hdr: 'Changed HDR for',
  media_params: 'Checked what it can tune',
  generate_image: 'Generated',
  generate_video: 'Started rendering',
  media_job: 'Checked on the render',
  import_workflow: 'Installed the workflow',
  ask_user: 'Asked you',
};

/* A finished, unremarkable call is one grey line; a card is for what wants
   looking at. This decides which, and it is called again when the call ends,
   because "unremarkable" is not knowable until then. */
function setCard(card, on) {
  card.classList.toggle('card', on);
  card.classList.toggle('open', on);
}

function toolCard(call, needsApproval) {
  const card = el('div', 'tool');
  const head = el('div', 'head');

  const verb = VERBS[call.name];
  if (verb) head.appendChild(el('span', 'verb', verb));
  else head.appendChild(el('span', 'name', call.name));
  head.appendChild(el('span', 'summary', call.summary || ''));
  head.appendChild(el('span', 'took'));
  head.appendChild(el('span', `risk ${call.risk}`, call.risk));
  head.appendChild(el('span', 'chev', '›'));

  // The whole head is the disclosure control: a chevron you have to hit
  // exactly is a worse target than the line it sits on.
  head.onclick = () => {
    const open = card.classList.toggle('open');
    if (open) card.classList.add('card');
    else card.classList.toggle('card', card.classList.contains('failed') || Boolean(card.querySelector('.decide')));
  };
  card.appendChild(head);
  append(card);
  state.tools.set(call.id, card);

  if (needsApproval) {
    setCard(card, true);
    const bar = el('div', 'decide');

    // The companion leans in where the decision is. It is the same creature
    // in the same state as the one on the perch — it just sits next to the
    // thing it is stuck on, so the reason it stopped is where you are looking.
    const who = el('span', 'who');
    bar.appendChild(who);
    const pet = companion.attach(who, { scale: 2 });

    bar.appendChild(el('span', 'why', 'Waiting for you.'));

    const remember = el('label');
    const box = el('input');
    box.type = 'checkbox';
    remember.appendChild(box);
    remember.appendChild(el('span', '', "don't ask for this exact call again"));
    // The server refuses to remember a destructive call. Say so here rather
    // than offering a checkbox that silently does nothing.
    if (call.risk === 'destructive') {
      box.disabled = true;
      remember.lastChild.textContent = 'destructive calls are always confirmed';
    }
    bar.appendChild(remember);

    const deny = el('button', 'ghost small', 'Deny');
    const allow = el('button', 'primary small', 'Allow');
    const decided = () => {
      pet.remove();
      bar.remove();
    };
    deny.onclick = () => {
      send({ type: 'tool.deny', call_id: call.id, reason: 'declined in the UI' });
      decided();
      companion.set('thinking');
    };
    allow.onclick = () => {
      send({ type: 'tool.approve', call_id: call.id, remember: box.checked });
      decided();
      // Approval is the only signal that this call is now running: the server
      // sends no separate "started" for a call it was already told about.
      companion.tool(call.name);
    };
    bar.appendChild(deny);
    bar.appendChild(allow);
    card.appendChild(bar);
    allow.focus();
  }
  return card;
}

function toolOutput(callId, text, stream) {
  const card = state.tools.get(callId);
  if (!card) return;
  let out = card.querySelector('.out.live');
  if (!out) {
    out = el('div', 'out live');
    card.appendChild(out);
    // Output arriving now is worth watching now — a build scrolling past is
    // the one thing in here nobody wants folded away.
    setCard(card, true);
  }
  if (stream === 'stderr') out.classList.add('err');
  out.textContent += text;
  // A long build must not push everything else off screen.
  if (out.textContent.length > 40000) out.textContent = out.textContent.slice(-40000);
  out.scrollTop = out.scrollHeight;
}

/* How much of a file changed, counted off the unified diff the server already
   sends. It is the same number `git diff --stat` would print, and it is free:
   the diff is on screen either way. */
function countDiff(text) {
  let add = 0;
  let del = 0;
  for (const line of text.split('\n')) {
    if (line.startsWith('+') && !line.startsWith('+++')) add++;
    else if (line.startsWith('-') && !line.startsWith('---')) del++;
  }
  return { add, del };
}

function noteChange(path, diff) {
  const counts = countDiff(diff);
  const at = state.turnFiles.get(path) || { add: 0, del: 0 };
  at.add += counts.add;
  at.del += counts.del;
  state.turnFiles.set(path, at);
  state.added += counts.add;
  state.removed += counts.del;
  showDiffstat();
}

function showDiffstat() {
  const box = $('#diffstat');
  box.textContent = '';
  if (!state.added && !state.removed) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  box.appendChild(el('span', 'add', `+${state.added}`));
  box.appendChild(document.createTextNode(' '));
  box.appendChild(el('span', 'del', `-${state.removed}`));
  box.title = 'what the agent has changed in this session';
}

function toolDone(result) {
  const card = state.tools.get(result.id);
  if (!card) return;
  if (!result.ok) card.classList.add('failed');
  // A finished call must not still be offering Allow and Deny. This shows up
  // on replay, where the proposal is re-rendered long after it was decided.
  card.querySelector('.decide')?.remove();

  const live = card.querySelector('.out.live');
  if (live) live.classList.remove('live');

  // What the agent saw. This is the whole of "watch it work": every frame it
  // acted on, in order, with the point it clicked marked on the frame before
  // the click. A picture of the screen is worth far more here than the
  // sentence the model wrote about it.
  const shot = result.display && result.display.image;
  if (shot) {
    card.appendChild(renderScreenshot(result.display));
  }

  const diff = result.display && result.display.diff;
  if (diff) {
    card.appendChild(renderDiff(diff));
    if (result.ok && result.display.path) noteChange(result.display.path, diff);
  } else if (!live && !shot && result.content) {
    const out = el('div', result.ok ? 'out' : 'out err');
    out.textContent = result.content.length > 4000 ? result.content.slice(0, 4000) + '\n…' : result.content;
    card.appendChild(out);
  }

  if (result.name === 'desktop_click' && result.display && result.ok) {
    markClick(card, result.display.x, result.display.y, result.display.label);
  }

  // Anything that failed, changed a file, or has a picture in it stays a card
  // and stays open. A successful read collapses to its line: it is in the
  // transcript so you can check, not so you have to.
  setCard(card, !result.ok || Boolean(shot || diff));

  const when = card.querySelector('.took');
  if (when && result.duration_ms > 400) when.textContent = took(result.duration_ms);
}

/* ------------------------------------------------------------ what changed */

/* The receipt at the end of a turn: which files it touched, by how much, and
   one button that puts them back. The list is built from the diffs that came
   past during the turn, so it says exactly what this client watched happen. */
function changedCard() {
  if (!state.turnFiles.size) return;
  const files = [...state.turnFiles.entries()];
  state.turnFiles = new Map();

  const box = el('div', 'changed');
  const top = el('div', 'top');
  top.appendChild(el('span', 'what', `${files.length} file${files.length === 1 ? '' : 's'} changed`));

  const undo = el('button', 'ghost small undo');
  undo.appendChild(icon('undo'));
  undo.appendChild(el('span', '', 'Undo'));
  undo.onclick = () => undoLast(undo);
  top.appendChild(undo);
  box.appendChild(top);

  for (const [path, counts] of files) {
    const row = el('div', 'file');
    const short = path.replace(/^.*\/([^/]+\/[^/]+)$/, '$1');
    const name = el('span', 'path', short);
    name.title = path;
    row.appendChild(name);
    if (counts.add) row.appendChild(el('span', 'add', `+${counts.add}`));
    if (counts.del) row.appendChild(el('span', 'del', `-${counts.del}`));
    box.appendChild(row);
  }
  append(box);
}

/* Undo means the newest checkpoint, which is the one this turn made. Anything
   more selective than that is what the Rewind dialog is for — and rewinding
   to the middle of a session is exactly what the checkpoint code refuses to
   do, because it would leave a state that never existed. */
async function undoLast(button) {
  const res = await api(`/api/sessions/${state.sessionId}/checkpoints`);
  if (!res || !res.ok) return;
  const { enabled, checkpoints } = await res.json();
  if (!enabled || !checkpoints.length) {
    notice('There is no checkpoint to undo to — checkpoints are off for this session.', 'error');
    return;
  }
  const cp = checkpoints[0];
  if (!confirm(`Put files back to before "${cp.label}"?\n\nThis undoes that turn and every one after it.`)) return;
  button.disabled = true;
  await restore(cp.id);
  button.disabled = false;
}

function questionCard(ev) {
  const card = el('div', 'question');
  card.appendChild(el('div', 'q', ev.question));

  const answer = (text) => {
    send({ type: 'question.answer', question_id: ev.question_id, answer: text });
    companion.set('thinking');
    card.querySelector('.opts')?.remove();
    card.querySelector('form')?.remove();
    card.appendChild(el('div', 'notice', `You answered: ${text}`));
  };

  if (ev.options && ev.options.length) {
    const opts = el('div', 'opts');
    for (const option of ev.options) {
      const button = el('button', 'ghost small', option);
      button.onclick = () => answer(option);
      opts.appendChild(button);
    }
    card.appendChild(opts);
  }

  const form = el('form');
  const input = el('input');
  input.placeholder = 'or write your own answer';
  const go = el('button', 'primary small', 'Answer');
  form.onsubmit = (e) => {
    e.preventDefault();
    if (input.value.trim()) answer(input.value.trim());
  };
  form.appendChild(input);
  form.appendChild(go);
  card.appendChild(form);
  append(card);
  input.focus();
}

/* ------------------------------------------------------------ agent socket */

function send(command) {
  if (state.agent && state.agent.readyState === WebSocket.OPEN) {
    state.agent.send(JSON.stringify(command));
    return;
  }
  // The socket reconnects on its own, so a command sent during the gap is
  // held rather than dropped. Losing what someone typed — silently, because
  // the UI had already echoed it — is the worst failure this client has.
  state.outbox.push(command);
  notice('Not connected — held until the connection is back.');
}

function flushOutbox() {
  if (!state.agent || state.agent.readyState !== WebSocket.OPEN) return;
  const held = state.outbox.splice(0);
  for (const command of held) state.agent.send(JSON.stringify(command));
  if (held.length) notice(`Reconnected — sent ${held.length} held message${held.length === 1 ? '' : 's'}.`);
}

function setBusy(busy) {
  state.busy = busy;
  // The send button becomes the stop button while a turn is running: one
  // place to look, and no way to queue a second turn into a busy session.
  $('#send').disabled = busy || !state.sessionId;
  $('#send').hidden = busy;
  $('#stop').hidden = !busy;
}

function setLink(on) {
  const pill = $('#link');
  pill.textContent = on ? 'live' : 'offline';
  pill.className = `pill ${on ? 'on' : 'off'}`;
}

// Close codes the server uses for "this will never work". Retrying any of
// them is how one dead session id becomes an unbounded stream of identical
// errors — which is exactly what happened when the daemon was restarted under
// a page that still remembered a session from the previous one.
const FATAL_CLOSE = new Set([4400, 4401, 4404]);

function connectAgent(sessionId) {
  if (state.agent) {
    state.agent.onclose = null;
    state.agent.close();
  }
  state.sessionId = sessionId;

  // Resume rather than replay everything: on a reconnect we only want the gap.
  const ws = new WebSocket(socket('/ws/agent', { session: sessionId, since: state.seq }));
  state.agent = ws;

  ws.onopen = () => {
    setLink(true);
    state.retries = 0;
    // Whatever arrives next is the replay of what was missed, not news.
    companion.resumed();
    $('#input').disabled = false;
    $('#send').disabled = false;
    $('#voice-toggle').disabled = false;
    $('#rewind').hidden = false;
    $('#nav-rewind').disabled = false;
    flushOutbox();
  };

  ws.onmessage = (msg) => handleAgentEvent(JSON.parse(msg.data));

  ws.onclose = (event) => {
    setLink(false);
    if (state.sessionId !== sessionId) return;      // we moved on deliberately

    if (FATAL_CLOSE.has(event.code)) {
      // Nothing to retry. The commonest cause by far is the daemon having been
      // restarted while this page kept the old session id, so rather than
      // stopping with an error, forget it and attach to whatever is there now.
      forgetSession();
      state.sessionId = null;
      state.retries = 0;
      loadSessions();
      return;
    }

    // Genuinely transient — a laptop lid, a daemon restarting. Backed off
    // rather than hammered, and capped so it keeps trying all day at a
    // sensible rate.
    state.retries = (state.retries || 0) + 1;
    const wait = Math.min(1000 * 2 ** (state.retries - 1), 20000);
    setTimeout(() => {
      if (state.sessionId === sessionId) connectAgent(sessionId);
    }, wait);
  };
}

function handleAgentEvent(ev) {
  // Autopilot renders the same events as one line each, so it is fed here
  // rather than opening a second socket onto the same session — two readers
  // of one event stream is two places for the sequence number to drift.
  narrate(ev);

  switch (ev.type) {
    case 'session.started':
      // The promise, spelled out, once: what this session may do without
      // asking. The chips under the composer carry the short forms of the
      // same facts, so repeating the model here would only crowd the title.
      $('#session-meta').textContent = ev.policy;
      $('#mode-wrap').title = ev.policy;
      dressChips({ model: ev.model, root: ev.cwd });
      break;

    case 'policy.changed':
      $('#mode').value = ev.mode;
      dressChips({ policy: ev.mode });
      notice(`Approval is now: ${ev.policy || ev.mode}.`);
      break;

    case 'turn.started':
      state.turnNode = null;
      // The server's clock, not this one: a reattached client replays turns
      // that ended hours ago, and timing them against `now` would report
      // every one of them as having taken a millisecond.
      state.turnStart = ev.at || 0;
      state.turnFiles = new Map();
      setBusy(true);
      companion.set('thinking');
      // Rendered from the event rather than on submit, so a live client and a
      // reattached one show the same conversation. The local echo is removed
      // first to avoid showing it twice.
      if (ev.text) {
        const echoed = state.echo && state.echo.querySelector('.body');
        if (echoed && echoed.textContent === ev.text) state.echo.remove();
        state.echo = null;
        userTurn(ev.text);
      }
      break;

    case 'text.delta':
      appendText(ev.text);
      break;

    case 'thinking.delta':
      appendThinking(ev.text);
      break;

    case 'tool.proposed':
      state.turnNode = null;
      toolCard(ev.call, ev.needs_approval);
      // What it is about to do, or the fact that it cannot do it without you.
      if (ev.needs_approval) companion.set('waiting');
      else companion.tool(ev.call.name);
      break;

    case 'tool.output.delta':
      toolOutput(ev.call_id, ev.text, ev.stream);
      break;

    case 'tool.completed':
      toolDone(ev.result);
      if (ev.result.ok) companion.set('thinking');
      else companion.flash('error');
      break;

    case 'tool.denied': {
      const card = state.tools.get(ev.call_id);
      if (card) {
        card.classList.add('denied');
        card.querySelector('.decide')?.remove();
        card.appendChild(el('div', 'out', `Not run: ${ev.reason}`));
      }
      companion.set('thinking');
      break;
    }

    case 'question.asked':
      state.turnNode = null;
      questionCard(ev);
      companion.set('waiting');
      break;

    case 'turn.completed': {
      setBusy(false);
      state.turnNode = null;
      companion.set('idle');
      // How long that took, and what it cost you in files. Both are things
      // you would otherwise have to reconstruct by scrolling. Said only when
      // both ends of it are known — a made-up duration is worse than none.
      if (state.turnStart && ev.at) {
        append(el('div', 'worked', `Worked for ${took((ev.at - state.turnStart) * 1000)}`));
      }
      state.turnStart = 0;
      changedCard();
      if (ev.stop_reason === 'interrupted') notice('Interrupted.');
      else if (ev.stop_reason === 'max_steps') notice('Stopped: too many steps.', 'error');
      else if (ev.stop_reason === 'error') companion.flash('error');
      else companion.flash('success');
      break;
    }

    case 'error':
      notice(ev.message, 'error');
      setBusy(false);
      companion.set('idle');
      companion.flash('error');
      break;

    case 'session.ended':
      notice(`Session ended: ${ev.reason}`);
      setBusy(false);
      companion.set('idle');
      break;

    case 'pong':
      return; // no sequence of its own
  }

  // The server stamps each event with its sequence number; remembering the
  // highest is what lets a reconnect ask for the gap instead of the lot.
  if (typeof ev.seq === 'number' && ev.seq > state.seq) state.seq = ev.seq;
  refreshSessionsSoon();
}

/* ------------------------------------------------------------------ voice */

/* The composer's microphone button opens the same call Talk mode uses; the
   difference is only the surface around it. See voice.js — the call itself
   lives there precisely so that these two cannot drift apart on the delicate
   part, which is barge-in. */

function setVoiceState(text) {
  $('#voice-state').textContent = text;
}

function startStripVoice() {
  const call = new Voice({
    ready: (ev) => {
      $('#voice-strip').hidden = false;
      setVoiceState(`${ev.stt} → ${ev.llm} → ${ev.tts}`);
    },
    listening: () => companion.set('listening'),
    thinking: () => companion.set('thinking'),
    partial: (text) => { $('#voice-partial').textContent = text; },
    heard: (text, submitted) => {
      $('#voice-partial').textContent = '';
      if (submitted) userTurn(text);
      else notice(`Heard "${text}" — too short to answer.`);
    },
    said: (delta) => appendText(delta),
    speaking: () => {
      state.turnNode = null;
      companion.set('speaking');
    },
    cancelled: () => {
      notice('— interrupted');
      companion.set('listening');
    },
    finished: () => companion.set('idle'),
    error: (message) => notice(message, 'error'),
    stopped: () => {
      state.voice = null;
      companion.set('idle');
      $('#voice-strip').hidden = true;
      $('#voice-toggle').classList.remove('on');
      $('#voice-toggle').title = 'Start voice';
    },
  });
  return call;
}

/* --------------------------------------------------------------- requests */

/* `api` itself is in dom.js, because every mode needs it and the knowledge
   that the daemon has gone must exist in exactly one place — two copies means
   one of them showing "live" while the other has been failing for a minute. */
onReachable((ok) => {
  if (ok) return;
  const pill = $('#link');
  pill.textContent = 'no daemon';
  pill.className = 'pill off';
});

/* ------------------------------------------------------------------ rewind */

async function loadRewind() {
  const list = $('#rewind-list');
  list.textContent = '';
  const res = await api(`/api/sessions/${state.sessionId}/checkpoints`);
  if (!res || !res.ok) return;
  const { enabled, checkpoints } = await res.json();
  if (!enabled) return;

  if (!checkpoints.length) {
    const li = el('li', 'empty', 'Nothing to undo — no files have been changed yet.');
    list.appendChild(li);
    return;
  }

  for (const cp of checkpoints) {
    const li = el('li');
    const what = el('div', 'what');
    what.appendChild(el('span', 'label', cp.label || `turn ${cp.turn_id}`));
    const names = cp.paths.map((p) => p.split('/').pop()).join(', ');
    what.appendChild(el('div', 'files', `${cp.files} file${cp.files === 1 ? '' : 's'} · ${names}`));
    li.appendChild(what);

    const go = el('button', 'ghost small', 'Undo to here');
    go.onclick = async () => {
      // Named plainly rather than as "restore": what it does is throw away
      // work, and the button should say so before it is pressed.
      if (!confirm(`Put files back to before "${cp.label}"?\n\nThis undoes that turn and every one after it.`)) return;
      if (await restore(cp.id)) loadRewind();
    };
    li.appendChild(go);
    list.appendChild(li);
  }
}

/* Shared by the Rewind dialog and by the Undo button on a turn's receipt:
   both throw the same work away, so both had better report it the same way. */
async function restore(checkpointId) {
  const r = await api(`/api/sessions/${state.sessionId}/restore`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ checkpoint: checkpointId }),
  });
  if (!r) {
    notice('The daemon is not reachable.', 'error');
    return false;
  }
  if (!r.ok) {
    notice((await r.json()).detail || 'could not rewind', 'error');
    return false;
  }
  const report = await r.json();
  const bits = [];
  if (report.restored.length) bits.push(`${report.restored.length} restored`);
  if (report.deleted.length) bits.push(`${report.deleted.length} deleted`);
  if (report.skipped && Object.keys(report.skipped).length) {
    bits.push(`${Object.keys(report.skipped).length} skipped`);
  }
  notice(`Rewound: ${bits.join(', ') || 'nothing to change'}.`);
  // Worth saying out loud: those files had been edited by hand since, and
  // that work has just been overwritten.
  if (report.changed_since && report.changed_since.length) {
    notice(
      `${report.changed_since.length} file(s) had been changed since the agent wrote them, ` +
      `and were overwritten: ${report.changed_since.join(', ')}`,
      'error',
    );
  }
  return true;
}

function wireRewind() {
  const open = () => { loadRewind(); $('#rewind-dialog').showModal(); };
  $('#rewind').onclick = open;
  $('#nav-rewind').onclick = open;
  $('#rewind-close').onclick = () => $('#rewind-dialog').close();
}

/* ------------------------------------------------------------------ memory */

/* Memory is off on some installs entirely, and off per person on the rest.
   The button is only shown when the endpoints exist, so nothing here implies
   a feature that is not there. */

async function memorySettings() {
  const res = await api('/api/memory/settings');
  if (!res || res.status === 404) return null;   // unreachable, or not enabled here
  return res.ok ? res.json() : null;
}

async function loadMemory() {
  const settings = await memorySettings();
  if (!settings) return;

  $('#mem-enabled').checked = settings.enabled;
  $('#mem-auto').checked = settings.auto_capture;
  // Automatic capture is meaningless without the master switch, and offering
  // it as though it were independent would misrepresent what it does.
  $('#mem-auto').disabled = !settings.enabled;

  const list = $('#mem-list');
  list.textContent = '';

  if (!settings.enabled) {
    // Switching memory off stops it storing and stops it recalling; it does
    // not delete what is already there. Saying only "nothing is being stored"
    // would read as "your data is gone", which would be untrue.
    $('#mem-count').textContent = settings.count
      ? `Nothing new is being stored, and nothing is being recalled. ${settings.count} earlier `
        + `memor${settings.count === 1 ? 'y is' : 'ies are'} still kept — use Forget everything to delete them.`
      : 'Nothing is being stored, and nothing is kept.';
    return;
  }

  const res = await api('/api/memory');
  if (!res || !res.ok) return;
  const { memories } = await res.json();
  $('#mem-count').textContent = memories.length
    ? `${memories.length} remembered.`
    : 'Nothing remembered yet.';

  for (const m of memories) {
    const li = el('li');
    const text = el('div', 'text');
    text.appendChild(el('div', '', m.text));
    const when = new Date(m.created_at * 1000).toLocaleDateString();
    text.appendChild(el('div', 'meta', `${m.kind} · ${when}${m.hits ? ` · used ${m.hits}×` : ''}`));
    li.appendChild(text);

    const forget = el('button', 'ghost forget', 'Forget');
    forget.onclick = async () => {
      await api(`/api/memory/${m.id}`, { method: 'DELETE' });
      loadMemory();
    };
    li.appendChild(forget);
    list.appendChild(li);
  }
}

async function saveMemorySettings() {
  await api('/api/memory/settings', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      enabled: $('#mem-enabled').checked,
      auto_capture: $('#mem-auto').checked,
    }),
  });
  loadMemory();
}

function wireMemory() {
  $('#mem-enabled').onchange = saveMemorySettings;
  $('#mem-auto').onchange = saveMemorySettings;

  $('#open-memory').onclick = () => {
    loadMemory();
    $('#memory-dialog').showModal();
  };
  $('#mem-close').onclick = () => $('#memory-dialog').close();

  $('#mem-wipe').onclick = async () => {
    // Irreversible and it also switches memory back off, so it is worth one
    // question — but only one.
    if (!confirm('Delete everything Roost remembers about you? This cannot be undone.')) return;
    await api('/api/memory', { method: 'DELETE' });
    loadMemory();
  };

  memorySettings().then((settings) => {
    if (settings) $('#open-memory').hidden = false;
  });
}

/* ------------------------------------------------------------------- shell */

let refreshTimer = null;
function refreshSessionsSoon() {
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(() => loadSessions().catch(() => {}), 700);
}

async function loadSessions() {
  const res = await api('/api/sessions');
  if (!res || !res.ok) return;
  const { sessions } = await res.json();

  // The poll knows the truth sooner than the socket does. If the session we
  // are holding is not in the list, it is gone — the daemon was restarted —
  // and waiting for the websocket to work that out means sitting in backoff
  // for up to twenty seconds on an id that will never resolve.
  if (state.sessionId && !sessions.some((s) => s.id === state.sessionId)) {
    forgetSession();
    state.sessionId = null;
    state.retries = 0;
    if (state.agent) {
      state.agent.onclose = null;      // do not let it schedule another retry
      state.agent.close();
      state.agent = null;
    }
  }

  // On a fresh page, reattach to whatever was open. The session outlived the
  // tab; the point of that is undermined if reopening starts from nothing.
  if (!state.sessionId && sessions.length) {
    const wanted = lastSession();
    const resume = sessions.find((s) => s.id === wanted) || sessions[0];
    if (resume) openSession(resume);
  }

  // Grouped by working root. A root is the only thing a session is really
  // *about* — everything else about it is history — so it is what the list
  // sorts itself under, and one machine running three projects reads as
  // three projects rather than as nine chats.
  const list = $('#sessions');
  list.textContent = '';

  const roots = new Map();
  for (const s of sessions) {
    if (!roots.has(s.root)) roots.set(s.root, []);
    roots.get(s.root).push(s);
  }

  for (const [root, group] of roots) {
    const head = el('li', 'group');
    head.appendChild(icon('folder'));
    head.appendChild(el('span', '', basename(root)));
    head.title = root;
    list.appendChild(head);

    for (const s of group) {
      const li = el('li');
      li.setAttribute('aria-current', String(s.id === state.sessionId));

      let status = `${s.turns} turn${s.turns === 1 ? '' : 's'}`;
      const dot = el('span', 'state');
      if (s.busy) {
        dot.classList.add('busy');
        status = 'working';
      }
      if (s.waiting_on) {
        dot.classList.add('waiting');
        status = s.waiting_on === 'approval' ? 'needs approval' : 'asked you something';
      }
      // Colour is never the only carrier: the same thing is in the row's
      // label, which is what a screen reader and a hover both get.
      li.appendChild(dot);
      li.appendChild(el('span', 'name', s.title || s.id));
      li.appendChild(el('span', 'when', s.busy ? 'now' : ago(s.idle_for)));
      li.title = `${status} · ${s.model}`;
      li.setAttribute('aria-label', `${s.title || s.id} — ${status}`);

      li.onclick = () => openSession(s);
      list.appendChild(li);
    }
  }
}

function basename(path) {
  return String(path || '').replace(/\/+$/, '').split('/').pop() || String(path || '');
}

/* Put a session in the window: its name in the bar, its facts in the chips
   under the composer, and its socket attached. */
function openSession(summary) {
  $('#session-title').textContent = summary.title || summary.id;
  dressChips(summary);
  selectSession(summary.id);
}

/* What the next thing you say will be run under: which folder, which model,
   what it is allowed to do. Kept beside the box you type in, because that is
   what all three of them qualify. */
function dressChips(patch) {
  state.info = { ...(state.info || {}), ...patch };
  const info = state.info;

  $('#mode-wrap').hidden = false;
  if (info.policy) $('#mode').value = info.policy;

  const root = $('#root-chip');
  if (info.root) {
    root.hidden = false;
    root.querySelector('span').textContent = basename(info.root);
    root.title = info.root;
  }

  const model = $('#model-chip');
  if (info.model) {
    model.hidden = false;
    model.textContent = info.model;
    model.title = `answering with ${info.model}`;
  }
  dressHero();
}

async function loadProviders() {
  const res = await api('/api/providers');
  if (!res || !res.ok) return;
  const data = await res.json();
  const box = $('#providers');
  box.textContent = '';

  for (const [modality, ids] of Object.entries(data.capabilities || {})) {
    const row = el('div', 'row');
    row.appendChild(el('span', '', modality));
    const provider = ids[0];
    const info = (data.providers || []).find((p) => p.id === provider);
    row.appendChild(el('span', info && info.local ? 'local' : '', provider));
    box.appendChild(row);
  }
  if (!Object.keys(data.capabilities || {}).length) {
    box.appendChild(el('div', 'row', 'no providers configured'));
  }

  // Which provider is answering, next to which model is answering. The pair
  // is one fact, and it belongs under the composer with the rest of them.
  const chat = (data.capabilities || {}).chat || [];
  const chip = $('#provider-chip');
  if (chat.length) {
    const info = (data.providers || []).find((p) => p.id === chat[0]);
    chip.hidden = false;
    chip.textContent = chat[0];
    chip.title = info && info.local ? 'running on this machine' : 'a remote provider';
    chip.classList.toggle('local', Boolean(info && info.local));
  }

  const select = $('#new-dialog select[name=provider]');
  // Rebuilt rather than appended to: this runs again whenever a connection is
  // added, and appending would give three copies of Groq after three edits.
  select.textContent = '';
  const auto = el('option', '', 'first that can do chat');
  auto.value = '';
  select.appendChild(auto);
  for (const p of data.providers || []) {
    if (!p.modalities.includes('chat')) continue;
    const option = el('option', '', `${p.label}${p.local ? ' (local)' : ''}`);
    option.value = p.id;
    select.appendChild(option);
  }

  // The studio's provider list is the subset that can actually make something.
  wireStudio((data.providers || []).filter(
    (p) => p.modalities.includes('image') || p.modalities.includes('video'),
  ));
}

const LAST_SESSION = 'roost.session';

function rememberSession(id) {
  // Wrapped because a browser with site data blocked throws on access rather
  // than returning null, and losing the page over a convenience is silly.
  try { localStorage.setItem(LAST_SESSION, id); } catch { /* private window */ }
}

function forgetSession() {
  try { localStorage.removeItem(LAST_SESSION); } catch { /* private window */ }
}

function lastSession() {
  try { return localStorage.getItem(LAST_SESSION); } catch { return null; }
}

function selectSession(id) {
  if (id === state.sessionId) return;
  clearTranscript();
  state.tools.clear();
  state.seq = 0;
  state.turnNode = null;
  state.turnStart = 0;
  state.turnFiles = new Map();
  // The diff stat counts what this client watched happen in one session, so
  // moving to another one starts it again rather than carrying a total over.
  state.added = 0;
  state.removed = 0;
  showDiffstat();
  // Held commands belong to the session they were typed for, not the next one.
  state.outbox = [];
  rememberSession(id);
  connectAgent(id);
  loadSessions();
}

async function createSession(form) {
  const body = {
    root: form.root.value.trim() || null,
    model: form.model.value.trim() || null,
    provider: form.provider.value || null,
    mode: form.mode.value,
    tools: form.tools.value ? form.tools.value.split(',') : [],
  };
  const res = await api('/api/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res) {
    notice('The daemon is not reachable.', 'error');
    return;
  }
  if (!res.ok) {
    notice((await res.json()).detail || 'could not create the session', 'error');
    return;
  }
  const session = await res.json();
  openSession({ ...session, policy: body.mode });
}

function wire() {
  const input = $('#input');

  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, window.innerHeight * 0.4) + 'px';
    companion.typing();
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      $('#composer').requestSubmit();
    }
  });

  $('#composer').addEventListener('submit', (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text || state.busy) return;
    // Echoed immediately so typing feels instant; the server's turn.started
    // replaces it, so live and replayed conversations render identically.
    state.echo = userTurn(text);
    send({ type: 'turn.submit', text });
    input.value = '';
    input.style.height = 'auto';
  });

  $('#stop').onclick = () => send({ type: 'turn.interrupt' });

  const newSession = () => $('#new-dialog').showModal();
  $('#new-session').onclick = newSession;
  $('#nav-new').onclick = newSession;
  $('#new-dialog').addEventListener('close', function () {
    if (this.returnValue === 'create') createSession(this.querySelector('form'));
  });

  // Approval is a live control, not a label. Changing it takes effect from
  // the next tool call — the one already in flight was decided under the old
  // rule, and pretending otherwise would be a lie about what ran.
  $('#mode').onchange = (e) => {
    if (!state.sessionId) return;
    send({ type: 'policy.set', mode: e.target.value });
  };

  // The provider list is long on a well-configured machine and irrelevant
  // most of the time, so it starts folded.
  $('#providers-toggle').onclick = function () {
    const open = this.getAttribute('aria-expanded') !== 'true';
    this.setAttribute('aria-expanded', String(open));
    $('#providers').hidden = !open;
  };

  buildStarters();
  settle();

  $('#voice-toggle').onclick = async () => {
    if (state.voice) {
      state.voice.stop();
      return;
    }
    const voice = startStripVoice();
    try {
      // The strip's call is attached to the open session, so speaking to it
      // here can act on the machine — which is the difference between this
      // and Talk mode's default, where the tick box is off.
      await voice.start({ agentSession: state.sessionId });
      state.voice = voice;
      $('#voice-toggle').classList.add('on');
      $('#voice-toggle').title = 'End voice';
    } catch (err) {
      notice(`Voice could not start: ${err.message}`, 'error');
      voice.stop();
    }
  };

  $('#voice-interrupt').onclick = () => state.voice && state.voice.interrupt();
  $('#voice-mute').onclick = (e) => {
    if (!state.voice) return;
    const muted = !state.voice.muted;
    state.voice.setMuted(muted);
    e.target.textContent = muted ? 'Unmute' : 'Mute';
  };

  // Escape stops whatever is running: the fastest possible way to say "no",
  // and it has to mean that in every mode. Ordered by what is most urgent to
  // stop — something currently talking at you comes before a turn quietly
  // grinding away in the background.
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (isLive()) endLive();
    else if (talking()) stopTalking();
    else if (state.voice) state.voice.interrupt();
    else if (state.busy) send({ type: 'turn.interrupt' });
  });

  // The modes worth a shortcut are the two that change what the keyboard is
  // for. Ctrl+Shift rather than a bare letter, because a bare letter would
  // fire while someone is typing into the composer.
  document.addEventListener('keydown', (e) => {
    if (!e.ctrlKey || !e.shiftKey) return;
    if (e.key.toLowerCase() === 't') {
      e.preventDefault();
      $('#talk-button').click();
    }
  });
}

// The `native` class is added by the desktop shell itself, and only when it
// actually made the window transparent. It is not sniffed from the user agent:
// whether transparency worked is a property of the compositor, which the page
// has no way to observe, and dropping the background without it would leave
// the interface floating on nothing.

/* ------------------------------------------------------------------ modes */

/* The one button. It does the whole thing — switches the app to talking and
   opens the microphone — because a "talk mode" you then have to press a
   second control to actually start talking in is two buttons wearing one
   coat. Pressing it again puts you back where you were. */
function wireTheOneButton() {
  $('#talk-button').onclick = () => {
    if (mode() === 'talk' || mode() === 'live') {
      go('code');
      return;
    }
    go('talk');
    startTalking();
  };
}

/* What each mode owns while it is on screen, and what it must give back.
 *
 * Handing things back is the half that matters. A microphone left open
 * because someone clicked away from Talk is a microphone left open; a frame
 * stream left running is a screen being captured for nobody. So every mode
 * that takes a resource releases it here, in one place, rather than each of
 * them remembering to. */
function wireModeLifecycle() {
  onEnter('talk', () => companion.set('idle'));
  onLeave('talk', () => stopTalking());

  onEnter('live', () => companion.set('idle'));
  onLeave('live', () => endLive());

  onEnter('watch', () => refreshRuns());

  onEnter('studio', () => openStudio());
  onLeave('studio', () => stopPolling());

  onEnter('search', () => focusSearch());
}

wire();
wireMemory();
wireRewind();
wireTheOneButton();

const sessionId = () => state.sessionId;
wireTalk({ sessionId });
wireLive({ sessionId });
wireWatch({
  root: () => (state.info && state.info.root) || null,
  // Autopilot's narration comes off the ordinary agent socket, so watching a
  // run means attaching to its session — which also puts its approvals and
  // questions in the Code tab, with the full context around them.
  attach: (id) => selectSession(id),
});
wireSearch();
// Adding or removing a connection changes what every picker on the page can
// offer, so the whole lot is refreshed rather than the dialog patching them.
wireConnections(() => loadProviders());
wireModeLifecycle();

// Last, and that ordering is load-bearing: this restores the mode you were
// last in and runs that mode's enter hook as it does. Anything registered
// after it is registered too late, which shows up as reopening the page on
// the studio and getting an empty one.
wireModes();

/* The companion, everywhere it has something to say.
 *
 * One creature, one state, and now eight places: the perch it lives on, the
 * sidebar where it doubles as the app's own mark, the middle of an empty
 * session, the composer where it says in words what it is doing, the voice
 * strip, and the three new modes — talking, live, and watching it work.
 * Approval bars grow another on demand. They are all driven from the same
 * object, so the animal is never in two moods at once, and choosing a
 * different one — or none — changes every one of them together. */
companion.mount($('#perch'), { place: 'above' });
companion.mount($('#brand-pet'), { place: 'below', scale: 2 });
companion.attach($('#hero-pet'), { scale: 6 });
companion.attach($('#buddy-pet'), { scale: 1 });
companion.attach($('#voice-pet'), { scale: 1 });

// The words half of the same signal, for anyone who would rather read it
// than watch a moth — and for a screen reader, which cannot watch anything.
companion.watch((what) => {
  $('#buddy-state').textContent = caption(what);
});

loadProviders();
loadSessions();
// Never allowed to throw: this poll is what notices the daemon coming back,
// so it has to survive every second it is down.
setInterval(() => loadSessions().catch(() => {}), 5000);
