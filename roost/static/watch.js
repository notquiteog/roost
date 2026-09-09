/* Autopilot: a goal, a screen, and no interface.
 *
 * Everything else here is a conversation. This is a window onto a machine
 * doing something, which is a different thing to build — there is nothing to
 * type, nothing to approve unless it stops and asks, and the only two things
 * on screen are what it is looking at and one line of what it is doing.
 *
 * The narration comes from the ordinary agent socket and the picture comes
 * from a separate one, which is not an accident. The narration is small,
 * ordered, and replayed to a client that reconnects; the video is large,
 * continuous and worthless five seconds later. On one channel you would
 * either store an hour of stale frames to replay, or lose the record of what
 * happened when someone reloads.
 */

import { companion } from './companions/index.js';
import { $, api, el, json, post, socket } from './dom.js';

let screen = null;
let run = null;
let frameUrl = null;
let hooks = {};

function step(text, kind = '') {
  const box = $('#watch-narration');
  const node = el('div', `step ${kind}`, text);
  // The previous "now" stops being now.
  for (const old of box.querySelectorAll('.step.now')) old.classList.remove('now');
  box.appendChild(node);
  box.scrollTop = box.scrollHeight;
  return node;
}

function doing(text) {
  $('#watch-doing').textContent = text;
}

function showFrame(blob) {
  const img = $('#watch-frame');
  const next = URL.createObjectURL(blob);
  // The previous object URL is revoked only once the new frame has decoded,
  // because revoking the one currently painted makes the image flash white
  // between frames — at one and a half frames a second, that is a strobe.
  img.onload = () => {
    if (frameUrl) URL.revokeObjectURL(frameUrl);
    frameUrl = next;
  };
  img.src = next;
  $('#watch-offline').hidden = true;
}

function watchScreen(runId) {
  screen = new WebSocket(socket('/ws/autopilot', { run: runId }));
  screen.binaryType = 'blob';
  screen.onmessage = (msg) => {
    if (msg.data instanceof Blob) showFrame(msg.data);
    else {
      const ev = JSON.parse(msg.data);
      if (ev.type === 'error') step(ev.message, 'bad');
    }
  };
  screen.onclose = () => {
    $('#watch-offline').hidden = false;
    $('#watch-offline').textContent = 'the screen stream ended';
  };
}

/* Agent events, rendered as narration rather than as a transcript. Only three
   things are worth a line: what it said, what it is about to do, and anything
   that stopped it. Tool output is deliberately not shown — this mode is for
   watching, and a build log scrolling past is the transcript again. */
export function narrate(ev) {
  if (!run || ev.session_id !== run.session) return;

  switch (ev.type) {
    case 'text.delta': {
      const box = $('#watch-narration');
      const last = box.lastElementChild;
      if (last && last.classList.contains('said') && last.dataset.open === '1') {
        last.textContent += ev.text;
      } else {
        const node = step(ev.text, 'said');
        node.dataset.open = '1';
      }
      break;
    }
    case 'tool.proposed': {
      for (const old of $('#watch-narration').querySelectorAll('[data-open="1"]')) {
        delete old.dataset.open;
      }
      const node = step(ev.call.summary || ev.call.name, ev.needs_approval ? 'stuck' : 'now');
      if (ev.needs_approval) {
        node.textContent = `waiting for you: ${node.textContent}`;
        doing('needs your approval — the Code tab has the buttons');
      } else {
        doing(ev.call.summary || ev.call.name);
      }
      companion.tool(ev.call.name);
      break;
    }
    case 'question.asked':
      step(`it asked: ${ev.question}`, 'stuck');
      doing('waiting for your answer — answer it in the Code tab');
      companion.set('waiting');
      break;
    case 'tool.denied':
      step(`declined: ${ev.reason}`, 'stuck');
      break;
    case 'error':
      step(ev.message, 'bad');
      companion.flash('error');
      break;
    case 'turn.completed':
      doing(ev.stop_reason === 'end_turn' ? 'finished' : ev.stop_reason);
      step(ev.stop_reason === 'end_turn' ? 'Done.' : `Stopped: ${ev.stop_reason}`,
           ev.stop_reason === 'end_turn' ? '' : 'stuck');
      companion.set('idle');
      break;
  }
}

async function start(event) {
  event.preventDefault();
  const goal = $('#watch-goal').value.trim();
  if (!goal) return;

  const button = $('#watch-go');
  button.disabled = true;
  $('#watch-go').textContent = 'starting…';

  const res = await post('/api/autopilot', {
    goal,
    mode: $('#watch-mode').value,
    share_screen: $('#watch-share').checked,
    root: hooks.root ? hooks.root() : null,
  });

  button.disabled = false;
  $('#watch-go').textContent = 'Start';

  if (!res) {
    step('the daemon is not reachable', 'bad');
    return;
  }
  if (!res.ok) {
    const detail = (await res.json().catch(() => ({}))).detail || 'could not start';
    // Refusals here are the useful kind — "this would take your mouse" — so
    // they go in front of the form rather than into a log nobody has opened.
    const box = $('#watch-start');
    let warn = box.querySelector('.explain.warn');
    if (!warn) {
      warn = el('p', 'explain warn');
      box.appendChild(warn);
    }
    warn.textContent = detail;
    return;
  }

  run = await res.json();
  $('#watch-start').hidden = true;
  $('#watch-live').hidden = false;
  $('#watch-goal-text').textContent = run.goal;
  $('#watch-stage').textContent = run.shares_pointer ? 'your screen' : 'its own screen';
  $('#watch-stage').className = `pill ${run.shares_pointer ? '' : 'on'}`;
  $('#watch-narration').textContent = '';
  $('#watch-offline').hidden = false;
  $('#watch-offline').textContent = 'waiting for the first frame…';
  doing('starting');
  companion.set('thinking');

  watchScreen(run.id);
  // The narration arrives on the agent socket, so the shell is asked to
  // attach to this session — which also means the Code tab is where an
  // approval or a question gets answered, with the full context around it.
  if (hooks.attach) hooks.attach(run.session);
}

export async function stopRun() {
  if (!run) return;
  await api(`/api/autopilot/${run.id}`, { method: 'DELETE' });
  if (screen) screen.close();
  screen = null;
  step('Stopped.', 'stuck');
  doing('stopped');
  run = null;
  $('#watch-live').hidden = true;
  $('#watch-start').hidden = false;
}

export function running() {
  return Boolean(run);
}

export async function refreshRuns() {
  const data = await json('/api/autopilot');
  if (!data || !data.runs.length || run) return;
  // A run left going from a previous page load. Reattaching to it beats
  // showing an empty form next to an agent that is still working.
  const [first] = data.runs;
  run = first;
  $('#watch-start').hidden = true;
  $('#watch-live').hidden = false;
  $('#watch-goal-text').textContent = first.goal;
  $('#watch-stage').textContent = first.shares_pointer ? 'your screen' : 'its own screen';
  step('Reattached to a run that was already going.');
  watchScreen(first.id);
  if (hooks.attach) hooks.attach(first.session);
}

export function wireWatch(handlers = {}) {
  hooks = handlers;
  companion.attach($('#watch-pet'), { scale: 1 });
  $('#watch-start').onsubmit = start;
  $('#watch-stop').onclick = stopRun;
}
