/* Roost's browser client.
 *
 * No framework and no build step: this is served from the machine it runs on,
 * so a toolchain would be a dependency with nothing to show for it.
 *
 * Two sockets. The agent socket is attached to rather than owned — closing
 * this page pauses your view of a session, it does not stop it — so on
 * connect we send the last sequence number we saw and the server replays the
 * gap. The voice socket carries binary audio in both directions and JSON for
 * everything else.
 */

import { companion } from './companions/index.js';

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

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
  // The locally echoed user bubble, removed when the server confirms the turn.
  echo: null,
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
  return node;
}

function notice(text, kind = '') {
  return append(el('div', `notice ${kind}`, text));
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

function toolCard(call, needsApproval) {
  const card = el('div', 'tool');
  const head = el('div', 'head');
  head.appendChild(el('span', 'name', call.name));
  head.appendChild(el('span', 'summary', call.summary || ''));
  head.appendChild(el('span', `risk ${call.risk}`, call.risk));
  card.appendChild(head);
  append(card);
  state.tools.set(call.id, card);

  if (needsApproval) {
    const bar = el('div', 'decide');
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
    deny.onclick = () => {
      send({ type: 'tool.deny', call_id: call.id, reason: 'declined in the UI' });
      bar.remove();
      companion.set('thinking');
    };
    allow.onclick = () => {
      send({ type: 'tool.approve', call_id: call.id, remember: box.checked });
      bar.remove();
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
  }
  if (stream === 'stderr') out.classList.add('err');
  out.textContent += text;
  // A long build must not push everything else off screen.
  if (out.textContent.length > 40000) out.textContent = out.textContent.slice(-40000);
  out.scrollTop = out.scrollHeight;
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
  } else if (!live && !shot && result.content) {
    const out = el('div', result.ok ? 'out' : 'out err');
    out.textContent = result.content.length > 4000 ? result.content.slice(0, 4000) + '\n…' : result.content;
    card.appendChild(out);
  }

  if (result.name === 'desktop_click' && result.display && result.ok) {
    markClick(card, result.display.x, result.display.y, result.display.label);
  }

  const meta = card.querySelector('.summary');
  if (meta && result.duration_ms > 400) meta.textContent += `   ${(result.duration_ms / 1000).toFixed(1)}s`;
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
  $('#send').disabled = busy || !state.sessionId;
  $('#stop').hidden = !busy;
}

function setLink(on) {
  const pill = $('#link');
  pill.textContent = on ? 'live' : 'offline';
  pill.className = `pill ${on ? 'on' : 'off'}`;
}

function connectAgent(sessionId) {
  if (state.agent) {
    state.agent.onclose = null;
    state.agent.close();
  }
  state.sessionId = sessionId;

  const url = new URL(`/ws/agent`, location.href);
  url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  url.searchParams.set('session', sessionId);
  // Resume rather than replay everything: on a reconnect we only want the gap.
  url.searchParams.set('since', String(state.seq));
  const token = new URLSearchParams(location.search).get('token');
  if (token) url.searchParams.set('token', token);

  const ws = new WebSocket(url);
  state.agent = ws;

  ws.onopen = () => {
    setLink(true);
    // Whatever arrives next is the replay of what was missed, not news.
    companion.resumed();
    $('#input').disabled = false;
    $('#send').disabled = false;
    $('#voice-toggle').disabled = false;
    flushOutbox();
  };

  ws.onmessage = (msg) => handleAgentEvent(JSON.parse(msg.data));

  ws.onclose = () => {
    setLink(false);
    // The session is still there; only our view of it went away. Retry
    // quietly rather than announcing a failure that is usually a laptop lid.
    setTimeout(() => {
      if (state.sessionId === sessionId) connectAgent(sessionId);
    }, 1500);
  };
}

function handleAgentEvent(ev) {
  switch (ev.type) {
    case 'session.started':
      $('#session-meta').textContent = `${ev.model} · ${ev.policy}`;
      break;

    case 'turn.started':
      state.turnNode = null;
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

    case 'turn.completed':
      setBusy(false);
      state.turnNode = null;
      companion.set('idle');
      if (ev.stop_reason === 'interrupted') notice('Interrupted.');
      else if (ev.stop_reason === 'max_steps') notice('Stopped: too many steps.', 'error');
      else if (ev.stop_reason === 'error') companion.flash('error');
      else companion.flash('success');
      break;

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

class Player {
  /* Schedules PCM16 into the audio graph and can drop it all instantly.
   *
   * That second part is the whole reason this is not an <audio> element:
   * barge-in has to stop what is already queued, and a media element gives
   * you no handle on individual buffers. */
  constructor(rate) {
    this.rate = rate;
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.next = 0;
    this.sources = new Set();
    this.utterance = null;
  }

  push(utteranceId, pcm) {
    if (utteranceId !== this.utterance) {
      this.utterance = utteranceId;
      // A small lead so the first buffer is not scheduled in the past on a
      // loaded machine, which drops it silently.
      this.next = this.ctx.currentTime + 0.06;
    }
    const buffer = this.ctx.createBuffer(1, pcm.length, this.rate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 32768;

    const source = this.ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.ctx.destination);
    const at = Math.max(this.next, this.ctx.currentTime);
    source.start(at);
    this.next = at + buffer.duration;
    this.sources.add(source);
    source.onended = () => this.sources.delete(source);
  }

  cancel() {
    for (const source of this.sources) {
      try { source.stop(); } catch { /* already finished */ }
    }
    this.sources.clear();
    this.next = 0;
    this.utterance = null;
  }

  get speaking() {
    return this.sources.size > 0;
  }
}

class Voice {
  constructor() {
    this.ws = null;
    this.player = null;
    this.stream = null;
    this.ctx = null;
    this.muted = false;
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });

    const url = new URL('/ws/voice', location.href);
    url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const token = new URLSearchParams(location.search).get('token');
    if (token) url.searchParams.set('token', token);

    this.ws = new WebSocket(url);
    this.ws.binaryType = 'arraybuffer';

    await new Promise((resolve, reject) => {
      this.ws.onopen = resolve;
      this.ws.onerror = () => reject(new Error('could not open the voice socket'));
    });

    this.ws.send(JSON.stringify({ type: 'voice.start', format: { sample_rate: 16000, encoding: 'pcm16' } }));
    this.ws.onmessage = (msg) => this.onMessage(msg);
    this.ws.onclose = () => this.stop(true);
  }

  async beginCapture(rate) {
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    await this.ctx.audioWorklet.addModule('/static/capture-worklet.js');
    const source = this.ctx.createMediaStreamSource(this.stream);
    const node = new AudioWorkletNode(this.ctx, 'roost-capture', {
      processorOptions: { targetRate: rate },
    });
    node.port.onmessage = (e) => {
      if (this.muted) return;
      if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(e.data.buffer);
    };
    source.connect(node);
    // Deliberately not connected to the destination: routing the microphone
    // to the speakers is how you get feedback.
    this.captureNode = node;
  }

  onMessage(msg) {
    if (msg.data instanceof ArrayBuffer) {
      // 12 ASCII bytes of utterance id, then PCM16.
      const view = new Uint8Array(msg.data);
      const id = new TextDecoder().decode(view.slice(0, 12)).trim();
      const pcm = new Int16Array(msg.data.slice(12));
      if (this.player) this.player.push(id, pcm);
      setVoiceDot('speaking');
      return;
    }

    const ev = JSON.parse(msg.data);
    switch (ev.type) {
      case 'voice.ready':
        this.player = new Player(ev.output_format.sample_rate);
        this.beginCapture(ev.format.sample_rate);
        $('#voice-strip').hidden = false;
        setVoiceState(`${ev.stt} → ${ev.llm} → ${ev.tts}`);
        break;

      case 'vad.speech_started':
        setVoiceDot('hearing');
        companion.set('listening');
        break;

      case 'vad.speech_stopped':
        setVoiceDot('');
        companion.set('thinking');
        break;

      case 'transcript.partial':
        $('#voice-partial').textContent = ev.text;
        break;

      case 'transcript.final':
        $('#voice-partial').textContent = '';
        if (ev.submitted) userTurn(ev.text);
        else notice(`Heard "${ev.text}" — too short to answer.`);
        break;

      case 'assistant.text.delta':
        appendText(ev.text);
        break;

      case 'speech.started':
        state.turnNode = null;
        companion.set('speaking');
        break;

      case 'speech.cancelled':
        // The decisive moment. Everything already scheduled has to go, or the
        // assistant keeps talking over you for as long as the buffer lasts.
        if (this.player) this.player.cancel();
        setVoiceDot('');
        notice('— interrupted');
        companion.set('listening');
        break;

      case 'speech.stopped':
        setVoiceDot('');
        companion.set('idle');
        break;

      case 'voice.error':
        notice(ev.message, 'error');
        break;
    }
  }

  interrupt() {
    if (this.player) this.player.cancel();
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'voice.interrupt' }));
    }
  }

  setMuted(muted) {
    this.muted = muted;
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'voice.mute', muted }));
    }
  }

  stop(remote) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN && !remote) {
      this.ws.send(JSON.stringify({ type: 'voice.stop' }));
      this.ws.close();
    }
    if (this.player) this.player.cancel();
    if (this.captureNode) this.captureNode.disconnect();
    if (this.ctx) this.ctx.close();
    if (this.stream) for (const track of this.stream.getTracks()) track.stop();
    this.ws = this.ctx = this.stream = this.player = null;
    companion.set('idle');
    $('#voice-strip').hidden = true;
    $('#voice-toggle').textContent = 'Start voice';
    state.voice = null;
  }
}

function setVoiceDot(cls) {
  $('#voice-dot').className = `dot ${cls}`;
}
function setVoiceState(text) {
  $('#voice-state').textContent = text;
}


/* ------------------------------------------------------------------ memory */

/* Memory is off on some installs entirely, and off per person on the rest.
   The button is only shown when the endpoints exist, so nothing here implies
   a feature that is not there. */

async function memorySettings() {
  const res = await fetch('/api/memory/settings');
  if (res.status === 404) return null;      // not enabled on this server
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

  const res = await fetch('/api/memory');
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
      await fetch(`/api/memory/${m.id}`, { method: 'DELETE' });
      loadMemory();
    };
    li.appendChild(forget);
    list.appendChild(li);
  }
}

async function saveMemorySettings() {
  await fetch('/api/memory/settings', {
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
    await fetch('/api/memory', { method: 'DELETE' });
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
  refreshTimer = setTimeout(loadSessions, 700);
}

async function loadSessions() {
  const res = await fetch('/api/sessions');
  if (!res.ok) return;
  const { sessions } = await res.json();

  // On a fresh page, reattach to whatever was open. The session outlived the
  // tab; the point of that is undermined if reopening starts from nothing.
  if (!state.sessionId && sessions.length) {
    const wanted = lastSession();
    const resume = sessions.find((s) => s.id === wanted) || sessions[0];
    if (resume) {
      $('#session-title').textContent = resume.title || resume.id;
      $('#mode-wrap').hidden = false;
      $('#mode').value = resume.policy;
      selectSession(resume.id);
    }
  }

  const list = $('#sessions');
  list.textContent = '';

  for (const s of sessions) {
    const li = el('li');
    li.setAttribute('aria-current', String(s.id === state.sessionId));
    li.appendChild(el('span', 'name', s.title || s.id));

    const sub = el('div', 'sub');
    let status = `${s.turns} turn${s.turns === 1 ? '' : 's'}`;
    if (s.busy) status = 'working…';
    if (s.waiting_on) status = s.waiting_on === 'approval' ? 'needs approval' : 'asked you something';
    sub.appendChild(el('span', '', status));
    li.appendChild(sub);

    li.onclick = () => selectSession(s.id);
    list.appendChild(li);
  }
}

async function loadProviders() {
  const res = await fetch('/api/providers');
  if (!res.ok) return;
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

  const select = $('#new-dialog select[name=provider]');
  for (const p of data.providers || []) {
    if (!p.modalities.includes('chat')) continue;
    const option = el('option', '', `${p.label}${p.local ? ' (local)' : ''}`);
    option.value = p.id;
    select.appendChild(option);
  }
}

const LAST_SESSION = 'roost.session';

function rememberSession(id) {
  // Wrapped because a browser with site data blocked throws on access rather
  // than returning null, and losing the page over a convenience is silly.
  try { localStorage.setItem(LAST_SESSION, id); } catch { /* private window */ }
}

function lastSession() {
  try { return localStorage.getItem(LAST_SESSION); } catch { return null; }
}

function selectSession(id) {
  if (id === state.sessionId) return;
  transcript.textContent = '';
  state.tools.clear();
  state.seq = 0;
  state.turnNode = null;
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
  };
  const res = await fetch('/api/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    notice((await res.json()).detail || 'could not create the session', 'error');
    return;
  }
  const session = await res.json();
  $('#session-title').textContent = session.title;
  $('#mode-wrap').hidden = false;
  $('#mode').value = body.mode;
  selectSession(session.id);
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

  $('#new-session').onclick = () => $('#new-dialog').showModal();
  $('#new-dialog').addEventListener('close', function () {
    if (this.returnValue === 'create') createSession(this.querySelector('form'));
  });

  $('#voice-toggle').onclick = async () => {
    if (state.voice) {
      state.voice.stop();
      return;
    }
    const voice = new Voice();
    try {
      await voice.start();
      state.voice = voice;
      $('#voice-toggle').textContent = 'End voice';
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

  // Escape stops whatever is running: the fastest possible way to say "no".
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (state.voice) state.voice.interrupt();
    else if (state.busy) send({ type: 'turn.interrupt' });
  });
}

// The `native` class is added by the desktop shell itself, and only when it
// actually made the window transparent. It is not sniffed from the user agent:
// whether transparency worked is a property of the compositor, which the page
// has no way to observe, and dropping the background without it would leave
// the interface floating on nothing.

wire();
wireMemory();
companion.mount($('#perch'));
loadProviders();
loadSessions();
setInterval(loadSessions, 5000);
