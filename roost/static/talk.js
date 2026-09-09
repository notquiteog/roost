/* Talking to it, and nothing else.
 *
 * The brief was one button that turns the app into something you speak to.
 * The temptation is to make that a toggle on the existing screen — leave the
 * transcript, leave the composer, add a microphone. That is not the same
 * thing: a screen with a text box on it is a screen you type into, and the
 * mode is supposed to be *only* speaking.
 *
 * So this pane has no composer, no tool cards and no approval bars in the
 * ordinary sense. What it has is what someone who is not looking at the
 * screen needs: a control big enough to hit without aiming, a line of what
 * was just heard, and a record of the conversation for when they look back.
 *
 * Acting on the machine is a tick box, off by default and off per call.
 * Consenting to be listened to is not consenting to have commands run, and
 * the voice gateway enforces that separately — the box only decides whether
 * an agent session is attached at all.
 */

import { companion } from './companions/index.js';
import { $, el } from './dom.js';
import { Voice } from './voice.js';

let voice = null;
let handlers = {};
let saying = null;

function log(kind, text) {
  const box = $('#talk-log');
  const node = el('div', `said ${kind}`, text);
  box.appendChild(node);
  box.scrollTop = box.scrollHeight;
  return node;
}

function note(text) {
  const box = $('#talk-log');
  box.appendChild(el('div', 'did', text));
  box.scrollTop = box.scrollHeight;
}

function setState(text) {
  $('#talk-state').textContent = text;
}

function setRunning(on) {
  $('#talk-orb').classList.toggle('on', on);
  $('#talk-label').textContent = on ? 'Listening' : 'Start talking';
  $('#talk-mute').disabled = !on;
  $('#talk-stop').disabled = !on;
  $('#talk-acts').disabled = on;
}

export async function startTalking() {
  if (voice) return;

  const wantsTools = $('#talk-acts').checked;
  const session = handlers.sessionId ? handlers.sessionId() : null;
  if (wantsTools && !session) {
    setState('There is no session for it to act in — start one in Code first.');
    return;
  }

  setState('opening the microphone…');
  const call = new Voice({
    ready: (ev) => {
      setRunning(true);
      setState('Listening. Just talk — it will answer when you stop.');
      $('#talk-route').textContent = `${ev.stt} → ${ev.llm} → ${ev.tts}`;
      note(wantsTools ? 'This call can act on the machine.' : 'Talk only — it has no tools in this call.');
      companion.set('listening');
    },
    listening: () => {
      companion.set('listening');
      setState('Listening.');
    },
    thinking: () => {
      companion.set('thinking');
      setState('Thinking…');
    },
    partial: (text) => { $('#talk-heard').textContent = text; },
    heard: (text, submitted) => {
      $('#talk-heard').textContent = '';
      if (submitted) log('you', text);
      else note(`heard "${text}" — too short to answer`);
    },
    said: (delta) => {
      // One node per utterance, appended to. A node per delta would produce a
      // log made of single words.
      if (!saying) saying = log('it', '');
      saying.textContent += delta;
      $('#talk-log').scrollTop = $('#talk-log').scrollHeight;
    },
    speaking: () => {
      companion.set('speaking');
      setState('Speaking. Talk over it to interrupt.');
    },
    cancelled: () => {
      if (saying) saying.textContent += ' —';
      saying = null;
      note('interrupted');
      setState('Listening.');
      companion.set('listening');
    },
    finished: () => {
      saying = null;
      companion.set('idle');
      setState('Listening.');
    },
    error: (message) => {
      note(`error: ${message}`);
      companion.flash('error');
    },
    stopped: () => {
      voice = null;
      saying = null;
      setRunning(false);
      setState('The microphone is off.');
      companion.set('idle');
    },
  });

  try {
    await call.start({ agentSession: wantsTools ? session : null });
    voice = call;
  } catch (err) {
    setState(`Could not start: ${err.message}`);
    call.stop();
  }
}

export function stopTalking() {
  if (voice) voice.stop();
  voice = null;
}

export function talking() {
  return Boolean(voice);
}

export function wireTalk(hooks = {}) {
  handlers = hooks;
  companion.attach($('#talk-pet'), { scale: 5 });

  $('#talk-orb').onclick = () => (voice ? stopTalking() : startTalking());
  $('#talk-stop').onclick = () => stopTalking();
  $('#talk-mute').onclick = (e) => {
    if (!voice) return;
    const muted = !voice.muted;
    voice.setMuted(muted);
    e.target.textContent = muted ? 'Unmute' : 'Mute';
    setState(muted ? 'Muted — it cannot hear you.' : 'Listening.');
  };

  // The tick box only means anything before a call starts, and saying so is
  // better than letting someone tick it mid-conversation and wonder why
  // nothing changed.
  $('#talk-acts').onchange = (e) => {
    setState(e.target.checked
      ? 'It will be able to act on the machine in this call, under the same approval rules.'
      : 'Talk only — it will have no tools.');
  };
}
