/* The realtime call: speech in, speech out, one model.
 *
 * Structurally similar to Talk and different in the one way that matters. In
 * Talk, this page and the gateway between them decide when you have stopped
 * speaking and when to cut the assistant off. Here the model upstream decides
 * both, and this end mostly relays — which is why the barge-in path looks
 * thin and is not: it has to tell the server *how much of the reply was
 * actually heard*, because the server believes it delivered all of it, and a
 * model whose memory contains three sentences nobody heard will refer back to
 * them.
 *
 * The tool cards are here on purpose. The realtime model can run things on
 * the machine, and an approval prompt that appeared only in the Code
 * transcript would be an approval prompt nobody in this mode ever sees.
 */

import { companion } from './companions/index.js';
import { $, el, socket } from './dom.js';
import { Capture, Player } from './voice.js';

// Not negotiable: the realtime API is 24 kHz both ways, and 16 kHz audio
// makes it hear everything a third too slow — which presents as bad
// transcription rather than as a format error.
const RATE = 24000;

let ws = null;
let player = null;
let capture = null;
let hooks = {};
let saying = null;

function log(kind, text) {
  const box = $('#live-log');
  const node = el('div', `said ${kind}`, text);
  box.appendChild(node);
  box.scrollTop = box.scrollHeight;
  return node;
}

function note(text) {
  const box = $('#live-log');
  box.appendChild(el('div', 'did', text));
  box.scrollTop = box.scrollHeight;
}

function setRunning(on) {
  $('#live-orb').classList.toggle('on', on);
  $('#live-label').textContent = on ? 'Live' : 'Go live';
  $('#live-stop').disabled = !on;
  $('#live-acts').disabled = on;
}

export async function goLive() {
  if (ws) return;

  const wantsTools = $('#live-acts').checked;
  const session = hooks.sessionId ? hooks.sessionId() : null;
  if (wantsTools && !session) {
    $('#live-state').textContent = 'There is no session for it to act in — start one in Code first.';
    return;
  }

  $('#live-state').textContent = 'connecting…';
  ws = new WebSocket(socket('/ws/realtime', { session: wantsTools ? session : null }));
  ws.binaryType = 'arraybuffer';

  ws.onmessage = (msg) => {
    if (msg.data instanceof ArrayBuffer) {
      // One continuous utterance per response; the id is only used to reset
      // the schedule, and the server does not give us one, so a constant does.
      if (player) player.push('live', new Int16Array(msg.data));
      companion.set('speaking');
      return;
    }
    const ev = JSON.parse(msg.data);
    switch (ev.type) {
      case 'realtime.ready':
        setRunning(true);
        $('#live-state').textContent = 'Live. Talk over it whenever you like.';
        $('#live-model').textContent = `${ev.model} · ${ev.voice}`;
        note(ev.session
          ? `Acting in session ${ev.session}. ${ev.policy}`
          : 'Talk only — it has no tools in this call.');
        player = new Player(ev.sample_rate || RATE);
        capture = new Capture();
        capture.start(ev.sample_rate || RATE, (buf) => {
          if (ws && ws.readyState === WebSocket.OPEN) ws.send(buf);
        }).catch((err) => note(`the microphone could not start: ${err.message}`));
        companion.set('listening');
        break;

      case 'realtime.heard':
        if (ev.text) log('you', ev.text);
        $('#live-heard').textContent = '';
        break;

      case 'realtime.said':
        if (!saying) saying = log('it', '');
        saying.textContent += ev.text;
        $('#live-log').scrollTop = $('#live-log').scrollHeight;
        break;

      case 'realtime.speech.started':
        companion.set('listening');
        $('#live-state').textContent = 'Listening.';
        break;

      case 'realtime.interrupted':
        // The server has been told where we actually stopped hearing; this end
        // has to drop what is still queued or it keeps talking regardless.
        if (player) player.cancel();
        if (saying) saying.textContent += ' —';
        saying = null;
        note('interrupted');
        break;

      case 'realtime.audio.done':
        saying = null;
        companion.set('idle');
        break;

      case 'realtime.tool':
        note(`running ${ev.name}`);
        companion.tool(ev.name);
        break;

      case 'realtime.error':
        note(`error: ${ev.message}`);
        companion.flash('error');
        if (ev.fatal) endLive();
        break;
    }
  };

  ws.onclose = () => {
    if (!ws) return;
    endLive(true);
  };
  ws.onerror = () => note('the realtime connection failed');
}

export function endLive(remote) {
  if (ws && !remote && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'realtime.stop' }));
    ws.close();
  }
  if (player) player.close();
  if (capture) capture.stop();
  ws = player = capture = null;
  saying = null;
  setRunning(false);
  $('#live-state').textContent = 'Speech to speech, one model, no transcript in the middle.';
  companion.set('idle');
}

export function isLive() {
  return Boolean(ws);
}

export function wireLive(handlers = {}) {
  hooks = handlers;
  companion.attach($('#live-pet'), { scale: 5 });
  $('#live-orb').onclick = () => (ws ? endLive() : goLive());
  $('#live-stop').onclick = () => endLive();
}
