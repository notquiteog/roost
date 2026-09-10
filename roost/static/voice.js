/* The voice call, as a thing two screens can both use.
 *
 * This was inside the main client until Talk mode needed the same call with a
 * completely different surface around it. Copying it would have been the
 * expensive kind of duplication: barge-in is the delicate part, and a second
 * copy is a second place for the cancel path to drift — which would present
 * as the assistant talking over you on one screen and not the other.
 *
 * So the call is here and knows nothing about the page. It reports what
 * happened through handlers; what to draw is the caller's problem.
 */

import { socket } from './dom.js';

export class Player {
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
    // Samples handed to the graph for the current utterance. The realtime
    // protocol needs this on a barge-in: the server believes it delivered
    // everything it sent, and has to be told where the person stopped hearing.
    this.played = 0;
  }

  push(utteranceId, pcm) {
    if (utteranceId !== this.utterance) {
      this.utterance = utteranceId;
      this.played = 0;
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
    this.played += pcm.length;
    this.sources.add(source);
    source.onended = () => this.sources.delete(source);
  }

  /* How much of the current utterance has actually reached the speakers,
     rather than how much was handed over. The difference is everything that
     is still scheduled ahead of `currentTime`, and it is the number the
     server needs. */
  heardMs() {
    const ahead = Math.max(0, this.next - this.ctx.currentTime);
    return Math.max(0, Math.round((this.played / this.rate - ahead) * 1000));
  }

  cancel() {
    for (const source of this.sources) {
      try { source.stop(); } catch { /* already finished */ }
    }
    this.sources.clear();
    this.next = 0;
    this.played = 0;
    this.utterance = null;
  }

  get speaking() {
    return this.sources.size > 0;
  }

  close() {
    this.cancel();
    try { this.ctx.close(); } catch { /* already closed */ }
  }
}

/* Microphone capture, shared by both kinds of call.
 *
 * Deliberately never connected to the destination: routing the microphone to
 * the speakers is how you get feedback, and it is a one-line mistake. */
export class Capture {
  constructor() {
    this.stream = null;
    this.ctx = null;
    this.node = null;
    this.muted = false;
  }

  async start(rate, onFrame) {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    await this.ctx.audioWorklet.addModule('/static/capture-worklet.js');
    const source = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, 'roost-capture', { processorOptions: { targetRate: rate } });
    this.node.port.onmessage = (e) => {
      if (!this.muted) onFrame(e.data.buffer);
    };
    source.connect(this.node);
  }

  stop() {
    if (this.node) this.node.disconnect();
    if (this.ctx) { try { this.ctx.close(); } catch { /* already closed */ } }
    if (this.stream) for (const track of this.stream.getTracks()) track.stop();
    this.node = this.ctx = this.stream = null;
  }
}

/* One duplex voice call over /ws/voice.
 *
 * `on` is a bag of handlers, all optional. Anything not supplied is simply
 * not reported — a caller that only wants to hear the audio does not have to
 * write eight empty functions. */
export class Voice {
  constructor(on = {}) {
    this.on = on;
    this.ws = null;
    this.player = null;
    this.capture = new Capture();
    this.muted = false;
    this.running = false;
  }

  fire(name, ...args) {
    const fn = this.on[name];
    if (fn) fn(...args);
  }

  async start(options = {}) {
    const url = socket('/ws/voice');
    this.ws = new WebSocket(url);
    this.ws.binaryType = 'arraybuffer';

    await new Promise((resolve, reject) => {
      this.ws.onopen = resolve;
      this.ws.onerror = () => reject(new Error('could not open the voice socket'));
    });

    this.ws.send(JSON.stringify({
      type: 'voice.start',
      format: { sample_rate: 16000, encoding: 'pcm16' },
      // Opting a voice call into acting on the machine is per call, because
      // consenting to be listened to is not the same as consenting to have
      // commands run.
      agent_session_id: options.agentSession || undefined,
      stt: options.stt || undefined,
      tts: options.tts || undefined,
      llm: options.llm || undefined,
      model: options.model || undefined,
      voice: options.voice || undefined,
    }));

    this.ws.onmessage = (msg) => this.receive(msg);
    this.ws.onclose = () => this.stop(true);
    this.running = true;
  }

  receive(msg) {
    if (msg.data instanceof ArrayBuffer) {
      // 12 ASCII bytes of utterance id, then PCM16.
      const view = new Uint8Array(msg.data);
      const id = new TextDecoder().decode(view.slice(0, 12)).trim();
      const pcm = new Int16Array(msg.data.slice(12));
      if (this.player) this.player.push(id, pcm);
      this.fire('speaking');
      return;
    }

    const ev = JSON.parse(msg.data);
    switch (ev.type) {
      case 'voice.ready':
        this.player = new Player(ev.output_format.sample_rate);
        this.capture.start(ev.format.sample_rate, (buf) => {
          if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(buf);
        }).catch((err) => this.fire('error', `the microphone could not start: ${err.message}`));
        this.fire('ready', ev);
        break;

      case 'vad.speech_started': this.fire('listening'); break;
      case 'vad.speech_stopped': this.fire('thinking'); break;
      case 'transcript.partial': this.fire('partial', ev.text); break;
      case 'transcript.final': this.fire('heard', ev.text, ev.submitted); break;
      case 'assistant.text.delta': this.fire('said', ev.text); break;
      case 'speech.started': this.fire('speaking'); break;

      case 'speech.cancelled':
        // The decisive moment. Everything already scheduled has to go, or the
        // assistant keeps talking over you for as long as the buffer lasts.
        if (this.player) this.player.cancel();
        this.fire('cancelled');
        break;

      case 'speech.stopped': this.fire('finished'); break;

      // The turn has stopped and is waiting on the person. It is spoken as
      // well — that is the point of it — but a client that reconnects
      // mid-suspension has no other way to tell waiting from idle.
      case 'voice.waiting': this.fire('waiting', ev); break;
      case 'voice.answered': this.fire('answered', ev); break;
      case 'voice.error': this.fire('error', ev.message); break;
    }
  }

  interrupt() {
    if (this.player) this.player.cancel();
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'voice.interrupt' }));
    }
  }

  say(text) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'voice.text', text }));
    }
  }

  setMuted(muted) {
    this.muted = muted;
    this.capture.muted = muted;
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'voice.mute', muted }));
    }
  }

  stop(remote) {
    if (!this.running) return;
    this.running = false;
    if (this.ws && this.ws.readyState === WebSocket.OPEN && !remote) {
      this.ws.send(JSON.stringify({ type: 'voice.stop' }));
      this.ws.close();
    }
    if (this.player) this.player.close();
    this.capture.stop();
    this.ws = null;
    this.player = null;
    this.fire('stopped');
  }
}
