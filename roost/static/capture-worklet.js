/* Microphone capture, downsampled to whatever the server asked for.
 *
 * A worklet rather than a ScriptProcessor because this runs on the audio
 * thread: a main-thread capture stutters whenever the page renders, and in a
 * duplex call a stutter is a dropped syllable.
 *
 * Downsampling is a box average over the input window, not a plain decimation.
 * Taking every third sample of 48 kHz audio folds everything above 8 kHz back
 * into the speech band as aliasing, which sounds like a lisp and measurably
 * hurts transcription. Averaging is a crude low-pass, but it is one, and it
 * costs an add per sample.
 */
class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const target = (options.processorOptions && options.processorOptions.targetRate) || 16000;
    // `sampleRate` is a worklet global: the context's real rate, which the
    // browser chooses and which is not always 48000.
    this.ratio = sampleRate / target;
    this.sum = 0;
    this.count = 0;
    this.out = [];
    // 20 ms at the target rate, matching the server's frame size so it does
    // not have to reassemble ours.
    this.frame = Math.round(target * 0.02);
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    // No input yet is normal before the mic settles; returning false here
    // would permanently remove the node from the graph.
    if (!channel) return true;

    for (let i = 0; i < channel.length; i++) {
      this.sum += channel[i];
      this.count++;
      if (this.count >= this.ratio) {
        let s = this.sum / this.count;
        s = Math.max(-1, Math.min(1, s));
        this.out.push(s < 0 ? s * 0x8000 : s * 0x7fff);
        // Carry the fractional remainder rather than resetting, so a
        // non-integer ratio does not drift out of time over a long call.
        this.count -= this.ratio;
        this.sum = 0;
      }
    }

    while (this.out.length >= this.frame) {
      const pcm = new Int16Array(this.out.splice(0, this.frame));
      this.port.postMessage(pcm, [pcm.buffer]);
    }
    return true;
  }
}

registerProcessor('roost-capture', CaptureProcessor);
