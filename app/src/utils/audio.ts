// High-performance Web Audio API synthesizer for instant zero-latency feedback
let audioCtx: AudioContext | null = null;

function getAudioContext(): AudioContext | null {
  if (typeof window === "undefined") return null;
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    if (AudioContextClass) {
      audioCtx = new AudioContextClass();
    }
  }
  return audioCtx;
}

export type SoundType = "pop" | "end" | "bubble" | "slap" | "chime";

export function playAudioFeedback(type: SoundType, enabled: boolean = true) {
  if (!enabled) return;
  try {
    const ctx = getAudioContext();
    if (!ctx) return;

    if (ctx.state === "suspended") {
      ctx.resume().catch(() => {});
    }

    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    const now = ctx.currentTime;

    switch (type) {
      case "pop": {
        osc.frequency.setValueAtTime(440, now);
        osc.frequency.exponentialRampToValueAtTime(110, now + 0.1);
        gain.gain.setValueAtTime(0.12, now);
        osc.start(now);
        osc.stop(now + 0.1);
        break;
      }
      case "end": {
        osc.frequency.setValueAtTime(300, now);
        osc.frequency.linearRampToValueAtTime(100, now + 0.3);
        gain.gain.setValueAtTime(0.15, now);
        osc.start(now);
        osc.stop(now + 0.3);
        break;
      }
      case "bubble": {
        osc.frequency.setValueAtTime(600, now);
        osc.frequency.exponentialRampToValueAtTime(1200, now + 0.15);
        gain.gain.setValueAtTime(0.08, now);
        osc.start(now);
        osc.stop(now + 0.15);
        break;
      }
      case "slap": {
        osc.frequency.setValueAtTime(800, now);
        osc.frequency.exponentialRampToValueAtTime(180, now + 0.12);
        gain.gain.setValueAtTime(0.18, now);
        osc.start(now);
        osc.stop(now + 0.15);
        break;
      }
      case "chime": {
        osc.frequency.setValueAtTime(350, now);
        osc.frequency.exponentialRampToValueAtTime(500, now + 0.1);
        gain.gain.setValueAtTime(0.1, now);
        osc.start(now);
        osc.stop(now + 0.15);
        break;
      }
    }
  } catch (e) {
    console.warn("Audio feedback error:", e);
  }
}
