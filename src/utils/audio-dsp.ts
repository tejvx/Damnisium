import type { AudioAugConfig } from "../types";

export interface DecodedAudioData {
  sampleRate: number;
  channels: number;
  durationSec: number;
  mono: Float32Array;
  buffer: AudioBuffer;
}

let sharedCtx: AudioContext | null = null;

export function getAudioContext(): AudioContext {
  if (!sharedCtx) {
    sharedCtx = new AudioContext();
  }
  return sharedCtx;
}

export async function decodeAudioBytes(bytes: Uint8Array): Promise<DecodedAudioData> {
  const ctx = getAudioContext();
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const buffer = await ctx.decodeAudioData(copy.buffer);
  const len = buffer.length;
  const mono = new Float32Array(len);
  const ch = buffer.numberOfChannels;
  for (let c = 0; c < ch; c++) {
    const data = buffer.getChannelData(c);
    for (let i = 0; i < len; i++) {
      mono[i] += data[i] / ch;
    }
  }
  return {
    sampleRate: buffer.sampleRate,
    channels: ch,
    durationSec: buffer.duration,
    mono,
    buffer,
  };
}

function fftRadix2(re: Float32Array, im: Float32Array): void {
  const n = re.length;
  let j = 0;
  for (let i = 0; i < n; i++) {
    if (i < j) {
      const tr = re[i];
      const ti = im[i];
      re[i] = re[j];
      im[i] = im[j];
      re[j] = tr;
      im[j] = ti;
    }
    let m = n >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }

  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const step = (-2 * Math.PI) / size;
    for (let i = 0; i < n; i += size) {
      for (let k = 0; k < half; k++) {
        const angle = k * step;
        const wr = Math.cos(angle);
        const wi = Math.sin(angle);
        const idx = i + k + half;
        const tr = wr * re[idx] - wi * im[idx];
        const ti = wr * im[idx] + wi * re[idx];
        re[idx] = re[i + k] - tr;
        im[idx] = im[i + k] - ti;
        re[i + k] += tr;
        im[i + k] += ti;
      }
    }
  }
}

function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel: number): number {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

export function buildMelSpectrogramImage(
  mono: Float32Array,
  sampleRate: number,
  width = 720,
  height = 128
): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx || mono.length < 512) {
    return canvas;
  }

  const fftSize = 512;
  const bins = fftSize / 2;
  const nMels = 64;
  const hann = new Float32Array(fftSize);
  for (let i = 0; i < fftSize; i++) {
    hann[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (fftSize - 1)));
  }

  const minMel = hzToMel(40);
  const maxMel = hzToMel(Math.min(8000, sampleRate / 2));
  const melPoints = new Float32Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) {
    const hz = melToHz(minMel + ((maxMel - minMel) * i) / (nMels + 1));
    melPoints[i] = Math.floor(((fftSize + 1) * hz) / sampleRate);
  }

  const re = new Float32Array(fftSize);
  const im = new Float32Array(fftSize);
  const mag = new Float32Array(bins);
  const matrix = new Float32Array(width * nMels);
  let maxDb = -100;
  let minDb = 0;

  for (let col = 0; col < width; col++) {
    const center = Math.floor((col / width) * (mono.length - fftSize));
    for (let i = 0; i < fftSize; i++) {
      re[i] = (mono[center + i] ?? 0) * hann[i];
      im[i] = 0;
    }
    fftRadix2(re, im);
    for (let k = 0; k < bins; k++) {
      mag[k] = re[k] * re[k] + im[k] * im[k];
    }
    for (let m = 0; m < nMels; m++) {
      const left = melPoints[m];
      const mid = melPoints[m + 1];
      const right = melPoints[m + 2];
      let sum = 1e-9;
      for (let k = Math.max(0, Math.floor(left)); k <= Math.min(bins - 1, Math.ceil(right)); k++) {
        let weight = 0;
        if (k >= left && k <= mid && mid > left) {
          weight = (k - left) / (mid - left);
        } else if (k > mid && k <= right && right > mid) {
          weight = (right - k) / (right - mid);
        }
        sum += mag[k] * weight;
      }
      const db = 10 * Math.log10(sum);
      matrix[col * nMels + m] = db;
      if (db > maxDb) maxDb = db;
      if (db < minDb) minDb = db;
    }
  }

  const img = ctx.createImageData(width, height);
  const range = Math.max(12, maxDb - Math.max(minDb, maxDb - 65));
  const floor = maxDb - range;

  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      const melIdx = Math.min(nMels - 1, Math.floor(((height - 1 - y) / height) * nMels));
      const v = Math.max(0, Math.min(1, (matrix[x * nMels + melIdx] - floor) / range));
      const p = (y * width + x) * 4;
      img.data[p] = Math.round(v < 0.5 ? v * 120 : 60 + (v - 0.5) * 390);
      img.data[p + 1] = Math.round(v < 0.4 ? v * 40 : (v - 0.4) * 380);
      img.data[p + 2] = Math.round(v < 0.6 ? 25 + v * 340 : 229 - (v - 0.6) * 200);
      img.data[p + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

export function playAudioSegment(
  decoded: DecodedAudioData,
  startSec: number,
  endSec: number,
  aug: AudioAugConfig | null,
  onEnded: () => void
): { stop: () => void; startedAt: number; rate: number } {
  const ctx = getAudioContext();
  if (ctx.state === "suspended") {
    void ctx.resume();
  }
  const srcBuf = decoded.buffer;
  const playBuf = ctx.createBuffer(1, srcBuf.length, srcBuf.sampleRate);
  const out = playBuf.getChannelData(0);
  const gainScale = aug ? Math.pow(10, aug.gainDb / 20) : 1;
  const noise = aug ? aug.noiseLevel : 0;

  for (let i = 0; i < out.length; i++) {
    let s = decoded.mono[i] * gainScale;
    if (noise > 0) {
      const n = (Math.random() * 2 - 1) * noise;
      s += n;
    }
    out[i] = Math.max(-1, Math.min(1, s));
  }

  const source = ctx.createBufferSource();
  source.buffer = playBuf;
  const pitchRatio = aug ? Math.pow(2, aug.pitchSemitones / 12) : 1;
  const speedRatio = aug ? aug.speedFactor : 1;
  const rate = Math.max(0.5, Math.min(2.0, pitchRatio * speedRatio));
  source.playbackRate.value = rate;
  source.connect(ctx.destination);

  const safeStart = Math.max(0, Math.min(startSec, decoded.durationSec - 0.02));
  const dur = Math.max(0.05, Math.min(endSec, decoded.durationSec) - safeStart);
  const startedAt = ctx.currentTime;
  let stopped = false;
  source.onended = () => {
    if (!stopped) {
      onEnded();
    }
  };
  source.start(0, safeStart, dur);
  return {
    stop: () => {
      stopped = true;
      try {
        source.stop();
      } catch {}
    },
    startedAt,
    rate,
  };
}

export function resampleTo16kHz(mono: Float32Array, sampleRate: number): Float32Array {
  if (sampleRate === 16000) {
    return mono;
  }
  const targetLen = Math.max(1, Math.round((mono.length * 16000) / sampleRate));
  const out = new Float32Array(targetLen);
  const ratio = sampleRate / 16000;
  for (let i = 0; i < targetLen; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const a = mono[idx] ?? 0;
    const b = mono[Math.min(mono.length - 1, idx + 1)] ?? a;
    out[i] = a + (b - a) * frac;
  }
  return out;
}

export function extractLogMelPatches(
  mono: Float32Array,
  sampleRate: number,
  nMels = 64,
  patchFrames = 96
): Float32Array {
  const y16 = resampleTo16kHz(mono, sampleRate);
  const minSamples = 15600;
  const signal =
    y16.length >= minSamples
      ? y16
      : (() => {
          const p = new Float32Array(minSamples);
          p.set(y16);
          return p;
        })();

  const fftSize = 512;
  const hop = 160;
  const bins = fftSize / 2;
  const hann = new Float32Array(fftSize);
  for (let i = 0; i < fftSize; i++) {
    hann[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (fftSize - 1)));
  }

  const minMel = hzToMel(40);
  const maxMel = hzToMel(7500);
  const melPoints = new Float32Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) {
    const hz = melToHz(minMel + ((maxMel - minMel) * i) / (nMels + 1));
    melPoints[i] = Math.floor(((fftSize + 1) * hz) / 16000);
  }

  const re = new Float32Array(fftSize);
  const im = new Float32Array(fftSize);
  const mag = new Float32Array(bins);
  const out = new Float32Array(patchFrames * nMels);
  let sumAll = 0;

  for (let frame = 0; frame < patchFrames; frame++) {
    const start = frame * hop;
    for (let i = 0; i < fftSize; i++) {
      re[i] = (signal[start + i] ?? 0) * hann[i];
      im[i] = 0;
    }
    fftRadix2(re, im);
    for (let k = 0; k < bins; k++) {
      mag[k] = re[k] * re[k] + im[k] * im[k];
    }
    for (let m = 0; m < nMels; m++) {
      const left = melPoints[m];
      const mid = melPoints[m + 1];
      const right = melPoints[m + 2];
      let sum = 1e-6;
      for (
        let k = Math.max(0, Math.floor(left));
        k <= Math.min(bins - 1, Math.ceil(right));
        k++
      ) {
        let weight = 0;
        if (k >= left && k <= mid && mid > left) {
          weight = (k - left) / (mid - left);
        } else if (k > mid && k <= right && right > mid) {
          weight = (right - k) / (right - mid);
        }
        sum += mag[k] * weight;
      }
      const val = Math.log(sum);
      out[frame * nMels + m] = val;
      sumAll += val;
    }
  }

  const mean = sumAll / out.length;
  let sq = 0;
  for (let i = 0; i < out.length; i++) {
    const d = out[i] - mean;
    sq += d * d;
  }
  const std = Math.sqrt(sq / out.length) + 1e-5;
  for (let i = 0; i < out.length; i++) {
    out[i] = (out[i] - mean) / std;
  }
  return out;
}

export async function startMicRecorder(): Promise<{ stop: () => Promise<DecodedAudioData> }> {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("Microphone capture is unavailable in this environment.");
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const ctx = getAudioContext();
  if (ctx.state === "suspended") {
    await ctx.resume();
  }
  const source = ctx.createMediaStreamSource(stream);
  const processor = ctx.createScriptProcessor(4096, 1, 1);
  const chunks: Float32Array[] = [];
  let totalLen = 0;

  processor.onaudioprocess = (e) => {
    const input = e.inputBuffer.getChannelData(0);
    const copy = new Float32Array(input.length);
    copy.set(input);
    chunks.push(copy);
    totalLen += copy.length;
  };

  source.connect(processor);
  processor.connect(ctx.destination);

  return {
    stop: async () => {
      processor.disconnect();
      source.disconnect();
      for (const tr of stream.getTracks()) {
        tr.stop();
      }
      const mono = new Float32Array(Math.max(1, totalLen));
      let off = 0;
      for (const ch of chunks) {
        mono.set(ch, off);
        off += ch.length;
      }
      const buffer = ctx.createBuffer(1, mono.length, ctx.sampleRate);
      buffer.copyToChannel(mono, 0);
      return {
        sampleRate: ctx.sampleRate,
        channels: 1,
        durationSec: mono.length / ctx.sampleRate,
        mono,
        buffer,
      };
    },
  };
}
