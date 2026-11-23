import type * as TransformersTypes from "@huggingface/transformers";
import type * as OrtTypes from "onnxruntime-web/wasm";
import { api } from "../api";
import { COCO_CLASSES } from "../constants";
import type { DecodedAudioData } from "./audio-dsp";
import { extractLogMelPatches, resampleTo16kHz } from "./audio-dsp";

let ortLib: typeof OrtTypes | null = null;
let hfLib: typeof TransformersTypes | null = null;

async function ensureOrt(): Promise<typeof OrtTypes> {
  if (!ortLib) {
    ortLib = await import("onnxruntime-web/wasm");
    ortLib.env.wasm.numThreads = 1;
    ortLib.env.wasm.wasmPaths = "/ort/";
  }
  return ortLib;
}

async function ensureHf(): Promise<typeof TransformersTypes> {
  if (!hfLib) {
    hfLib = await import("@huggingface/transformers");
    hfLib.env.allowRemoteModels = false;
    hfLib.env.allowLocalModels = true;
    hfLib.env.useBrowserCache = false;
    const onnxEnv = hfLib.env.backends.onnx as {
      wasm?: { wasmPaths?: string; numThreads?: number };
    };
    if (onnxEnv.wasm) {
      onnxEnv.wasm.wasmPaths = "/ort/";
      onnxEnv.wasm.numThreads = 1;
    } else {
      onnxEnv.wasm = { wasmPaths: "/ort/", numThreads: 1 };
    }
  }
  return hfLib;
}

export interface EvalBoxDetection {
  x: number;
  y: number;
  w: number;
  h: number;
  score: number;
  classIndex: number;
  className: string;
  maskData?: Uint8Array;
  maskW?: number;
  maskH?: number;
}

export interface EvalClassificationScore {
  label: string;
  score: number;
}

export interface EvalTranscriptChunk {
  text: string;
  startSec: number | null;
  endSec: number | null;
}

type DisposablePipeline = {
  (input: unknown, options?: Record<string, unknown>): Promise<unknown>;
  dispose?: () => Promise<void>;
};

let activeOrtPath: string | null = null;
let activeOrtSession: OrtTypes.InferenceSession | null = null;
let activeHfKey: string | null = null;
let activeHfPipe: DisposablePipeline | null = null;
let activeGgufInUse = false;
const runtimeListeners = new Set<() => void>();

function notifyRuntimeChange(): void {
  for (const cb of runtimeListeners) {
    cb();
  }
}

export function onRuntimeChange(cb: () => void): () => void {
  runtimeListeners.add(cb);
  return () => {
    runtimeListeners.delete(cb);
  };
}

export function setGgufRuntimeActive(active: boolean): void {
  if (activeGgufInUse !== active) {
    activeGgufInUse = active;
    notifyRuntimeChange();
  }
}

export async function disposeTestRuntime(): Promise<void> {
  const hadAny = hasLoadedRuntime();
  if (activeOrtSession) {
    try {
      await activeOrtSession.release();
    } catch {}
    activeOrtSession = null;
    activeOrtPath = null;
  }
  if (activeHfPipe) {
    try {
      if (typeof activeHfPipe.dispose === "function") {
        await activeHfPipe.dispose();
      }
    } catch {}
    activeHfPipe = null;
    activeHfKey = null;
  }
  if (activeGgufInUse) {
    activeGgufInUse = false;
    try {
      await api.unloadGgufRuntime();
    } catch {}
  }
  if (hadAny) {
    notifyRuntimeChange();
  }
}

export function hasLoadedRuntime(): boolean {
  return activeOrtSession !== null || activeHfPipe !== null || activeGgufInUse;
}

async function getOrtSession(onnxPath: string): Promise<OrtTypes.InferenceSession> {
  const ort = await ensureOrt();
  if (activeOrtSession && activeOrtPath === onnxPath) {
    return activeOrtSession;
  }
  await disposeTestRuntime();
  const bytes = await api.readBinaryFile(onnxPath);
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ["wasm"],
  });
  activeOrtPath = onnxPath;
  activeOrtSession = session;
  notifyRuntimeChange();
  return session;
}

async function getHfPipeline(
  task: "automatic-speech-recognition" | "text-classification" | "text-generation",
  onnxDir: string,
  onnxPath: string
): Promise<DisposablePipeline> {
  const { pipeline } = await ensureHf();
  const key = `${task}::${onnxDir}`;
  if (activeHfPipe && activeHfKey === key) {
    return activeHfPipe;
  }
  await disposeTestRuntime();
  const baseUrl = api.fileSrc(onnxDir).replace(/\/+$/, "");
  const subfolder = onnxPath.replace(/\\/g, "/").includes("/onnx/") ? "onnx" : "";
  const created = (await pipeline(task, baseUrl, {
    local_files_only: true,
    device: "wasm",
    subfolder,
  })) as unknown as DisposablePipeline;
  activeHfKey = key;
  activeHfPipe = created;
  notifyRuntimeChange();
  return created;
}

function loadHtmlImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed loading image for evaluation"));
    img.src = src;
  });
}

async function preprocessLetterbox(
  img: HTMLImageElement,
  size: number
): Promise<{ tensor: OrtTypes.Tensor; scale: number; padX: number; padY: number }> {
  const ort = await ensureOrt();
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2D canvas unavailable");

  const iw = img.naturalWidth || img.width;
  const ih = img.naturalHeight || img.height;
  const scale = Math.min(size / iw, size / ih);
  const nw = Math.round(iw * scale);
  const nh = Math.round(ih * scale);
  const padX = (size - nw) / 2;
  const padY = (size - nh) / 2;

  ctx.fillStyle = "rgb(114, 114, 114)";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(img, padX, padY, nw, nh);

  const pixels = ctx.getImageData(0, 0, size, size).data;
  const area = size * size;
  const floatData = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    const p = i * 4;
    floatData[i] = pixels[p] / 255;
    floatData[area + i] = pixels[p + 1] / 255;
    floatData[2 * area + i] = pixels[p + 2] / 255;
  }
  return {
    tensor: new ort.Tensor("float32", floatData, [1, 3, size, size]),
    scale,
    padX,
    padY,
  };
}

async function preprocessImageNet(img: HTMLImageElement, size: number): Promise<OrtTypes.Tensor> {
  const ort = await ensureOrt();
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2D canvas unavailable");
  ctx.drawImage(img, 0, 0, size, size);
  const pixels = ctx.getImageData(0, 0, size, size).data;
  const area = size * size;
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  const floatData = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    const p = i * 4;
    floatData[i] = (pixels[p] / 255 - mean[0]) / std[0];
    floatData[area + i] = (pixels[p + 1] / 255 - mean[1]) / std[1];
    floatData[2 * area + i] = (pixels[p + 2] / 255 - mean[2]) / std[2];
  }
  return new ort.Tensor("float32", floatData, [1, 3, size, size]);
}

function softmax(logits: Float32Array): number[] {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > max) max = logits[i];
  }
  const exps: number[] = [];
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - max);
    exps.push(e);
    sum += e;
  }
  return exps.map((e) => e / (sum || 1));
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function boxIou(a: EvalBoxDetection, b: EvalBoxDetection): number {
  const ax2 = a.x + a.w;
  const ay2 = a.y + a.h;
  const bx2 = b.x + b.w;
  const by2 = b.y + b.h;
  const ix1 = Math.max(a.x, b.x);
  const iy1 = Math.max(a.y, b.y);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

export function filterDetectionsLive(
  rawCandidates: EvalBoxDetection[],
  confThresh: number,
  iouThresh: number
): EvalBoxDetection[] {
  const filtered = rawCandidates
    .filter((d) => d.score >= confThresh)
    .sort((a, b) => b.score - a.score);
  const kept: EvalBoxDetection[] = [];
  for (const cand of filtered) {
    let suppressed = false;
    for (const prev of kept) {
      if (prev.classIndex === cand.classIndex && boxIou(prev, cand) > iouThresh) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      kept.push(cand);
    }
  }
  return kept;
}

function decodeProtoMask(
  coeffs: number[],
  protos: Float32Array,
  protoC: number,
  protoH: number,
  protoW: number,
  bx: number,
  by: number,
  bw: number,
  bh: number,
  iw: number,
  ih: number,
  inputSize: number,
  scale: number,
  padX: number,
  padY: number
): Uint8Array {
  const out = new Uint8Array(protoW * protoH);
  const area = protoW * protoH;
  for (let py = 0; py < protoH; py++) {
    const imgYNorm = ((py + 0.5) * (inputSize / protoH) - padY) / scale / ih;
    if (imgYNorm < by || imgYNorm > by + bh) continue;
    for (let px = 0; px < protoW; px++) {
      const imgXNorm = ((px + 0.5) * (inputSize / protoW) - padX) / scale / iw;
      if (imgXNorm < bx || imgXNorm > bx + bw) continue;
      let acc = 0;
      const cell = py * protoW + px;
      for (let c = 0; c < protoC; c++) {
        acc += coeffs[c] * protos[c * area + cell];
      }
      if (sigmoid(acc) > 0.5) {
        out[cell] = 1;
      }
    }
  }
  return out;
}

export async function runImageYoloRaw(
  onnxPath: string,
  imageSrc: string,
  inputSize: number,
  classes: string[],
  isSeg: boolean,
  floorConf = 0.08
): Promise<{ candidates: EvalBoxDetection[]; latencyMs: number; width: number; height: number }> {
  const t0 = performance.now();
  const session = await getOrtSession(onnxPath);
  const img = await loadHtmlImage(imageSrc);
  const iw = img.naturalWidth || img.width;
  const ih = img.naturalHeight || img.height;
  if (iw <= 0 || ih <= 0) {
    throw new Error("Image has zero dimensions for evaluation");
  }

  const { tensor, scale, padX, padY } = await preprocessLetterbox(img, inputSize);
  let outputs: Record<string, OrtTypes.Tensor> | null = null;
  let dims: readonly number[] = [];
  let data: Float32Array;
  let protoDims: readonly number[] = [];
  let protoData: Float32Array | null = null;
  try {
    outputs = await session.run({ [session.inputNames[0]]: tensor });
    const out0 = outputs[session.outputNames[0]];
    dims = [...out0.dims];
    data = Float32Array.from(out0.data as Float32Array);

    const protoTensor = isSeg && session.outputNames.length > 1 ? outputs[session.outputNames[1]] : null;
    if (protoTensor) {
      protoDims = [...protoTensor.dims];
      protoData = Float32Array.from(protoTensor.data as Float32Array);
    }
  } finally {
    try {
      tensor.dispose();
    } catch {}
    if (outputs) {
      for (const outT of Object.values(outputs)) {
        try {
          outT.dispose();
        } catch {}
      }
    }
  }
  const protoC = protoDims.length === 4 ? protoDims[1] : 0;
  const protoH = protoDims.length === 4 ? protoDims[2] : 0;
  const protoW = protoDims.length === 4 ? protoDims[3] : 0;

  const labelNames = classes.length > 0 ? classes : COCO_CLASSES;
  const candidates: EvalBoxDetection[] = [];

  if (dims.length === 3 && dims[2] === 6) {
    const count = dims[1];
    for (let i = 0; i < count; i++) {
      const off = i * 6;
      const score = data[off + 4];
      if (score < floorConf) continue;
      const x1 = Math.max(0, (data[off] - padX) / scale / iw);
      const y1 = Math.max(0, (data[off + 1] - padY) / scale / ih);
      const x2 = Math.min(1, (data[off + 2] - padX) / scale / iw);
      const y2 = Math.min(1, (data[off + 3] - padY) / scale / ih);
      const clsIdx = Math.round(data[off + 5]);
      candidates.push({
        x: x1,
        y: y1,
        w: Math.max(0.005, x2 - x1),
        h: Math.max(0.005, y2 - y1),
        score,
        classIndex: clsIdx,
        className: labelNames[clsIdx] ?? `class_${clsIdx}`,
      });
    }
  } else if (dims.length === 3) {
    const channelFirst = dims[1] <= 256 && dims[2] > dims[1];
    const channels = channelFirst ? dims[1] : dims[2];
    const anchors = channelFirst ? dims[2] : dims[1];
    const maskChannels = protoC > 0 ? protoC : 0;
    const numClasses = Math.max(1, channels - 4 - maskChannels);
    const readVal = (c: number, a: number) =>
      channelFirst ? data[c * anchors + a] : data[a * channels + c];

    for (let a = 0; a < anchors; a++) {
      let bestCls = 0;
      let bestScore = -1;
      for (let c = 0; c < numClasses; c++) {
        const s = readVal(4 + c, a);
        if (s > bestScore) {
          bestScore = s;
          bestCls = c;
        }
      }
      if (bestScore < floorConf) continue;

      const cx = readVal(0, a);
      const cy = readVal(1, a);
      const bw = readVal(2, a);
      const bh = readVal(3, a);
      const sf = cx <= 1.5 && cy <= 1.5 && bw <= 1.5 ? inputSize : 1;
      const x1 = Math.max(0, (cx * sf - (bw * sf) / 2 - padX) / scale / iw);
      const y1 = Math.max(0, (cy * sf - (bh * sf) / 2 - padY) / scale / ih);
      const nw = Math.min(1 - x1, (bw * sf) / scale / iw);
      const nh = Math.min(1 - y1, (bh * sf) / scale / ih);
      if (nw <= 0.004 || nh <= 0.004) continue;

      let maskData: Uint8Array | undefined;
      if (protoData && protoC > 0) {
        const coeffs: number[] = [];
        for (let mc = 0; mc < protoC; mc++) {
          coeffs.push(readVal(4 + numClasses + mc, a));
        }
        maskData = decodeProtoMask(
          coeffs,
          protoData,
          protoC,
          protoH,
          protoW,
          x1,
          y1,
          nw,
          nh,
          iw,
          ih,
          inputSize,
          scale,
          padX,
          padY
        );
      }

      candidates.push({
        x: x1,
        y: y1,
        w: nw,
        h: nh,
        score: bestScore,
        classIndex: bestCls,
        className: labelNames[bestCls] ?? `class_${bestCls}`,
        maskData,
        maskW: protoW || undefined,
        maskH: protoH || undefined,
      });
    }
  }

  return {
    candidates,
    latencyMs: Math.round(performance.now() - t0),
    width: iw,
    height: ih,
  };
}

export async function runImageClassification(
  onnxPath: string,
  imageSrc: string,
  inputSize: number,
  classes: string[]
): Promise<{ scores: EvalClassificationScore[]; latencyMs: number; width: number; height: number }> {
  const t0 = performance.now();
  const session = await getOrtSession(onnxPath);
  const img = await loadHtmlImage(imageSrc);
  const tensor = await preprocessImageNet(img, inputSize);
  let outputs: Record<string, OrtTypes.Tensor> | null = null;
  let logits: Float32Array;
  try {
    outputs = await session.run({ [session.inputNames[0]]: tensor });
    logits = Float32Array.from(outputs[session.outputNames[0]].data as Float32Array);
  } finally {
    try {
      tensor.dispose();
    } catch {}
    if (outputs) {
      for (const outT of Object.values(outputs)) {
        try {
          outT.dispose();
        } catch {}
      }
    }
  }
  const probs = softmax(logits);
  const scores = probs
    .map((score, i) => ({
      label: classes[i] ?? `class_${i}`,
      score,
    }))
    .sort((a, b) => b.score - a.score);

  return {
    scores,
    latencyMs: Math.round(performance.now() - t0),
    width: img.naturalWidth || img.width,
    height: img.naturalHeight || img.height,
  };
}

export async function runAudioMelClassification(
  onnxPath: string,
  decoded: DecodedAudioData,
  classes: string[]
): Promise<{ scores: EvalClassificationScore[]; latencyMs: number }> {
  const t0 = performance.now();
  const ort = await ensureOrt();
  const session = await getOrtSession(onnxPath);
  const patch = extractLogMelPatches(decoded.mono, decoded.sampleRate, 64, 96);
  const tensor = new ort.Tensor("float32", patch, [1, 1, 96, 64]);
  let outputs: Record<string, OrtTypes.Tensor> | null = null;
  let logits: Float32Array;
  try {
    outputs = await session.run({ [session.inputNames[0]]: tensor });
    logits = Float32Array.from(outputs[session.outputNames[0]].data as Float32Array);
  } finally {
    try {
      tensor.dispose();
    } catch {}
    if (outputs) {
      for (const outT of Object.values(outputs)) {
        try {
          outT.dispose();
        } catch {}
      }
    }
  }
  const probs = softmax(logits);
  const scores = probs
    .map((score, i) => ({
      label: classes[i] ?? `class_${i}`,
      score,
    }))
    .sort((a, b) => b.score - a.score);
  return {
    scores,
    latencyMs: Math.round(performance.now() - t0),
  };
}

export async function runWhisperTranscription(
  onnxDir: string,
  onnxPath: string,
  decoded: DecodedAudioData
): Promise<{ text: string; chunks: EvalTranscriptChunk[]; latencyMs: number }> {
  const t0 = performance.now();
  const asr = await getHfPipeline("automatic-speech-recognition", onnxDir, onnxPath);
  const audio16k = resampleTo16kHz(decoded.mono, decoded.sampleRate);
  const out = (await asr(audio16k, { return_timestamps: true })) as {
    text?: string;
    chunks?: { text: string; timestamp?: [number | null, number | null] }[];
  };
  const chunks: EvalTranscriptChunk[] = (out.chunks ?? []).map((c) => ({
    text: (c.text ?? "").trim(),
    startSec: c.timestamp?.[0] ?? null,
    endSec: c.timestamp?.[1] ?? null,
  }));
  return {
    text: (out.text ?? "").trim(),
    chunks,
    latencyMs: Math.round(performance.now() - t0),
  };
}

export async function runDistilBertIntent(
  onnxDir: string,
  onnxPath: string,
  text: string
): Promise<{ scores: EvalClassificationScore[]; latencyMs: number }> {
  const t0 = performance.now();
  const clf = await getHfPipeline("text-classification", onnxDir, onnxPath);
  const raw = (await clf(text, { top_k: null })) as
    | { label: string; score: number }[]
    | { label: string; score: number };
  const arr = Array.isArray(raw) ? raw : [raw];
  const scores = arr
    .map((r) => ({ label: r.label, score: r.score }))
    .sort((a, b) => b.score - a.score);
  return {
    scores,
    latencyMs: Math.round(performance.now() - t0),
  };
}

export async function runQwenGeneration(
  onnxDir: string,
  onnxPath: string,
  prompt: string,
  systemPrompt: string,
  maxNewTokens: number,
  temperature: number
): Promise<{ text: string; latencyMs: number; tokPerSec: number }> {
  const t0 = performance.now();
  const gen = await getHfPipeline("text-generation", onnxDir, onnxPath);
  const formatted = systemPrompt.trim()
    ? `System: ${systemPrompt.trim()}\nUser: ${prompt.trim()}\nAssistant:`
    : `User: ${prompt.trim()}\nAssistant:`;
  const out = (await gen(formatted, {
    max_new_tokens: maxNewTokens,
    temperature,
    do_sample: temperature > 0.05,
    return_full_text: false,
  })) as { generated_text?: string }[];
  const text = (out?.[0]?.generated_text ?? "").trim();
  const ms = Math.max(1, Math.round(performance.now() - t0));
  const approxToks = Math.max(1, Math.ceil(text.length / 4));
  return {
    text,
    latencyMs: ms,
    tokPerSec: Number(((approxToks * 1000) / ms).toFixed(1)),
  };
}
