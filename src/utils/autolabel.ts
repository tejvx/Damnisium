import type * as OrtTypes from "onnxruntime-web/wasm";
import { api } from "../api";
import { COCO_CLASSES } from "../constants";
import type { BoundingBox, LabelClass } from "../types";
import { randomId } from "./format";

let ortMod: typeof OrtTypes | null = null;
let cachedModelPath: string | null = null;
let cachedSession: OrtTypes.InferenceSession | null = null;

async function ensureOrt(): Promise<typeof OrtTypes> {
  if (!ortMod) {
    ortMod = await import("onnxruntime-web/wasm");
    ortMod.env.wasm.numThreads = 1;
    ortMod.env.wasm.wasmPaths = "/ort/";
  }
  return ortMod;
}

export interface RawDetection {
  x: number;
  y: number;
  w: number;
  h: number;
  score: number;
  classIndex: number;
  className: string;
}

export async function disposeAutoLabelSession(): Promise<void> {
  if (cachedSession) {
    try {
      await cachedSession.release();
    } catch { }
    cachedSession = null;
    cachedModelPath = null;
  }
}

async function getSession(localModelPath: string): Promise<OrtTypes.InferenceSession> {
  const ort = await ensureOrt();
  if (cachedSession && cachedModelPath === localModelPath) {
    return cachedSession;
  }
  await disposeAutoLabelSession();
  const bytes = await api.readBinaryFile(localModelPath);
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ["wasm"],
  });
  cachedModelPath = localModelPath;
  cachedSession = session;
  return session;
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
  if (!ctx) {
    throw new Error("2D context unavailable for ONNX preprocessing");
  }

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
  const tensor = new ort.Tensor("float32", floatData, [1, 3, size, size]);
  return { tensor, scale, padX, padY };
}

function boxIou(a: RawDetection, b: RawDetection): number {
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

function applyNms(dets: RawDetection[], iouThresh: number): RawDetection[] {
  const sorted = [...dets].sort((a, b) => b.score - a.score);
  const kept: RawDetection[] = [];
  for (const cand of sorted) {
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

function loadHtmlImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed loading image for auto-label"));
    img.src = src;
  });
}

export async function runAutoLabelOnImage(
  localModelPath: string,
  imageSrc: string,
  inputSize: number,
  confThresh: number,
  iouThresh: number,
  projectClasses: LabelClass[],
  modelClasses: string[] = []
): Promise<{ detections: RawDetection[]; mappedBoxes: BoundingBox[]; missingClassNames: string[] }> {
  const session = await getSession(localModelPath);
  const img = await loadHtmlImage(imageSrc);
  const iw = img.naturalWidth || img.width;
  const ih = img.naturalHeight || img.height;
  if (iw <= 0 || ih <= 0) {
    throw new Error("Image has zero dimensions for auto-labeling");
  }

  const resolveClassName = (clsIdx: number): string =>
    modelClasses.length > 0
      ? (modelClasses[clsIdx] ?? `class_${clsIdx}`)
      : (COCO_CLASSES[clsIdx] ?? `class_${clsIdx}`);

  const { tensor, scale, padX, padY } = await preprocessLetterbox(img, inputSize);
  let outputs: Record<string, OrtTypes.Tensor> | null = null;
  let dims: readonly number[] = [];
  let data: Float32Array;
  try {
    const inputName = session.inputNames[0];
    outputs = await session.run({ [inputName]: tensor });
    const outputTensor = outputs[session.outputNames[0]];
    dims = [...outputTensor.dims];
    data = Float32Array.from(outputTensor.data as Float32Array);
  } finally {
    try {
      tensor.dispose();
    } catch { }
    if (outputs) {
      for (const outT of Object.values(outputs)) {
        try {
          outT.dispose();
        } catch { }
      }
    }
  }

  const raw: RawDetection[] = [];

  if (dims.length === 3 && dims[2] === 6) {
    const count = dims[1];
    for (let i = 0; i < count; i++) {
      const off = i * 6;
      const score = data[off + 4];
      if (score < confThresh) continue;
      const x1 = (data[off] - padX) / scale / iw;
      const y1 = (data[off + 1] - padY) / scale / ih;
      const x2 = (data[off + 2] - padX) / scale / iw;
      const y2 = (data[off + 3] - padY) / scale / ih;
      const clsIdx = Math.round(data[off + 5]);
      raw.push({
        x: Math.max(0, x1),
        y: Math.max(0, y1),
        w: Math.max(0.005, Math.min(1, x2) - Math.max(0, x1)),
        h: Math.max(0.005, Math.min(1, y2) - Math.max(0, y1)),
        score,
        classIndex: clsIdx,
        className: resolveClassName(clsIdx),
      });
    }
  } else if (dims.length === 3) {
    const channelFirst = dims[1] <= 128 && dims[2] > dims[1];
    const channels = channelFirst ? dims[1] : dims[2];
    const anchors = channelFirst ? dims[2] : dims[1];
    const numClasses = Math.max(1, channels - 4);

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
      if (bestScore < confThresh) continue;

      const cx = readVal(0, a);
      const cy = readVal(1, a);
      const bw = readVal(2, a);
      const bh = readVal(3, a);
      const scaleFactor = cx <= 1.5 && cy <= 1.5 && bw <= 1.5 ? inputSize : 1;

      const x1 = ((cx * scaleFactor - (bw * scaleFactor) / 2) - padX) / scale / iw;
      const y1 = ((cy * scaleFactor - (bh * scaleFactor) / 2) - padY) / scale / ih;
      const nw = (bw * scaleFactor) / scale / iw;
      const nh = (bh * scaleFactor) / scale / ih;

      const nx = Math.max(0, Math.min(0.99, x1));
      const ny = Math.max(0, Math.min(0.99, y1));
      raw.push({
        x: nx,
        y: ny,
        w: Math.max(0.005, Math.min(1 - nx, nw)),
        h: Math.max(0.005, Math.min(1 - ny, nh)),
        score: bestScore,
        classIndex: bestCls,
        className: resolveClassName(bestCls),
      });
    }
  }

  const kept = applyNms(raw, iouThresh);
  const classLookup = new Map<string, LabelClass>();
  for (const c of projectClasses) {
    classLookup.set(c.name.trim().toLowerCase(), c);
  }

  const missingSet = new Set<string>();
  const mappedBoxes: BoundingBox[] = [];
  for (const det of kept) {
    const matched = classLookup.get(det.className.toLowerCase());
    if (!matched) {
      missingSet.add(det.className);
    }
    mappedBoxes.push({
      id: randomId("box"),
      classId: matched ? matched.id : `pending:${det.className}`,
      x: det.x,
      y: det.y,
      w: det.w,
      h: det.h,
      confidence: Number(det.score.toFixed(3)),
    });
  }

  return {
    detections: kept,
    mappedBoxes,
    missingClassNames: Array.from(missingSet),
  };
}
