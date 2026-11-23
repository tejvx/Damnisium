import { api } from "../../api";
import type { EvalBundleInfo } from "../../types";
import {
  decodeAudioBytes,
  type DecodedAudioData,
  playAudioSegment,
  startMicRecorder,
} from "../../utils/audio-dsp";
import { el, showToast, svgIcon } from "../../utils/dom";
import { formatTime } from "../../utils/format";
import {
  type EvalClassificationScore,
  type EvalTranscriptChunk,
  runAudioMelClassification,
  runWhisperTranscription,
} from "../../utils/test-runtime";

interface AudioEvalResult {
  id: string;
  filename: string;
  decoded: DecodedAudioData;
  latencyMs: number;
  scores: EvalClassificationScore[];
  transcript: string;
  chunks: EvalTranscriptChunk[];
}

let results: AudioEvalResult[] = [];
let activeIdx = 0;
let activePlay: { stop: () => void } | null = null;
let micControl: { stop: () => Promise<DecodedAudioData> } | null = null;
let batchRunning = false;
let batchCancel = false;
let waveResizeObs: ResizeObserver | null = null;

export function resetAudioEvalState(): void {
  if (waveResizeObs) {
    waveResizeObs.disconnect();
    waveResizeObs = null;
  }
  if (activePlay) {
    activePlay.stop();
    activePlay = null;
  }
  if (micControl) {
    const mc = micControl;
    micControl = null;
    void mc.stop().catch(() => {});
  }
  results = [];
  activeIdx = 0;
  batchRunning = false;
  batchCancel = true;
}

function drawWaveform(canvas: HTMLCanvasElement, decoded: DecodedAudioData | null): void {
  const ctx = canvas.getContext("2d");
  if (!ctx || !canvas.parentElement) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, canvas.parentElement.clientWidth);
  const h = Math.max(1, canvas.parentElement.clientHeight);
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#09090b";
  ctx.fillRect(0, 0, w, h);

  if (!decoded) {
    ctx.fillStyle = "#71717a";
    ctx.font = "13px Inter, Segoe UI, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Upload audio, test a folder, or record from microphone", w / 2, h / 2);
    return;
  }

  const mono = decoded.mono;
  const step = Math.max(1, Math.floor(mono.length / w));
  const mid = h / 2;
  ctx.strokeStyle = "#ec4899";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < w; x++) {
    const start = Math.floor((x / w) * mono.length);
    let min = 1;
    let max = -1;
    for (let k = 0; k < step; k++) {
      const s = mono[start + k] ?? 0;
      if (s < min) min = s;
      if (s > max) max = s;
    }
    ctx.moveTo(x, mid + min * (h * 0.42));
    ctx.lineTo(x, mid + max * (h * 0.42));
  }
  ctx.stroke();
}

export function renderAudioEvalPane(
  container: HTMLElement,
  bundle: EvalBundleInfo
): void {
  if (waveResizeObs) {
    waveResizeObs.disconnect();
    waveResizeObs = null;
  }
  container.innerHTML = "";
  const isAsr = bundle.runtimeKind === "hf_asr";
  if (activeIdx >= results.length) activeIdx = 0;
  const activeItem = results[activeIdx] ?? null;

  const leftPanel = el("div", { class: "studio-left" });
  const uploadBtn = el(
    "button",
    { class: "btn btn-sm btn-primary", type: "button", disabled: batchRunning },
    [svgIcon("upload", 12), "Audio Files"]
  );
  const folderBtn = el(
    "button",
    { class: "btn btn-sm", type: "button", disabled: batchRunning },
    [svgIcon("folder", 12), "Folder"]
  );
  const micBtn = el("button", { class: "btn btn-sm", type: "button", disabled: batchRunning }, [
    svgIcon("mic", 12),
    micControl ? "Stop Mic" : "Record Mic",
  ]);
  const stopBatchBtn = el(
    "button",
    {
      class: "btn btn-sm btn-danger",
      type: "button",
      style: batchRunning ? "" : "display:none",
    },
    ["Stop"]
  );
  stopBatchBtn.onclick = () => {
    batchCancel = true;
  };

  const progressText = el("div", { class: "dim mono", style: "font-size:11px" });

  const evaluateDecoded = async (
    filename: string,
    decoded: DecodedAudioData,
    idx: number
  ): Promise<AudioEvalResult> => {
    if (isAsr) {
      if (!bundle.onnxDir) {
        throw new Error("Whisper ONNX bundle directory missing.");
      }
      const out = await runWhisperTranscription(bundle.onnxDir, bundle.onnxPath, decoded);
      return {
        id: `${idx}_${filename}`,
        filename,
        decoded,
        latencyMs: out.latencyMs,
        scores: [],
        transcript: out.text,
        chunks: out.chunks,
      };
    }
    if (!bundle.onnxPath) {
      throw new Error("Audio ONNX model missing.");
    }
    const out = await runAudioMelClassification(bundle.onnxPath, decoded, bundle.classes);
    return {
      id: `${idx}_${filename}`,
      filename,
      decoded,
      latencyMs: out.latencyMs,
      scores: out.scores,
      transcript: "",
      chunks: [],
    };
  };

  const runBatchFiles = async (paths: string[]) => {
    batchRunning = true;
    batchCancel = false;
    results = [];
    activeIdx = 0;
    uploadBtn.setAttribute("disabled", "true");
    folderBtn.setAttribute("disabled", "true");
    micBtn.setAttribute("disabled", "true");
    stopBatchBtn.style.display = "inline-flex";
    try {
      for (let i = 0; i < paths.length; i++) {
        if (batchCancel) break;
        progressText.textContent = `Evaluating ${i + 1}/${paths.length}...`;
        const bytes = await api.readBinaryFile(paths[i]);
        const decoded = await decodeAudioBytes(bytes);
        const filename = paths[i].split(/[\\/]/).pop() ?? paths[i];
        results.push(await evaluateDecoded(filename, decoded, i));
      }
    } catch (err) {
      showToast(String(err), "error");
    } finally {
      batchRunning = false;
      if (container.isConnected) {
        renderAudioEvalPane(container, bundle);
      }
    }
  };

  uploadBtn.onclick = async () => {
    const sel = await api.pickFiles(
      [{ name: "Audio Files", extensions: ["wav", "mp3", "ogg", "flac", "m4a", "aac"] }],
      true
    );
    const paths = Array.isArray(sel) ? sel : typeof sel === "string" ? [sel] : [];
    if (paths.length > 0) void runBatchFiles(paths);
  };

  folderBtn.onclick = async () => {
    const folder = await api.pickFolder();
    if (typeof folder !== "string") return;
    const paths = await api.listFolderMedia(folder, "audio");
    if (paths.length === 0) {
      showToast("No audio files found in folder.", "info");
      return;
    }
    void runBatchFiles(paths);
  };

  micBtn.onclick = async () => {
    if (micControl) {
      try {
        const decoded = await micControl.stop();
        micControl = null;
        const item = await evaluateDecoded("mic_capture.wav", decoded, results.length);
        results = [item, ...results];
        activeIdx = 0;
        renderAudioEvalPane(container, bundle);
      } catch (err) {
        micControl = null;
        showToast(String(err), "error");
      }
      return;
    }
    try {
      micControl = await startMicRecorder();
      renderAudioEvalPane(container, bundle);
      showToast("Recording microphone... click Stop Mic to evaluate.", "info");
    } catch (err) {
      showToast(String(err), "error");
    }
  };

  const queue = el("div", { style: "flex:1;overflow-y:auto" });
  results.forEach((item, idx) => {
    const summary = isAsr
      ? item.transcript.slice(0, 32) || "No speech"
      : `${item.scores[0]?.label ?? "--"} (${Math.round((item.scores[0]?.score ?? 0) * 100)}%)`;
    const row = el(
      "div",
      { class: `audio-queue-item${idx === activeIdx ? " active" : ""}` },
      [
        el("div", { style: "min-width:0" }, [
          el("div", {
            style: "font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap",
          }, [item.filename]),
          el("div", { class: "dim mono", style: "font-size:10px" }, [summary]),
        ]),
        el("span", { class: "badge mono" }, [`${item.latencyMs}ms`]),
      ]
    );
    row.onclick = () => {
      activeIdx = idx;
      renderAudioEvalPane(container, bundle);
    };
    queue.appendChild(row);
  });

  leftPanel.append(
    el("div", { class: "panel-section" }, [
      el("div", { class: "panel-section-title" }, ["Audio Source"]),
      el("div", { style: "display:flex;gap:6px;flex-wrap:wrap" }, [
        uploadBtn,
        folderBtn,
        micBtn,
        stopBatchBtn,
      ]),
      progressText,
    ]),
    queue
  );

  const centerPanel = el("div", { class: "audio-center" });
  const playBtn = el("button", {
    class: "btn btn-sm btn-primary",
    type: "button",
    disabled: !activeItem,
  }, [svgIcon("play", 12), "Play Clip"]);
  const stopBtn = el("button", { class: "btn btn-sm", type: "button", disabled: !activeItem }, [
    "Stop",
  ]);
  playBtn.onclick = () => {
    if (!activeItem) return;
    if (activePlay) activePlay.stop();
    activePlay = playAudioSegment(
      activeItem.decoded,
      0,
      activeItem.decoded.durationSec,
      null,
      () => {
        activePlay = null;
      }
    );
  };
  stopBtn.onclick = () => {
    if (activePlay) {
      activePlay.stop();
      activePlay = null;
    }
  };

  const waveCanvas = el("canvas") as HTMLCanvasElement;
  const wavePane = el("div", { class: "audio-pane", style: "flex:1" }, [
    waveCanvas,
    el("div", { class: "audio-pane-label" }, ["WAVEFORM"]),
  ]);
  waveResizeObs = new ResizeObserver(() => {
    drawWaveform(waveCanvas, activeItem?.decoded ?? null);
  });
  waveResizeObs.observe(wavePane);
  centerPanel.append(
    el("div", { class: "audio-transport" }, [
      el("div", { style: "display:flex;gap:6px" }, [playBtn, stopBtn]),
      el("span", { class: "mono dim", style: "font-size:11px" }, [
        activeItem
          ? `${formatTime(activeItem.decoded.durationSec)} · ${activeItem.decoded.sampleRate} Hz · ${activeItem.latencyMs} ms`
          : "No clip loaded",
      ]),
    ]),
    wavePane
  );

  const rightPanel = el("div", { class: "studio-right" });
  const predSection = el("div", { class: "panel-section" });
  if (!activeItem) {
    predSection.appendChild(
      el("div", { class: "dim", style: "font-size:12px" }, ["No audio predictions yet"])
    );
  } else if (isAsr) {
    predSection.appendChild(
      el("div", { class: "panel-section-title" }, [
        "Whisper Transcription",
        el("span", { class: "mono" }, [`${activeItem.latencyMs} ms`]),
      ])
    );
    predSection.appendChild(
      el("div", { class: "chat-bubble", style: "margin-bottom:8px" }, [
        activeItem.transcript || "(Empty transcript)",
      ])
    );
    for (const ch of activeItem.chunks) {
      const ts =
        ch.startSec !== null && ch.endSec !== null
          ? `${formatTime(ch.startSec)} -> ${formatTime(ch.endSec)}`
          : "Segment";
      const row = el("div", { class: "transcript-row" }, [
        el("span", { class: "mono", style: "color:var(--rose);font-size:11px;white-space:nowrap" }, [
          ts,
        ]),
        el("span", { style: "font-size:12px" }, [ch.text]),
      ]);
      row.onclick = () => {
        if (ch.startSec !== null) {
          if (activePlay) activePlay.stop();
          activePlay = playAudioSegment(
            activeItem.decoded,
            ch.startSec,
            ch.endSec ?? activeItem.decoded.durationSec,
            null,
            () => {
              activePlay = null;
            }
          );
        }
      };
      predSection.appendChild(row);
    }
  } else {
    predSection.appendChild(
      el("div", { class: "panel-section-title" }, [
        "Predicted Audio Classes",
        el("span", { class: "mono" }, [`${activeItem.latencyMs} ms`]),
      ])
    );
    const list = el("div", { class: "eval-conf-list" });
    for (const s of activeItem.scores.slice(0, 10)) {
      const pct = Math.round(s.score * 100);
      list.appendChild(
        el("div", { class: "eval-conf-row" }, [
          el("div", { class: "eval-conf-head" }, [
            el("span", {}, [s.label]),
            el("span", { class: "mono" }, [`${pct}%`]),
          ]),
          el("div", { class: "eval-conf-track" }, [
            el("div", { class: "eval-conf-fill", style: `width:${pct}%` }),
          ]),
        ])
      );
    }
    predSection.appendChild(list);
  }
  rightPanel.appendChild(predSection);

  container.appendChild(
    el("div", { class: "eval-split-grid" }, [leftPanel, centerPanel, rightPanel])
  );
  requestAnimationFrame(() => drawWaveform(waveCanvas, activeItem?.decoded ?? null));
}
