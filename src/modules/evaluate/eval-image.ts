import { api } from "../../api";
import { CLASS_PALETTE } from "../../constants";
import { getState, setState } from "../../state";
import type { EvalBundleInfo } from "../../types";
import { el, showToast, svgIcon } from "../../utils/dom";
import {
  type EvalBoxDetection,
  type EvalClassificationScore,
  filterDetectionsLive,
  runImageClassification,
  runImageYoloRaw,
} from "../../utils/test-runtime";

interface ImageEvalResult {
  id: string;
  filename: string;
  absPath: string;
  srcUrl: string;
  width: number;
  height: number;
  latencyMs: number;
  rawCandidates: EvalBoxDetection[];
  clsScores: EvalClassificationScore[];
}

let results: ImageEvalResult[] = [];
let activeIdx = 0;
let confThresh = 0.35;
let iouThresh = 0.45;
let showLabels = true;
let showConfidence = true;
let classFilter = "all";
let batchRunning = false;
let batchCancel = false;
const imageCache = new Map<string, HTMLImageElement>();
let stageResizeObs: ResizeObserver | null = null;
let drawSeq = 0;

export function resetImageEvalState(): void {
  if (stageResizeObs) {
    stageResizeObs.disconnect();
    stageResizeObs = null;
  }
  imageCache.clear();
  results = [];
  activeIdx = 0;
  batchRunning = false;
  batchCancel = true;
}

function getFilteredDets(item: ImageEvalResult): EvalBoxDetection[] {
  return filterDetectionsLive(item.rawCandidates, confThresh, iouThresh).filter(
    (d) => classFilter === "all" || d.className === classFilter
  );
}

function renderLoadedImageToCanvas(
  canvas: HTMLCanvasElement,
  img: HTMLImageElement,
  item: ImageEvalResult,
  isCls: boolean
): void {
  const ctx = canvas.getContext("2d");
  const parent = canvas.parentElement;
  if (!ctx || !parent) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cw = Math.max(1, parent.clientWidth);
  const ch = Math.max(1, parent.clientHeight);
  canvas.width = Math.round(cw * dpr);
  canvas.height = Math.round(ch * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  ctx.fillStyle = "#09090b";
  ctx.fillRect(0, 0, cw, ch);

  if (img.naturalWidth <= 0 || img.naturalHeight <= 0) return;
  const pad = 20;
  const scale = Math.min((cw - pad * 2) / img.naturalWidth, (ch - pad * 2) / img.naturalHeight);
  const dw = img.naturalWidth * scale;
  const dh = img.naturalHeight * scale;
  const ox = (cw - dw) / 2;
  const oy = (ch - dh) / 2;
  ctx.drawImage(img, ox, oy, dw, dh);

  if (isCls) return;

  const filtered = getFilteredDets(item);
  for (const det of filtered) {
    const color = CLASS_PALETTE[det.classIndex % CLASS_PALETTE.length];
    if (det.maskData && det.maskW && det.maskH) {
      const off = document.createElement("canvas");
      off.width = det.maskW;
      off.height = det.maskH;
      const octx = off.getContext("2d");
      if (octx) {
        const imgD = octx.createImageData(det.maskW, det.maskH);
        const hex = color.replace("#", "");
        const r = parseInt(hex.slice(0, 2), 16);
        const g = parseInt(hex.slice(2, 4), 16);
        const b = parseInt(hex.slice(4, 6), 16);
        for (let i = 0; i < det.maskData.length; i++) {
          if (det.maskData[i]) {
            const p = i * 4;
            imgD.data[p] = r;
            imgD.data[p + 1] = g;
            imgD.data[p + 2] = b;
            imgD.data[p + 3] = 96;
          }
        }
        octx.putImageData(imgD, 0, 0);
        ctx.drawImage(off, ox, oy, dw, dh);
      }
    }

    const x = ox + det.x * dw;
    const y = oy + det.y * dh;
    const w = det.w * dw;
    const h = det.h * dh;

    ctx.fillStyle = `${color}22`;
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    if (showLabels || showConfidence) {
      const parts: string[] = [];
      if (showLabels) parts.push(det.className);
      if (showConfidence) parts.push(`${Math.round(det.score * 100)}%`);
      const text = parts.join(" ");
      ctx.font = "600 11px Inter, Segoe UI, sans-serif";
      const tw = ctx.measureText(text).width + 10;
      const ty = Math.max(0, y - 18);
      ctx.fillStyle = color;
      ctx.fillRect(x, ty, tw, 18);
      ctx.fillStyle = "#09090b";
      ctx.textAlign = "left";
      ctx.fillText(text, x + 5, ty + 13);
    }
  }
}

function drawResultCanvas(
  canvas: HTMLCanvasElement,
  item: ImageEvalResult | null,
  isCls: boolean
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const parent = canvas.parentElement;
  if (!parent) return;
  const seq = ++drawSeq;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cw = Math.max(1, parent.clientWidth);
  const ch = Math.max(1, parent.clientHeight);
  if (!item) {
    canvas.width = Math.round(cw * dpr);
    canvas.height = Math.round(ch * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    ctx.fillStyle = "#09090b";
    ctx.fillRect(0, 0, cw, ch);
    ctx.fillStyle = "#71717a";
    ctx.font = "13px Inter, Segoe UI, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Upload an image or choose a folder to run evaluation", cw / 2, ch / 2);
    return;
  }

  const cached = imageCache.get(item.srcUrl);
  if (cached && cached.complete && cached.naturalWidth > 0) {
    renderLoadedImageToCanvas(canvas, cached, item, isCls);
    return;
  }

  const img = new Image();
  img.onload = () => {
    if (imageCache.size >= 24) {
      const oldest = imageCache.keys().next().value;
      if (oldest) imageCache.delete(oldest);
    }
    imageCache.set(item.srcUrl, img);
    if (seq !== drawSeq || !canvas.isConnected) return;
    renderLoadedImageToCanvas(canvas, img, item, isCls);
  };
  img.src = item.srcUrl;
}

export function renderImageEvalPane(
  container: HTMLElement,
  bundle: EvalBundleInfo
): void {
  if (stageResizeObs) {
    stageResizeObs.disconnect();
    stageResizeObs = null;
  }
  container.innerHTML = "";
  const isCls = bundle.runtimeKind === "ort_img_cls";
  const isSeg = bundle.runtimeKind === "ort_yolo_seg";

  const visibleResults = results.filter((r) => {
    if (classFilter === "all") return true;
    if (isCls) return r.clsScores[0]?.label === classFilter;
    return filterDetectionsLive(r.rawCandidates, confThresh, iouThresh).some(
      (d) => d.className === classFilter
    );
  });

  if (activeIdx >= visibleResults.length) {
    activeIdx = 0;
  }
  const activeItem = visibleResults[activeIdx] ?? null;

  const leftPanel = el("div", { class: "studio-left" });
  const uploadImgBtn = el(
    "button",
    { class: "btn btn-sm btn-primary", type: "button", disabled: batchRunning },
    [svgIcon("upload", 12), "Test Images"]
  );
  const folderBtn = el(
    "button",
    { class: "btn btn-sm", type: "button", disabled: batchRunning },
    [svgIcon("folder", 12), "Test Folder"]
  );
  const cancelBtn = el("button", {
    class: "btn btn-sm btn-danger",
    type: "button",
    style: batchRunning ? "" : "display:none",
  }, ["Stop"]);
  cancelBtn.onclick = () => {
    batchCancel = true;
  };

  const progressText = el("div", { class: "dim mono", style: "font-size:11px" });

  const runBatchPaths = async (paths: string[]) => {
    if (!bundle.onnxPath) {
      showToast("No ONNX model found in selected bundle.", "error");
      return;
    }
    batchRunning = true;
    batchCancel = false;
    results = [];
    activeIdx = 0;
    uploadImgBtn.setAttribute("disabled", "true");
    folderBtn.setAttribute("disabled", "true");
    cancelBtn.style.display = "inline-flex";
    try {
      for (let i = 0; i < paths.length; i++) {
        if (batchCancel) break;
        const abs = paths[i];
        progressText.textContent = `Running ${i + 1}/${paths.length}...`;
        const srcUrl = api.fileSrc(abs);
        const filename = abs.split(/[\\/]/).pop() ?? abs;
        if (isCls) {
          const out = await runImageClassification(
            bundle.onnxPath,
            srcUrl,
            bundle.inputSize || 224,
            bundle.classes
          );
          results.push({
            id: `${i}_${filename}`,
            filename,
            absPath: abs,
            srcUrl,
            width: out.width,
            height: out.height,
            latencyMs: out.latencyMs,
            rawCandidates: [],
            clsScores: out.scores,
          });
        } else {
          const out = await runImageYoloRaw(
            bundle.onnxPath,
            srcUrl,
            bundle.inputSize || 640,
            bundle.classes,
            isSeg
          );
          results.push({
            id: `${i}_${filename}`,
            filename,
            absPath: abs,
            srcUrl,
            width: out.width,
            height: out.height,
            latencyMs: out.latencyMs,
            rawCandidates: out.candidates,
            clsScores: [],
          });
        }
      }
    } catch (err) {
      showToast(String(err), "error");
    } finally {
      batchRunning = false;
      if (container.isConnected) {
        renderImageEvalPane(container, bundle);
      }
    }
  };

  uploadImgBtn.onclick = async () => {
    const sel = await api.pickFiles(
      [{ name: "Images", extensions: ["png", "jpg", "jpeg", "webp", "bmp", "gif"] }],
      true
    );
    const paths = Array.isArray(sel) ? sel : typeof sel === "string" ? [sel] : [];
    if (paths.length > 0) void runBatchPaths(paths);
  };

  folderBtn.onclick = async () => {
    const folder = await api.pickFolder();
    if (typeof folder !== "string") return;
    try {
      const paths = await api.listFolderMedia(folder, "image");
      if (paths.length === 0) {
        showToast("No image files found in folder.", "info");
        return;
      }
      void runBatchPaths(paths);
    } catch (err) {
      showToast(String(err), "error");
    }
  };

  const classSel = el("select", { class: "select" }) as HTMLSelectElement;
  classSel.appendChild(el("option", { value: "all" }, ["All Classes"]));
  const allDetected = new Set<string>(bundle.classes);
  for (const r of results) {
    for (const c of r.rawCandidates) allDetected.add(c.className);
    if (r.clsScores[0]) allDetected.add(r.clsScores[0].label);
  }
  for (const cname of Array.from(allDetected)) {
    const o = el("option", { value: cname }, [cname]) as HTMLOptionElement;
    if (classFilter === cname) o.selected = true;
    classSel.appendChild(o);
  }
  classSel.onchange = () => {
    classFilter = classSel.value;
    activeIdx = 0;
    renderImageEvalPane(container, bundle);
  };

  leftPanel.append(
    el("div", { class: "panel-section" }, [
      el("div", { class: "panel-section-title" }, ["Input Source"]),
      el("div", { style: "display:flex;gap:6px;flex-wrap:wrap" }, [
        uploadImgBtn,
        folderBtn,
        cancelBtn,
      ]),
      progressText,
      el("div", { class: "field" }, [
        el("div", { class: "field-label" }, ["Filter Class"]),
        classSel,
      ]),
    ])
  );

  const strip = el("div", { class: "eval-batch-strip" });
  visibleResults.forEach((item, idx) => {
    const badgeText = isCls
      ? `${Math.round((item.clsScores[0]?.score ?? 0) * 100)}%`
      : String(getFilteredDets(item).length);
    const thumb = el(
      "div",
      { class: `thumb-card${idx === activeIdx ? " active" : ""}` },
      [
        el("img", { src: item.srcUrl, loading: "lazy", alt: item.filename }),
        el("span", { class: "thumb-badge", "data-eval-badge-idx": String(idx) }, [badgeText]),
      ]
    );
    thumb.onclick = () => {
      activeIdx = idx;
      renderImageEvalPane(container, bundle);
    };
    strip.appendChild(thumb);
  });
  leftPanel.appendChild(strip);

  const updateThumbBadges = () => {
    if (isCls) return;
    visibleResults.forEach((item, idx) => {
      const badgeEl = strip.querySelector(`[data-eval-badge-idx="${idx}"]`);
      if (badgeEl) {
        badgeEl.textContent = String(getFilteredDets(item).length);
      }
    });
  };

  const centerPanel = el("div", { class: "studio-center" });
  const canvas = el("canvas") as HTMLCanvasElement;
  const stage = el("div", { class: "eval-canvas-wrap" }, [canvas]);
  centerPanel.appendChild(stage);
  stageResizeObs = new ResizeObserver(() => {
    drawResultCanvas(canvas, activeItem, isCls);
  });
  stageResizeObs.observe(stage);

  const rightPanel = el("div", { class: "studio-right" });
  if (!isCls) {
    const confVal = el("span", { class: "mono" }, [confThresh.toFixed(2)]);
    const confSl = el("input", {
      type: "range",
      min: "0.05",
      max: "0.95",
      step: "0.05",
      value: String(confThresh),
    }) as HTMLInputElement;
    confSl.oninput = () => {
      confThresh = Number(confSl.value);
      confVal.textContent = confThresh.toFixed(2);
      drawResultCanvas(canvas, activeItem, isCls);
      updateThumbBadges();
      updateRightDetList();
    };

    const iouVal = el("span", { class: "mono" }, [iouThresh.toFixed(2)]);
    const iouSl = el("input", {
      type: "range",
      min: "0.1",
      max: "0.95",
      step: "0.05",
      value: String(iouThresh),
    }) as HTMLInputElement;
    iouSl.oninput = () => {
      iouThresh = Number(iouSl.value);
      iouVal.textContent = iouThresh.toFixed(2);
      drawResultCanvas(canvas, activeItem, isCls);
      updateThumbBadges();
      updateRightDetList();
    };

    const lblChk = el("input", { type: "checkbox", checked: showLabels }) as HTMLInputElement;
    lblChk.onchange = () => {
      showLabels = lblChk.checked;
      drawResultCanvas(canvas, activeItem, isCls);
    };
    const scoreChk = el("input", { type: "checkbox", checked: showConfidence }) as HTMLInputElement;
    scoreChk.onchange = () => {
      showConfidence = scoreChk.checked;
      drawResultCanvas(canvas, activeItem, isCls);
    };

    rightPanel.appendChild(
      el("div", { class: "panel-section" }, [
        el("div", { class: "panel-section-title" }, ["Live Filter Controls"]),
        el("div", { class: "slider-row" }, [
          el("div", { class: "slider-head" }, [el("span", {}, ["Confidence Threshold"]), confVal]),
          confSl,
        ]),
        el("div", { class: "slider-row" }, [
          el("div", { class: "slider-head" }, [el("span", {}, ["IoU Threshold"]), iouVal]),
          iouSl,
        ]),
        el("label", { class: "check-row" }, [el("span", {}, ["Show Class Labels"]), lblChk]),
        el("label", { class: "check-row" }, [el("span", {}, ["Show Confidence %"]), scoreChk]),
      ])
    );
  }

  const detHost = el("div", { class: "panel-section" });
  const updateRightDetList = () => {
    detHost.innerHTML = "";
    if (!activeItem) {
      detHost.appendChild(
        el("div", { class: "dim", style: "font-size:12px" }, ["No image evaluated yet"])
      );
      return;
    }

    const sendBtn = el("button", { class: "btn btn-sm", type: "button" }, [
      "Send to Labeling",
    ]);
    sendBtn.onclick = async () => {
      const state = getState();
      const targetProj =
        state.activeProject?.module === "image"
          ? state.activeProject
          : state.projects.find((p) => p.module === "image")
            ? await api.loadProject(state.projects.find((p) => p.module === "image")!.id)
            : null;
      if (!targetProj) {
        showToast("Create or open an Image project first.", "error");
        return;
      }
      try {
        const updated = await api.importImages(targetProj.id, [activeItem.absPath]);
        setState({
          activeProject: updated,
          projects: await api.listProjects(),
          activeView: "workspace",
        });
        showToast(`Sent ${activeItem.filename} to Image Labeling`, "success");
      } catch (err) {
        showToast(String(err), "error");
      }
    };

    detHost.appendChild(
      el("div", { class: "panel-section-title" }, ["Result Details", sendBtn])
    );
    detHost.appendChild(
      el("div", { class: "mono", style: "font-size:11px;word-break:break-all" }, [
        `${activeItem.filename} · ${activeItem.width}×${activeItem.height} · ${activeItem.latencyMs} ms`,
      ])
    );

    if (isCls) {
      const list = el("div", { class: "eval-conf-list", style: "margin-top:6px" });
      for (const s of activeItem.clsScores.slice(0, 8)) {
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
      detHost.appendChild(list);
    } else {
      const dets = getFilteredDets(activeItem);
      detHost.appendChild(
        el("div", { class: "dim mono", style: "font-size:11px" }, [
          `${dets.length} detection(s) above threshold`,
        ])
      );
      const list = el("div", { style: "display:flex;flex-direction:column;gap:4px" });
      for (const d of dets) {
        const color = CLASS_PALETTE[d.classIndex % CLASS_PALETTE.length];
        list.appendChild(
          el("div", { class: "anno-item" }, [
            el("span", { class: "class-swatch", style: `background:${color}` }),
            el("span", { style: "flex:1" }, [d.className]),
            el("span", { class: "mono" }, [`${Math.round(d.score * 100)}%`]),
          ])
        );
      }
      detHost.appendChild(list);
    }
  };

  updateRightDetList();
  rightPanel.appendChild(detHost);

  const grid = el("div", { class: "eval-split-grid" }, [leftPanel, centerPanel, rightPanel]);
  container.appendChild(grid);
  requestAnimationFrame(() => drawResultCanvas(canvas, activeItem, isCls));
}
