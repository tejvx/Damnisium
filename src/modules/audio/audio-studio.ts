import { api } from "../../api";
import { getState, setState } from "../../state";
import type { AudioAugConfig, AudioItem, AudioSlice, SplitType } from "../../types";
import {
  buildMelSpectrogramImage,
  decodeAudioBytes,
  type DecodedAudioData,
  playAudioSegment,
} from "../../utils/audio-dsp";
import { el, showToast, svgIcon } from "../../utils/dom";
import { formatTime, randomId } from "../../utils/format";

let lastAudioProjectId: string | null = null;
let activeAudioId: string | null = null;
let selectedSliceId: string | null = null;
let activeClassId = "";
let editingAudioClassId: string | null = null;
let decodedCache = new Map<string, { decoded: DecodedAudioData; specCanvas: HTMLCanvasElement }>();
let selStartSec: number | null = null;
let selEndSec: number | null = null;
let activePlayback: { stop: () => void; startedAt: number; offsetSec: number; rate: number } | null = null;
let playheadSec = 0;
let playbackRaf: number | null = null;
let audioKeyHandler: ((e: KeyboardEvent) => void) | null = null;
let audioMoveHandler: ((e: MouseEvent) => void) | null = null;
let audioUpHandler: (() => void) | null = null;
let audioResizeObs: ResizeObserver | null = null;

export function cleanupAudioStudio(): void {
  if (audioResizeObs) {
    audioResizeObs.disconnect();
    audioResizeObs = null;
  }
  if (activePlayback) {
    activePlayback.stop();
    activePlayback = null;
  }
  if (playbackRaf !== null) {
    cancelAnimationFrame(playbackRaf);
    playbackRaf = null;
  }
  if (audioKeyHandler) {
    window.removeEventListener("keydown", audioKeyHandler);
    audioKeyHandler = null;
  }
  if (audioMoveHandler) {
    window.removeEventListener("mousemove", audioMoveHandler);
    audioMoveHandler = null;
  }
  if (audioUpHandler) {
    window.removeEventListener("mouseup", audioUpHandler);
    audioUpHandler = null;
  }
}

export function renderAudioStudio(container: HTMLElement): void {
  cleanupAudioStudio();
  const state = getState();
  const project = state.activeProject;
  if (!project) return;

  if (lastAudioProjectId !== project.id) {
    lastAudioProjectId = project.id;
    activeAudioId = null;
    selectedSliceId = null;
    activeClassId = "";
    editingAudioClassId = null;
    selStartSec = null;
    selEndSec = null;
    playheadSec = 0;
    decodedCache.clear();
  }

  if (!activeClassId || !project.classes.some((c) => c.id === activeClassId)) {
    activeClassId = project.classes[0]?.id ?? "";
  }
  if (!activeAudioId || !project.audioItems.some((a) => a.id === activeAudioId)) {
    activeAudioId = project.audioItems[0]?.id ?? null;
    selectedSliceId = null;
    selStartSec = null;
    selEndSec = null;
  }

  const activeItem: AudioItem | null =
    project.audioItems.find((a) => a.id === activeAudioId) ?? null;

  container.innerHTML = "";
  const layout = el("div", { class: "audio-layout" });

  const leftPanel = el("div", { class: "studio-left" });
  const importBtn = el("button", { class: "btn btn-sm btn-primary", type: "button" }, [
    svgIcon("upload", 12),
    "Import Audio",
  ]);
  importBtn.onclick = async () => {
    const sel = await api.pickFiles(
      [{ name: "Audio Files", extensions: ["wav", "mp3", "ogg", "flac", "m4a"] }],
      true
    );
    const paths = Array.isArray(sel) ? sel : typeof sel === "string" ? [sel] : [];
    if (paths.length === 0) return;
    try {
      const updated = await api.importAudioFiles(project.id, paths);
      setState({ activeProject: updated, projects: await api.listProjects() });
      renderAudioStudio(container);
      showToast(`Imported ${paths.length} audio file(s)`, "success");
    } catch (err) {
      showToast(String(err), "error");
    }
  };

  const classIn = el("input", { class: "input", placeholder: "New sound class" }) as HTMLInputElement;
  const addClsBtn = el("button", { class: "btn btn-sm", type: "button" }, ["Add"]);
  const addClass = async () => {
    const name = classIn.value.trim();
    if (!name) return;
    try {
      const updated = await api.addLabelClass(project.id, name);
      activeClassId = updated.classes[updated.classes.length - 1]?.id ?? activeClassId;
      setState({ activeProject: updated, projects: await api.listProjects() });
      renderAudioStudio(container);
    } catch (err) {
      showToast(String(err), "error");
    }
  };
  addClsBtn.onclick = () => void addClass();
  classIn.onkeydown = (e) => {
    if (e.key === "Enter") void addClass();
  };

  const classList = el("div", { style: "display:flex;flex-direction:column;gap:3px" });
  for (const cls of project.classes) {
    if (editingAudioClassId === cls.id) {
      const renameIn = el("input", {
        class: "input",
        style: "height:24px;padding:2px 6px;font-size:12px;flex:1;min-width:0",
        value: cls.name,
      }) as HTMLInputElement;
      const saveRename = async () => {
        const nextName = renameIn.value.trim();
        if (!nextName) {
          showToast("Class name cannot be empty.", "error");
          return;
        }
        try {
          const updated = await api.updateLabelClass(project.id, cls.id, nextName, cls.color);
          editingAudioClassId = null;
          setState({ activeProject: updated, projects: await api.listProjects() });
          renderAudioStudio(container);
          showToast(`Renamed class to ${nextName}`, "success");
        } catch (err) {
          showToast(String(err), "error");
        }
      };
      const saveBtn = el(
        "button",
        { class: "btn btn-sm btn-ghost", type: "button", title: "Save class name" },
        [svgIcon("check", 11)]
      );
      saveBtn.onclick = (e) => {
        e.stopPropagation();
        void saveRename();
      };
      renameIn.onkeydown = (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          void saveRename();
        } else if (e.key === "Escape") {
          e.preventDefault();
          editingAudioClassId = null;
          renderAudioStudio(container);
        }
      };
      classList.appendChild(
        el("div", { class: `class-row${cls.id === activeClassId ? " active" : ""}` }, [
          el("span", { class: "class-swatch", style: `background:${cls.color}` }),
          renameIn,
          saveBtn,
        ])
      );
      setTimeout(() => {
        renameIn.focus();
        renameIn.select();
      }, 10);
      continue;
    }

    const editBtn = el(
      "button",
      { class: "btn btn-sm btn-ghost", type: "button", title: "Rename class" },
      [svgIcon("edit", 11)]
    );
    editBtn.onclick = (e) => {
      e.stopPropagation();
      editingAudioClassId = cls.id;
      renderAudioStudio(container);
    };

    const del = el("button", { class: "btn btn-sm btn-ghost", type: "button", title: "Delete class" }, [
      svgIcon("trash", 11),
    ]);
    del.onclick = async (e) => {
      e.stopPropagation();
      try {
        const updated = await api.deleteLabelClass(project.id, cls.id);
        setState({ activeProject: updated, projects: await api.listProjects() });
        renderAudioStudio(container);
      } catch (err) {
        showToast(String(err), "error");
      }
    };
    const nameSpan = el(
      "span",
      { class: "class-name", title: "Double-click to rename" },
      [cls.name]
    );
    nameSpan.ondblclick = (e) => {
      e.stopPropagation();
      editingAudioClassId = cls.id;
      renderAudioStudio(container);
    };
    const row = el(
      "div",
      { class: `class-row${cls.id === activeClassId ? " active" : ""}` },
      [
        el("span", { class: "class-swatch", style: `background:${cls.color}` }),
        nameSpan,
        editBtn,
        del,
      ]
    );
    row.onclick = () => {
      activeClassId = cls.id;
      renderAudioStudio(container);
    };
    classList.appendChild(row);
  }

  const queueList = el("div", { style: "flex:1;overflow-y:auto" });
  for (const item of project.audioItems) {
    const qRow = el(
      "div",
      { class: `audio-queue-item${item.id === activeAudioId ? " active" : ""}` },
      [
        el("div", { style: "min-width:0" }, [
          el("div", {
            style: "font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap",
          }, [item.filename]),
          el("div", { class: "dim mono", style: "font-size:10px", "data-audio-dur": item.id }, [
            item.durationSec > 0 ? formatTime(item.durationSec) : "Decode on open",
          ]),
        ]),
        el("span", { class: "badge mono" }, [String(item.slices.length)]),
      ]
    );
    qRow.onclick = () => {
      activeAudioId = item.id;
      selectedSliceId = null;
      selStartSec = null;
      selEndSec = null;
      playheadSec = 0;
      renderAudioStudio(container);
    };
    queueList.appendChild(qRow);
  }

  leftPanel.append(
    el("div", { class: "panel-section" }, [
      el("div", { class: "panel-section-title" }, ["Audio Queue"]),
      importBtn,
    ])
  );
  if (project.task === "classification") {
    leftPanel.append(
      el("div", { class: "panel-section" }, [
        el("div", { class: "panel-section-title" }, ["Sound Classes"]),
        el("div", { style: "display:flex;gap:5px" }, [classIn, addClsBtn]),
        classList,
      ])
    );
  }
  leftPanel.append(queueList);

  const centerPanel = el("div", { class: "audio-center" });
  const playBtn = el("button", { class: "btn btn-sm btn-primary", type: "button" }, [
    svgIcon("play", 12),
    "Play (Space)",
  ]);
  const addSliceBtn = el("button", { class: "btn btn-sm", type: "button" }, ["Add Slice (S)"]);
  const clearSelBtn = el("button", { class: "btn btn-sm btn-ghost", type: "button" }, [
    "Clear Region",
  ]);
  const timeReadout = el("span", { class: "mono dim", style: "font-size:11px" }, ["00:00.00"]);
  const selReadout = el("span", { class: "mono", style: "font-size:11px;color:var(--cyan)" }, [
    "Region: Full",
  ]);

  const transport = el("div", { class: "audio-transport" }, [
    el("div", { style: "display:flex;align-items:center;gap:6px" }, [
      playBtn,
      addSliceBtn,
      clearSelBtn,
    ]),
    el("div", { style: "display:flex;align-items:center;gap:12px" }, [selReadout, timeReadout]),
  ]);

  const waveCanvas = el("canvas") as HTMLCanvasElement;
  const specCanvas = el("canvas") as HTMLCanvasElement;
  const canvasesWrap = el("div", { class: "audio-canvases" }, [
    el("div", { class: "audio-pane" }, [
      waveCanvas,
      el("div", { class: "audio-pane-label" }, ["WAVEFORM · DRAG TO SELECT"]),
    ]),
    el("div", { class: "audio-pane" }, [
      specCanvas,
      el("div", { class: "audio-pane-label" }, ["MEL SPECTROGRAM"]),
    ]),
  ]);

  centerPanel.append(transport, canvasesWrap);

  const drawAudioViews = (decoded: DecodedAudioData, cachedSpec: HTMLCanvasElement) => {
    const syncCanvasSize = (c: HTMLCanvasElement) => {
      const p = c.parentElement;
      if (!p) return { w: 600, h: 160 };
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, p.clientWidth);
      const h = Math.max(1, p.clientHeight);
      if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
        c.width = Math.round(w * dpr);
        c.height = Math.round(h * dpr);
      }
      const ctx = c.getContext("2d")!;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { w, h };
    };

    const { w: ww, h: wh } = syncCanvasSize(waveCanvas);
    const wCtx = waveCanvas.getContext("2d")!;
    wCtx.clearRect(0, 0, ww, wh);
    wCtx.fillStyle = "#07090d";
    wCtx.fillRect(0, 0, ww, wh);

    const dur = Math.max(0.01, decoded.durationSec);
    if (activeItem) {
      for (const slice of activeItem.slices) {
        const cls = project.classes.find((c) => c.id === slice.classId);
        const color = cls?.color ?? "#38bdf8";
        const sx = (slice.startSec / dur) * ww;
        const ex = (slice.endSec / dur) * ww;
        wCtx.fillStyle = slice.id === selectedSliceId ? `${color}3a` : `${color}1e`;
        wCtx.fillRect(sx, 0, Math.max(2, ex - sx), wh);
        wCtx.strokeStyle = color;
        wCtx.lineWidth = 1;
        wCtx.strokeRect(sx, 0, Math.max(2, ex - sx), wh);
      }
    }

    if (selStartSec !== null && selEndSec !== null) {
      const x1 = (Math.min(selStartSec, selEndSec) / dur) * ww;
      const x2 = (Math.max(selStartSec, selEndSec) / dur) * ww;
      wCtx.fillStyle = "rgba(56, 189, 248, 0.2)";
      wCtx.fillRect(x1, 0, x2 - x1, wh);
      wCtx.strokeStyle = "#38bdf8";
      wCtx.strokeRect(x1, 0, x2 - x1, wh);
    }

    const mono = decoded.mono;
    const step = Math.max(1, Math.floor(mono.length / ww));
    const midY = wh / 2;
    wCtx.strokeStyle = "#38bdf8";
    wCtx.lineWidth = 1;
    wCtx.beginPath();
    for (let x = 0; x < ww; x++) {
      const startIdx = Math.floor((x / ww) * mono.length);
      let min = 1;
      let max = -1;
      for (let k = 0; k < step; k++) {
        const s = mono[startIdx + k] ?? 0;
        if (s < min) min = s;
        if (s > max) max = s;
      }
      wCtx.moveTo(x, midY + min * (wh * 0.44));
      wCtx.lineTo(x, midY + max * (wh * 0.44));
    }
    wCtx.stroke();

    const phX = (playheadSec / dur) * ww;
    wCtx.strokeStyle = "#ffffff";
    wCtx.lineWidth = 1.5;
    wCtx.beginPath();
    wCtx.moveTo(phX, 0);
    wCtx.lineTo(phX, wh);
    wCtx.stroke();

    const { w: sw, h: sh } = syncCanvasSize(specCanvas);
    const sCtx = specCanvas.getContext("2d")!;
    sCtx.clearRect(0, 0, sw, sh);
    sCtx.drawImage(cachedSpec, 0, 0, sw, sh);
    const sphX = (playheadSec / dur) * sw;
    sCtx.strokeStyle = "#ffffff";
    sCtx.lineWidth = 1.5;
    sCtx.beginPath();
    sCtx.moveTo(sphX, 0);
    sCtx.lineTo(sphX, sh);
    sCtx.stroke();

    timeReadout.textContent = `${formatTime(playheadSec)} / ${formatTime(dur)}`;
    if (selStartSec !== null && selEndSec !== null) {
      const a = Math.min(selStartSec, selEndSec);
      const b = Math.max(selStartSec, selEndSec);
      selReadout.textContent = `${formatTime(a)} - ${formatTime(b)}`;
    } else {
      selReadout.textContent = "Region: Full";
    }
  };

  const rightPanel = el("div", { class: "studio-right" });
  const aug: AudioAugConfig = { ...project.audioAug };

  const startPlayback = (start: number, end: number, withAug: AudioAugConfig | null) => {
    if (!activeItem) return;
    const cached = decodedCache.get(activeItem.id);
    if (!cached) return;
    if (activePlayback) {
      activePlayback.stop();
      activePlayback = null;
    }
    playheadSec = start;
    const pb = playAudioSegment(cached.decoded, start, end, withAug, () => {
      activePlayback = null;
      playBtn.innerHTML = "";
      playBtn.append(svgIcon("play", 12), "Play (Space)");
      drawAudioViews(cached.decoded, cached.specCanvas);
    });
    activePlayback = { ...pb, offsetSec: start };
    playBtn.innerHTML = "";
    playBtn.append(svgIcon("pause", 12), "Stop (Space)");

    const startWall = performance.now();
    const animate = () => {
      if (!activePlayback) return;
      playheadSec = Math.min(
        end,
        start + ((performance.now() - startWall) / 1000) * activePlayback.rate
      );
      drawAudioViews(cached.decoded, cached.specCanvas);
      playbackRaf = requestAnimationFrame(animate);
    };
    playbackRaf = requestAnimationFrame(animate);
  };

  const saveSlices = async (nextSlices: AudioSlice[], nextSplit: SplitType) => {
    if (!activeItem) return;
    try {
      const updated = await api.saveAudioSlices(project.id, activeItem.id, nextSplit, nextSlices);
      setState({ activeProject: updated, projects: await api.listProjects() });
      renderAudioStudio(container);
    } catch (err) {
      showToast(String(err), "error");
    }
  };

  const addCurrentSelectionAsSlice = () => {
    if (!activeItem) return;
    const cached = decodedCache.get(activeItem.id);
    const dur = cached?.decoded.durationSec || activeItem.durationSec || 1;
    const a = selStartSec !== null && selEndSec !== null ? Math.min(selStartSec, selEndSec) : 0;
    const b =
      selStartSec !== null && selEndSec !== null ? Math.max(selStartSec, selEndSec) : dur;
    if (b - a < 0.05) {
      showToast("Drag a wider selection on the waveform first.", "error");
      return;
    }
    const created: AudioSlice = {
      id: randomId("slc"),
      startSec: Number(a.toFixed(3)),
      endSec: Number(b.toFixed(3)),
      classId: project.task === "classification" ? activeClassId || null : null,
      transcript: "",
    };
    selectedSliceId = created.id;
    void saveSlices([...activeItem.slices, created], activeItem.split);
  };

  addSliceBtn.onclick = addCurrentSelectionAsSlice;
  clearSelBtn.onclick = () => {
    selStartSec = null;
    selEndSec = null;
    if (activeItem) {
      const cached = decodedCache.get(activeItem.id);
      if (cached) drawAudioViews(cached.decoded, cached.specCanvas);
    }
  };

  playBtn.onclick = () => {
    if (!activeItem) return;
    const cached = decodedCache.get(activeItem.id);
    if (!cached) return;
    if (activePlayback) {
      activePlayback.stop();
      activePlayback = null;
      playBtn.innerHTML = "";
      playBtn.append(svgIcon("play", 12), "Play (Space)");
      drawAudioViews(cached.decoded, cached.specCanvas);
      return;
    }
    const start =
      selStartSec !== null && selEndSec !== null ? Math.min(selStartSec, selEndSec) : 0;
    const end =
      selStartSec !== null && selEndSec !== null
        ? Math.max(selStartSec, selEndSec)
        : cached.decoded.durationSec;
    startPlayback(start, end, null);
  };

  if (activeItem) {
    let dragging = false;
    waveCanvas.addEventListener("mousedown", (e) => {
      const cached = decodedCache.get(activeItem.id);
      if (!cached) return;
      const rect = waveCanvas.getBoundingClientRect();
      const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const t = frac * cached.decoded.durationSec;
      dragging = true;
      selStartSec = t;
      selEndSec = t;
      playheadSec = t;
      drawAudioViews(cached.decoded, cached.specCanvas);
    });
    audioMoveHandler = (e: MouseEvent) => {
      if (!dragging) return;
      const cached = decodedCache.get(activeItem.id);
      if (!cached) return;
      const rect = waveCanvas.getBoundingClientRect();
      const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      selEndSec = frac * cached.decoded.durationSec;
      drawAudioViews(cached.decoded, cached.specCanvas);
    };
    audioUpHandler = () => {
      dragging = false;
    };
    window.addEventListener("mousemove", audioMoveHandler);
    window.addEventListener("mouseup", audioUpHandler);

    const targetId = activeItem.id;
    const loadCurrentAudio = async () => {
      let cached = decodedCache.get(targetId);
      if (!cached) {
        try {
          const bytes = await api.readBinaryFile(activeItem.relPath);
          const decoded = await decodeAudioBytes(bytes);
          const spec = buildMelSpectrogramImage(decoded.mono, decoded.sampleRate);
          if (decodedCache.size >= 5) {
            const oldestKey = decodedCache.keys().next().value;
            if (oldestKey) decodedCache.delete(oldestKey);
          }
          cached = { decoded, specCanvas: spec };
          decodedCache.set(targetId, cached);
          if (Math.abs(activeItem.durationSec - decoded.durationSec) > 0.05) {
            activeItem.durationSec = decoded.durationSec;
            const durEl = queueList.querySelector(`[data-audio-dur="${targetId}"]`);
            if (durEl) {
              durEl.textContent = formatTime(decoded.durationSec);
            }
            void api.updateAudioMetadata(
              project.id,
              targetId,
              decoded.durationSec,
              decoded.sampleRate,
              decoded.channels
            );
          }
        } catch (err) {
          showToast(`Failed decoding audio: ${String(err)}`, "error");
          return;
        }
      }
      if (!waveCanvas.isConnected || activeAudioId !== targetId) return;
      drawAudioViews(cached.decoded, cached.specCanvas);
    };
    audioResizeObs = new ResizeObserver(() => {
      const cached = decodedCache.get(targetId);
      if (cached && waveCanvas.isConnected && activeAudioId === targetId) {
        drawAudioViews(cached.decoded, cached.specCanvas);
      }
    });
    audioResizeObs.observe(canvasesWrap);
    void loadCurrentAudio();

    const splitSel = el("select", { class: "select" }) as HTMLSelectElement;
    for (const sp of ["train", "val", "test"] as SplitType[]) {
      const opt = el("option", { value: sp }, [sp]) as HTMLOptionElement;
      if (activeItem.split === sp) opt.selected = true;
      splitSel.appendChild(opt);
    }
    splitSel.onchange = () => {
      void saveSlices(activeItem.slices, splitSel.value as SplitType);
    };

    const delAudioBtn = el("button", { class: "btn btn-sm btn-danger", type: "button" }, [
      svgIcon("trash", 12),
      "Delete Audio",
    ]);
    delAudioBtn.onclick = async () => {
      try {
        decodedCache.delete(activeItem.id);
        const updated = await api.deleteAudioItem(project.id, activeItem.id);
        activeAudioId = null;
        setState({ activeProject: updated, projects: await api.listProjects() });
        renderAudioStudio(container);
      } catch (err) {
        showToast(String(err), "error");
      }
    };

    const slicesWrap = el("div", { style: "display:flex;flex-direction:column;gap:8px" });
    for (const slc of activeItem.slices) {
      const clsSel = el("select", { class: "select" }) as HTMLSelectElement;
      clsSel.appendChild(el("option", { value: "" }, ["Select Class"]));
      for (const c of project.classes) {
        const o = el("option", { value: c.id }, [c.name]) as HTMLOptionElement;
        if (slc.classId === c.id) o.selected = true;
        clsSel.appendChild(o);
      }
      clsSel.onchange = () => {
        const next = activeItem.slices.map((s) =>
          s.id === slc.id ? { ...s, classId: clsSel.value || null } : s
        );
        void saveSlices(next, activeItem.split);
      };

      const transIn = el("textarea", {
        class: "textarea",
        style: "min-height:54px",
        placeholder: "Transcript text for Whisper training",
      }) as HTMLTextAreaElement;
      transIn.value = slc.transcript;
      transIn.onchange = () => {
        const next = activeItem.slices.map((s) =>
          s.id === slc.id ? { ...s, transcript: transIn.value } : s
        );
        void saveSlices(next, activeItem.split);
      };

      const playSlcBtn = el("button", { class: "btn btn-sm", type: "button" }, [
        svgIcon("play", 11),
      ]);
      playSlcBtn.onclick = () => {
        selStartSec = slc.startSec;
        selEndSec = slc.endSec;
        selectedSliceId = slc.id;
        startPlayback(slc.startSec, slc.endSec, null);
      };

      const delSlcBtn = el("button", { class: "btn btn-sm btn-ghost", type: "button" }, [
        svgIcon("trash", 11),
      ]);
      delSlcBtn.onclick = () => {
        const next = activeItem.slices.filter((s) => s.id !== slc.id);
        void saveSlices(next, activeItem.split);
      };

      slicesWrap.appendChild(
        el("div", { class: `slice-card${slc.id === selectedSliceId ? " active" : ""}` }, [
          el("div", { style: "display:flex;justify-content:space-between;align-items:center" }, [
            el("span", { class: "mono", style: "font-size:11px;color:var(--cyan)" }, [
              `${formatTime(slc.startSec)} -> ${formatTime(slc.endSec)}`,
            ]),
            el("div", { style: "display:flex;gap:4px" }, [playSlcBtn, delSlcBtn]),
          ]),
          project.task === "classification" ? clsSel : transIn,
        ])
      );
    }

    rightPanel.append(
      el("div", { class: "panel-section" }, [
        el("div", { class: "panel-section-title" }, ["Clip Info", delAudioBtn]),
        el("div", { class: "mono", style: "font-size:11px;word-break:break-all" }, [
          activeItem.filename,
        ]),
        el("div", { class: "field" }, [el("div", { class: "field-label" }, ["Split"]), splitSel]),
      ]),
      el("div", { class: "panel-section" }, [
        el("div", { class: "panel-section-title" }, [`Slices (${activeItem.slices.length})`]),
        slicesWrap.children.length > 0
          ? slicesWrap
          : el("div", { class: "dim", style: "font-size:11px" }, [
            "Drag on waveform or click Add Slice (S) to tag or transcribe.",
          ]),
      ])
    );
  }

  const noiseVal = el("span", { class: "mono" }, [aug.noiseLevel.toFixed(3)]);
  const noiseSl = el("input", {
    type: "range",
    min: "0",
    max: "0.08",
    step: "0.005",
    value: String(aug.noiseLevel),
  }) as HTMLInputElement;
  noiseSl.oninput = () => {
    aug.noiseLevel = Number(noiseSl.value);
    noiseVal.textContent = aug.noiseLevel.toFixed(3);
  };

  const pitchVal = el("span", { class: "mono" }, [`${aug.pitchSemitones} st`]);
  const pitchSl = el("input", {
    type: "range",
    min: "-6",
    max: "6",
    step: "0.5",
    value: String(aug.pitchSemitones),
  }) as HTMLInputElement;
  pitchSl.oninput = () => {
    aug.pitchSemitones = Number(pitchSl.value);
    pitchVal.textContent = `${aug.pitchSemitones} st`;
  };

  const speedVal = el("span", { class: "mono" }, [`${aug.speedFactor.toFixed(2)}x`]);
  const speedSl = el("input", {
    type: "range",
    min: "0.8",
    max: "1.25",
    step: "0.05",
    value: String(aug.speedFactor),
  }) as HTMLInputElement;
  speedSl.oninput = () => {
    aug.speedFactor = Number(speedSl.value);
    speedVal.textContent = `${aug.speedFactor.toFixed(2)}x`;
  };

  const gainVal = el("span", { class: "mono" }, [`${aug.gainDb} dB`]);
  const gainSl = el("input", {
    type: "range",
    min: "-12",
    max: "12",
    step: "1",
    value: String(aug.gainDb),
  }) as HTMLInputElement;
  gainSl.oninput = () => {
    aug.gainDb = Number(gainSl.value);
    gainVal.textContent = `${aug.gainDb} dB`;
  };

  const specAugChk = el("input", { type: "checkbox", checked: aug.specAugment }) as HTMLInputElement;
  specAugChk.onchange = () => {
    aug.specAugment = specAugChk.checked;
  };

  const previewAugBtn = el("button", { class: "btn btn-sm", type: "button" }, [
    svgIcon("play", 11),
    "Preview Aug",
  ]);
  previewAugBtn.onclick = () => {
    if (!activeItem) return;
    const cached = decodedCache.get(activeItem.id);
    if (!cached) return;
    const start =
      selStartSec !== null && selEndSec !== null ? Math.min(selStartSec, selEndSec) : 0;
    const end =
      selStartSec !== null && selEndSec !== null
        ? Math.max(selStartSec, selEndSec)
        : cached.decoded.durationSec;
    startPlayback(start, end, aug);
  };

  const saveAugBtn = el("button", { class: "btn btn-sm btn-primary", type: "button" }, [
    "Save Aug",
  ]);
  saveAugBtn.onclick = async () => {
    try {
      const updated = await api.updateProjectSettings(
        project.id,
        project.imageAug,
        aug,
        project.textAug,
        project.splitRatio,
        false
      );
      setState({ activeProject: updated });
      showToast("Saved audio augmentations", "success");
    } catch (err) {
      showToast(String(err), "error");
    }
  };

  rightPanel.appendChild(
    el("div", { class: "panel-section" }, [
      el("div", { class: "panel-section-title" }, ["Audio Augmentations"]),
      el("div", { class: "slider-row" }, [
        el("div", { class: "slider-head" }, [el("span", {}, ["Noise Injection"]), noiseVal]),
        noiseSl,
      ]),
      el("div", { class: "slider-row" }, [
        el("div", { class: "slider-head" }, [el("span", {}, ["Pitch Shift"]), pitchVal]),
        pitchSl,
      ]),
      el("div", { class: "slider-row" }, [
        el("div", { class: "slider-head" }, [el("span", {}, ["Time Stretch"]), speedVal]),
        speedSl,
      ]),
      el("div", { class: "slider-row" }, [
        el("div", { class: "slider-head" }, [el("span", {}, ["Gain Jitter"]), gainVal]),
        gainSl,
      ]),
      el("label", { class: "check-row" }, [el("span", {}, ["SpecAugment"]), specAugChk]),
      el("div", { style: "display:flex;gap:6px" }, [previewAugBtn, saveAugBtn]),
    ])
  );

  audioKeyHandler = (e: KeyboardEvent) => {
    const tag = (e.target as HTMLElement | null)?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (e.code === "Space") {
      e.preventDefault();
      playBtn.click();
    } else if (e.key.toLowerCase() === "s" && !e.ctrlKey && !e.metaKey && !e.altKey) {
      addCurrentSelectionAsSlice();
    }
  };
  window.addEventListener("keydown", audioKeyHandler);

  layout.append(leftPanel, centerPanel, rightPanel);
  container.appendChild(layout);
}
