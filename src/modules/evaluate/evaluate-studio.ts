import { api } from "../../api";
import { openRenameRunModal } from "../../components/dashboard";
import { renderMetricChartSvg } from "../../components/svg-charts";
import { getState, setState } from "../../state";
import type { EvalBundleInfo, ProjectModule, ProjectTask } from "../../types";
import { disposeAutoLabelSession } from "../../utils/autolabel";
import { el, openModal, showToast, svgIcon } from "../../utils/dom";
import { disposeTestRuntime, hasLoadedRuntime, onRuntimeChange } from "../../utils/test-runtime";
import { renderAudioEvalPane, resetAudioEvalState } from "./eval-audio";
import { renderImageEvalPane, resetImageEvalState } from "./eval-image";
import { renderTextEvalPane, resetTextEvalState } from "./eval-text";

let evalModuleTab: ProjectModule = "image";
let customBundleOverride: EvalBundleInfo | null = null;
let runtimeUnsub: (() => void) | null = null;

export function cleanupEvaluateStudio(): void {
  if (runtimeUnsub) {
    runtimeUnsub();
    runtimeUnsub = null;
  }
  resetImageEvalState();
  resetAudioEvalState();
  resetTextEvalState();
  void disposeTestRuntime();
}

function openConfusionModal(bundle: EvalBundleInfo): void {
  const m = bundle.metrics;
  if (!m) return;
  openModal("Validation Metrics & Confusion Matrix", () => {
    const wrap = el("div", { style: "display:flex;flex-direction:column;gap:12px" });
    if (m.lossCurve.length > 0) {
      wrap.appendChild(
        el("div", { class: "chart-box" }, [
          el("div", { class: "chart-title" }, ["Loss Curve"]),
          renderMetricChartSvg(m.lossCurve, "loss"),
        ])
      );
    }
    if (m.confusion && m.confusion.labels.length > 0) {
      const tbl = el("table", { class: "conf-matrix-table" });
      const head = el("tr", {}, [
        el("th", {}, ["T \\ P"]),
        ...m.confusion.labels.map((l) => el("th", {}, [l])),
      ]);
      tbl.appendChild(el("thead", {}, [head]));
      const body = el("tbody");
      m.confusion.matrix.forEach((row, rIdx) => {
        body.appendChild(
          el("tr", {}, [
            el("th", {}, [m.confusion!.labels[rIdx] ?? String(rIdx)]),
            ...row.map((val) => el("td", {}, [String(val)])),
          ])
        );
      });
      tbl.appendChild(body);
      wrap.appendChild(tbl);
    }
    return wrap;
  });
}

function openLoadLocalBundleModal(onLoaded: (bundle: EvalBundleInfo) => void): void {
  openModal("Load Local ONNX Model or Bundle", (close) => {
    const wrap = el("div", { style: "display:flex;flex-direction:column;gap:12px" });
    const modSel = el("select", { class: "select" }, [
      el("option", { value: "image" }, ["Image ML"]),
      el("option", { value: "audio" }, ["Audio ML"]),
      el("option", { value: "text" }, ["Text / Q&A"]),
    ]) as HTMLSelectElement;
    modSel.value = evalModuleTab;

    const taskSel = el("select", { class: "select" }) as HTMLSelectElement;
    const fillTasks = () => {
      taskSel.innerHTML = "";
      const tasks: { v: ProjectTask; l: string }[] =
        modSel.value === "image"
          ? [
            { v: "detection", l: "Object Detection (YOLO ONNX)" },
            { v: "segmentation", l: "Instance Segmentation (YOLO-seg ONNX)" },
            { v: "classification", l: "Image Classification (MobileNet/EfficientNet ONNX)" },
          ]
          : modSel.value === "audio"
            ? [
              { v: "classification", l: "Audio Event Tagging (ONNX)" },
              { v: "transcription", l: "Speech Transcription (Whisper ONNX Dir)" },
            ]
            : [
              { v: "qa", l: "Q&A / Prompt Completion (Qwen ONNX Dir)" },
              { v: "intent", l: "Intent Classification (DistilBERT ONNX Dir)" },
            ];
      for (const t of tasks) {
        taskSel.appendChild(el("option", { value: t.v }, [t.l]));
      }
    };
    modSel.onchange = fillTasks;
    fillTasks();

    const pickOnnxFileBtn = el("button", { class: "btn btn-primary", type: "button" }, [
      "Choose .onnx / .gguf File",
    ]);
    const pickBundleDirBtn = el("button", { class: "btn", type: "button" }, [
      "Choose ONNX Folder",
    ]);

    const handlePicked = async (picked: string | null) => {
      if (!picked) return;
      try {
        const bundle = await api.inspectLocalEvalBundle(
          picked,
          modSel.value as ProjectModule,
          taskSel.value as ProjectTask
        );
        close();
        onLoaded(bundle);
      } catch (err) {
        showToast(String(err), "error");
      }
    };

    pickOnnxFileBtn.onclick = async () => {
      const sel = await api.pickFiles(
        [{ name: "Model Weights", extensions: ["onnx", "gguf"] }],
        false
      );
      void handlePicked(typeof sel === "string" ? sel : null);
    };
    pickBundleDirBtn.onclick = async () => {
      const dir = await api.pickFolder();
      void handlePicked(typeof dir === "string" ? dir : null);
    };

    wrap.append(
      el("div", { class: "field" }, [el("div", { class: "field-label" }, ["Module"]), modSel]),
      el("div", { class: "field" }, [el("div", { class: "field-label" }, ["Task Runtime"]), taskSel]),
      el("div", { class: "modal-actions" }, [pickBundleDirBtn, pickOnnxFileBtn])
    );
    return wrap;
  });
}

function openExportBundleModal(bundle: EvalBundleInfo): void {
  openModal("Export Standalone Model Bundle", () => {
    const wrap = el("div", { style: "display:flex;flex-direction:column;gap:12px" });
    const zipChk = el("input", { type: "checkbox", checked: true }) as HTMLInputElement;
    const statusBox = el("div", { class: "dim mono", style: "font-size:11px" }, [
      `Includes ${bundle.onnxPath.split(/[\\/]/).pop() ?? "weights"}, labels.json, metrics.json, infer_python.py, infer_rust.rs`,
    ]);
    const snippetsWrap = el("div", { style: "display:flex;flex-direction:column;gap:10px" });

    const renderResult = (py: string, rs: string, outPath: string | null) => {
      snippetsWrap.innerHTML = "";
      if (outPath) {
        snippetsWrap.appendChild(
          el("div", { class: "badge badge-success", style: "padding:6px 10px" }, [
            `Exported to: ${outPath}`,
          ])
        );
      }
      for (const [title, code] of [
        ["Python Inference Snippet", py],
        ["Rust Inference Snippet", rs],
      ] as const) {
        const copyBtn = el("button", { class: "btn btn-sm", type: "button" }, ["Copy"]);
        copyBtn.onclick = async () => {
          await navigator.clipboard.writeText(code);
          showToast(`Copied ${title}`, "success");
        };
        snippetsWrap.appendChild(
          el("div", { style: "display:flex;flex-direction:column;gap:4px" }, [
            el("div", { style: "display:flex;justify-content:space-between;align-items:center" }, [
              el("span", { class: "field-label" }, [title]),
              copyBtn,
            ]),
            el("pre", { class: "snippet-pre" }, [code]),
          ])
        );
      }
    };

    const exportBtn = el("button", { class: "btn btn-primary", type: "button" }, [
      svgIcon("package", 13),
      "Choose Destination & Export",
    ]);
    exportBtn.onclick = async () => {
      const dest = await api.pickFolder();
      if (typeof dest !== "string") return;
      exportBtn.setAttribute("disabled", "true");
      try {
        const res = await api.exportModelBundle(
          bundle.onnxPath,
          bundle.labelsPath,
          bundle.metricsPath,
          dest,
          bundle.runtimeKind,
          zipChk.checked
        );
        renderResult(res.pythonSnippet, res.rustSnippet, res.outputPath);
        showToast(`Bundle exported (${res.files.length} files)`, "success");
      } catch (err) {
        showToast(String(err), "error");
      } finally {
        exportBtn.removeAttribute("disabled");
      }
    };

    wrap.append(
      statusBox,
      el("div", { style: "display:flex;align-items:center;justify-content:space-between;gap:10px" }, [
        el("label", { class: "check-row" }, [el("span", {}, ["Package as .zip archive"]), zipChk]),
        exportBtn,
      ]),
      snippetsWrap
    );
    return wrap;
  });
}

export function renderEvaluateStudio(container: HTMLElement): void {
  if (runtimeUnsub) {
    runtimeUnsub();
    runtimeUnsub = null;
  }
  const state = getState();
  const allBundles = state.evalBundles;

  if (state.selectedEvalBundleId) {
    const matched = allBundles.find((b) => b.id === state.selectedEvalBundleId);
    if (matched) {
      evalModuleTab = matched.module;
    }
  } else if (state.activeProject && !customBundleOverride) {
    evalModuleTab = state.activeProject.module;
  }

  const moduleBundles = allBundles.filter((b) => b.module === evalModuleTab);
  const activeBundle =
    (customBundleOverride && customBundleOverride.module === evalModuleTab
      ? customBundleOverride
      : null) ??
    moduleBundles.find((b) => b.id === state.selectedEvalBundleId) ??
    moduleBundles[0] ??
    null;

  container.innerHTML = "";
  const shell = el("div", { class: "eval-shell workspace-enter" });

  const modSegmented = el("div", { class: "segmented" });
  for (const mod of ["image", "audio", "text"] as ProjectModule[]) {
    const label = mod === "image" ? "Image" : mod === "audio" ? "Audio" : "Text / Q&A";
    const btn = el(
      "button",
      { class: evalModuleTab === mod ? "active" : "", type: "button" },
      [label]
    );
    btn.onclick = async () => {
      if (evalModuleTab === mod) return;
      await disposeTestRuntime();
      resetImageEvalState();
      resetAudioEvalState();
      resetTextEvalState();
      evalModuleTab = mod;
      customBundleOverride = null;
      if (getState().selectedEvalBundleId !== null) {
        setState({ selectedEvalBundleId: null });
      } else {
        renderEvaluateStudio(container);
      }
    };
    modSegmented.appendChild(btn);
  }

  const bundleSel = el("select", { class: "select", style: "max-width:290px" }) as HTMLSelectElement;
  if (customBundleOverride && customBundleOverride.module === evalModuleTab) {
    bundleSel.appendChild(
      el("option", { value: customBundleOverride.id }, [`Local: ${customBundleOverride.label}`])
    );
  }
  for (const b of moduleBundles) {
    const o = el("option", { value: b.id }, [b.label]) as HTMLOptionElement;
    if (activeBundle?.id === b.id) o.selected = true;
    bundleSel.appendChild(o);
  }
  if (bundleSel.options.length === 0) {
    bundleSel.appendChild(el("option", { value: "" }, ["No trained models for this module"]));
  }
  bundleSel.onchange = async () => {
    await disposeTestRuntime();
    customBundleOverride = null;
    const nextId = bundleSel.value || null;
    if (getState().selectedEvalBundleId !== nextId) {
      setState({ selectedEvalBundleId: nextId });
    } else {
      renderEvaluateStudio(container);
    }
  };

  const renameBundleBtn =
    activeBundle && activeBundle.source === "colab_run"
      ? el("button", { class: "btn btn-sm", type: "button", title: "Rename trained model weights" }, [
          svgIcon("edit", 12),
          "Rename",
        ])
      : null;
  if (renameBundleBtn && activeBundle) {
    renameBundleBtn.onclick = () => {
      const matchedRun = getState().runs.find((r) => r.id === activeBundle.id);
      const currentName = matchedRun?.modelLabel ?? activeBundle.label;
      openRenameRunModal(activeBundle.id, currentName, () => {
        renderEvaluateStudio(container);
      });
    };
  }

  const loadLocalBtn = el("button", { class: "btn btn-sm", type: "button" }, [
    svgIcon("folder", 12),
    "Load .onnx / .gguf",
  ]);
  loadLocalBtn.onclick = () => {
    openLoadLocalBundleModal(async (bundle) => {
      await disposeTestRuntime();
      evalModuleTab = bundle.module;
      customBundleOverride = bundle;
      renderEvaluateStudio(container);
      showToast(`Loaded ${bundle.label}`, "success");
    });
  };

  const exportBundleBtn = activeBundle
    ? el("button", { class: "btn btn-sm btn-primary", type: "button" }, [
      svgIcon("package", 12),
      "Export Bundle",
    ])
    : null;
  if (exportBundleBtn && activeBundle) {
    exportBundleBtn.onclick = () => openExportBundleModal(activeBundle);
  }

  const unloadBtn = el(
    "button",
    { class: "btn btn-sm", type: "button", disabled: !hasLoadedRuntime() },
    ["Unload RAM"]
  );
  runtimeUnsub = onRuntimeChange(() => {
    unloadBtn.disabled = !hasLoadedRuntime();
  });
  unloadBtn.onclick = async () => {
    await disposeTestRuntime();
    await disposeAutoLabelSession();
    renderEvaluateStudio(container);
    showToast("Unloaded active model runtime from RAM", "info");
  };

  const header = el("div", { class: "eval-header" }, [
    el("div", { class: "eval-header-group" }, [
      el("span", { style: "font-weight:600;font-size:13px" }, ["Test & Evaluate"]),
      modSegmented,
    ]),
    el("div", { class: "eval-header-group" }, [bundleSel, renameBundleBtn, loadLocalBtn, exportBundleBtn, unloadBtn]),
  ]);
  shell.appendChild(header);

  if (activeBundle?.metrics) {
    const m = activeBundle.metrics;
    const strip = el("div", { class: "eval-metrics-strip" });
    if (m.primaryName && m.primaryValue !== null) {
      strip.appendChild(
        el("div", { class: "metric-chip" }, [
          el("span", { class: "dim" }, [m.primaryName]),
          el("span", { class: "metric-chip-val" }, [m.primaryValue.toFixed(4)]),
        ])
      );
    }
    if (m.valAccuracy !== null) {
      strip.appendChild(
        el("div", { class: "metric-chip" }, [
          el("span", { class: "dim" }, ["Val Accuracy"]),
          el("span", { class: "metric-chip-val" }, [`${(m.valAccuracy * 100).toFixed(1)}%`]),
        ])
      );
    }
    if (m.map50 !== null) {
      strip.appendChild(
        el("div", { class: "metric-chip" }, [
          el("span", { class: "dim" }, ["mAP50"]),
          el("span", { class: "metric-chip-val" }, [m.map50.toFixed(4)]),
        ])
      );
    }
    if (m.valLoss !== null) {
      strip.appendChild(
        el("div", { class: "metric-chip" }, [
          el("span", { class: "dim" }, ["Val Loss"]),
          el("span", { class: "metric-chip-val" }, [m.valLoss.toFixed(4)]),
        ])
      );
    }
    const inspectBtn = el("button", { class: "btn btn-sm", type: "button" }, [
      svgIcon("eye", 12),
      "Charts & Matrix",
    ]);
    inspectBtn.onclick = () => openConfusionModal(activeBundle);
    strip.appendChild(inspectBtn);
    shell.appendChild(strip);
  }

  const body = el("div", { class: "eval-body" });
  if (!activeBundle) {
    const emptyLoad = el("button", { class: "btn btn-primary", type: "button" }, [
      svgIcon("folder", 14),
      "Load Local .onnx / .gguf",
    ]);
    emptyLoad.onclick = () => loadLocalBtn.click();
    const hubJump =
      evalModuleTab === "text"
        ? el("button", { class: "btn", type: "button" }, [svgIcon("globe", 14), "Browse Uncensored Hub"])
        : null;
    if (hubJump) {
      hubJump.onclick = () => setState({ activeView: "hub" });
    }
    body.appendChild(
      el("div", { class: "empty-state" }, [
        el("div", { style: "font-weight:600;color:var(--text-1)" }, [
          `No ${evalModuleTab} model loaded`,
        ]),
        el("div", { class: "dim", style: "max-width:420px" }, [
          "Complete a Colab training run, download a GGUF from the Uncensored Hub, or load any local .onnx / .gguf file.",
        ]),
        el("div", { style: "display:flex;gap:8px;flex-wrap:wrap;justify-content:center" }, [
          emptyLoad,
          hubJump,
        ]),
      ])
    );
  } else if (evalModuleTab === "image") {
    renderImageEvalPane(body, activeBundle);
  } else if (evalModuleTab === "audio") {
    renderAudioEvalPane(body, activeBundle);
  } else {
    renderTextEvalPane(body, activeBundle);
  }

  shell.appendChild(body);
  container.appendChild(shell);
}
