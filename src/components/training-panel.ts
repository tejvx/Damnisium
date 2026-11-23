import { api } from "../api";
import { MODULE_TASKS } from "../constants";
import { getState, refreshConnectivity, setState, upsertRunInStore } from "../state";
import type {
  ColabRuntimeSpec,
  HfHubModel,
  PreflightReport,
  Project,
  TrainingConfig,
  TrainingRun,
} from "../types";
import { el, openConfirmModal, openModal, showToast, svgIcon } from "../utils/dom";
import { formatEta } from "../utils/format";
import { confirmAndDeleteProject, deleteTrainingRunWithFeedback, openRenameRunModal } from "./dashboard";
import { renderMetricChartSvg } from "./svg-charts";
import { openGoogleSignInModal } from "./google-signin-modal";

let lastTrainProjectId: string | null = null;
let loadingFallbackProjectId: string | null = null;
let showAllProjectRuns = false;
let selectedModelId = "yolov8n";
let isCustomSelected = false;
let customHfRepo: string | null = null;
let customWeightRelPath: string | null = null;
let resumeFromRunId: string | null = null;
let useQlora = false;
let colabSpecs: ColabRuntimeSpec[] = [];
let colabSpecsLoadedForEmail: string | null = null;
let colabSpecsLoading = false;
let colabSpecsError: string | null = null;
let selectedHardwareKey = "";
let preflightCacheProjectId: string | null = null;
let preflightCacheReport: PreflightReport | null = null;
let launchInFlight = false;
let launchCooldownUntilMs = 0;
let cooldownTimerId: number | null = null;
let lastPanelContainer: HTMLElement | null = null;
let draftEpochs: string | null = null;
let draftBatchSize: string | null = null;
let draftLr: string | null = null;
let draftDim: string | null = null;
let draftTrainPct: string | null = null;
let draftValPct: string | null = null;
let draftTestPct: string | null = null;
let draftExportOnnx = true;
let draftAutoSplit = true;
let draftModelName: string | null = null;
let autoTunedSplitProjectId: string | null = null;
const hfTelemetryCache = new Map<string, HfHubModel>();
const hfTelemetryInFlight = new Set<string>();

function resolveQuotaProjectId(): string {
  const cfg = getState().workspace?.config;
  const explicit = cfg?.gcpProjectId?.trim();
  if (explicit) return explicit;
  const cid = cfg?.oauthClientId?.trim();
  if (cid) {
    const prefix = cid.split("-")[0];
    if (prefix && /^\d+$/.test(prefix)) return prefix;
  }
  return "default";
}

function triggerHfTelemetryFetch(
  repoId: string,
  container: HTMLElement,
  forceRefresh = false
): void {
  const key = repoId.trim().toLowerCase();
  if (!key || hfTelemetryInFlight.has(key)) return;
  if (!forceRefresh && hfTelemetryCache.has(key)) return;
  hfTelemetryInFlight.add(key);
  void api
    .inspectHfRepo(repoId, forceRefresh)
    .then((model) => {
      hfTelemetryCache.set(key, model);
    })
    .catch(() => {})
    .finally(() => {
      hfTelemetryInFlight.delete(key);
      if (getState().activeView === "train" && lastPanelContainer === container) {
        renderTrainingPanel(container);
      }
    });
}

function parseApproxSizeGb(sizeLabel: string | null | undefined): number | null {
  if (!sizeLabel) return null;
  const m = sizeLabel.trim().match(/^([\d.]+)\s*(GB|MB)$/i);
  if (!m) return null;
  const val = Number(m[1]);
  if (!Number.isFinite(val)) return null;
  return m[2].toUpperCase() === "GB" ? val : val / 1024;
}

function formatWeightSizeLabel(hf: HfHubModel | undefined, fallbackLabel: string): string {
  if (hf?.sizeLabel) return hf.sizeLabel;
  if (hf?.totalWeightBytes && hf.totalWeightBytes > 0) {
    const gb = hf.totalWeightBytes / 1_073_741_824;
    if (gb >= 1) return `${gb.toFixed(1)} GB`;
    return `${Math.max(1, Math.round(hf.totalWeightBytes / 1_048_576))} MB`;
  }
  if (hf?.estimatedGb && hf.estimatedGb > 0) {
    if (hf.estimatedGb >= 1) return `${hf.estimatedGb.toFixed(1)} GB`;
    return `${Math.max(1, Math.round(hf.estimatedGb * 1024))} MB`;
  }
  return fallbackLabel;
}

function estimateTrainingVramLabel(
  sizeLabel: string,
  isCausalText: boolean,
  qloraEnabled: boolean
): string {
  const gb = parseApproxSizeGb(sizeLabel);
  if (gb === null) return "Auto (T4 15 GB)";
  const est = isCausalText
    ? qloraEnabled
      ? Math.max(1.2, gb * 0.55 + 1.4)
      : Math.max(1.8, gb * 2.2 + 1.6)
    : Math.max(0.8, gb * 2.6 + 0.7);
  return `~${est.toFixed(1)} GB VRAM`;
}

function buildColabErrorPanel(
  errorText: string,
  quotaProjectId: string,
  onRetry: () => void
): HTMLElement {
  const urlMatch = errorText.match(
    /https:\/\/console\.(?:developers|cloud)\.google\.com\/apis\/api\/colaboratory\.googleapis\.com[^\s|"'`)]*/i
  );
  const isDisabled =
    errorText.includes("SERVICE_DISABLED") ||
    errorText.includes("has not been used in project") ||
    errorText.includes("HTTP 403");
  const activationUrl =
    urlMatch?.[0] ??
    (isDisabled
      ? `https://console.developers.google.com/apis/api/colaboratory.googleapis.com/overview?project=${encodeURIComponent(quotaProjectId)}`
      : null);

  const actions: HTMLElement[] = [];
  if (activationUrl) {
    const enableBtn = el(
      "button",
      { class: "btn btn-sm btn-primary", type: "button" },
      [svgIcon("external", 12), "Enable Colab API"]
    );
    enableBtn.onclick = () => void api.openExternalUrl(activationUrl);
    actions.push(enableBtn);
  }
  const retryBtn = el(
    "button",
    { class: "btn btn-sm", type: "button" },
    [svgIcon("refresh", 12), "Retry"]
  );
  retryBtn.onclick = onRetry;
  actions.push(retryBtn);

  return el(
    "div",
    {
      class: "console-box",
      style: "border-color:var(--danger);color:#fca5a5;display:flex;flex-direction:column;gap:8px",
    },
    [
      el("div", { style: "white-space:pre-wrap;word-break:break-word" }, [errorText]),
      el("div", { style: "display:flex;gap:8px;flex-wrap:wrap" }, actions),
    ]
  );
}

function formatRuntimeSpecLabel(s: ColabRuntimeSpec): string {
  const v = s.variant.replace("VARIANT_", "") || "RUNTIME";
  const a = s.accelerator && s.accelerator !== "NONE" ? ` · ${s.accelerator}` : "";
  const sh = s.shape.replace("SHAPE_", "").toLowerCase();
  const shapeTxt = sh && sh !== "standard" ? ` (${sh})` : "";
  const eligTxt = s.eligible === false ? " (Pro / Ineligible)" : "";
  return `${v}${a}${shapeTxt}${eligTxt}`;
}

function runtimeSpecKey(s: ColabRuntimeSpec): string {
  return `${s.variant}::${s.accelerator}`;
}

function pickDefaultHardwareKey(specs: ColabRuntimeSpec[]): string {
  if (specs.length === 0) return "";
  const eligibleSpecs = specs.filter((s) => s.eligible !== false);
  const pool = eligibleSpecs.length > 0 ? eligibleSpecs : specs;
  const rank = (s: ColabRuntimeSpec): number => {
    const v = s.variant.toUpperCase();
    const a = s.accelerator.toUpperCase();
    const base = v.includes("GPU") ? 100 : v.includes("TPU") ? 50 : 10;
    const acc = a === "T4" ? 40 : a === "L4" ? 35 : a.startsWith("A100") ? 30 : a && a !== "NONE" ? 20 : 0;
    return base + acc;
  };
  const sorted = [...pool].sort((x, y) => rank(y) - rank(x));
  return runtimeSpecKey(sorted[0]);
}

function launchCooldownSeconds(): number {
  return Math.max(0, Math.ceil((launchCooldownUntilMs - Date.now()) / 1000));
}

function updateLaunchButtonsInPlace(): void {
  if (!lastPanelContainer || !lastPanelContainer.isConnected) return;
  const cdSec = launchCooldownSeconds();
  const launchEl = lastPanelContainer.querySelector('[data-launch-btn="1"]') as HTMLButtonElement | null;
  if (launchEl) {
    const baseLabel = launchEl.getAttribute("data-base-label") ?? "Launch on Colab";
    const baseDisabled = launchEl.getAttribute("data-base-disabled") === "1";
    const textSpan = launchEl.querySelector('[data-btn-text="1"]');
    const labelText = launchInFlight
      ? "Uploading Run..."
      : cdSec > 0
      ? `Cooldown (${cdSec}s)...`
      : baseLabel;
    if (textSpan) {
      textSpan.textContent = labelText;
    }
    launchEl.disabled = baseDisabled || launchInFlight || cdSec > 0;
  }
  const offlineEl = lastPanelContainer.querySelector('[data-offline-btn="1"]') as HTMLButtonElement | null;
  if (offlineEl) {
    const baseDisabled = offlineEl.getAttribute("data-base-disabled") === "1";
    offlineEl.disabled = baseDisabled || launchInFlight || cdSec > 0;
  }
}

function scheduleCooldownTick(): void {
  if (cooldownTimerId !== null) {
    window.clearInterval(cooldownTimerId);
    cooldownTimerId = null;
  }
  cooldownTimerId = window.setInterval(() => {
    updateLaunchButtonsInPlace();
    if (Date.now() >= launchCooldownUntilMs && !launchInFlight) {
      if (cooldownTimerId !== null) {
        window.clearInterval(cooldownTimerId);
        cooldownTimerId = null;
      }
    }
  }, 1000);
}

function defaultHyperparams(project: Project): {
  epochs: number;
  batchSize: number;
  lr: number;
  imgSize: number;
  maxSeqLen: number;
} {
  if (project.module === "image") {
    return { epochs: 50, batchSize: 16, lr: 0.001, imgSize: project.task === "classification" ? 224 : 640, maxSeqLen: 512 };
  }
  if (project.module === "audio") {
    return {
      epochs: project.task === "transcription" ? 8 : 30,
      batchSize: 8,
      lr: project.task === "transcription" ? 0.0001 : 0.001,
      imgSize: 640,
      maxSeqLen: 512,
    };
  }
  return {
    epochs: project.task === "intent" ? 6 : 3,
    batchSize: project.task === "intent" ? 16 : 4,
    lr: project.task === "intent" ? 0.00003 : 0.0002,
    imgSize: 640,
    maxSeqLen: project.task === "intent" ? 256 : 512,
  };
}

function recommendedDatasetDefaults(
  project: Project,
  preflight: PreflightReport | null
): {
  epochs: number;
  batchSize: number;
  lr: number;
  imgSize: number;
  maxSeqLen: number;
  splitRatio: { train: number; val: number; test: number };
  itemCount: number;
  summary: string;
} {
  const base = defaultHyperparams(project);
  const rawCount =
    project.module === "image"
      ? project.imageItems.length
      : project.module === "audio"
      ? Math.max(
          project.audioItems.reduce((acc, a) => acc + a.slices.length, 0),
          project.audioItems.length
        )
      : project.textItems.length;
  const n = preflight && preflight.labeledItems > 0 ? preflight.labeledItems : rawCount;

  const splitRatio =
    n <= 2
      ? { train: 100, val: 0, test: 0 }
      : n <= 5
      ? { train: 67, val: 33, test: 0 }
      : n <= 19
      ? { train: 75, val: 25, test: 0 }
      : n <= 199
      ? { train: 80, val: 15, test: 5 }
      : { train: 85, val: 10, test: 5 };

  let epochs = base.epochs;
  let batchSize = base.batchSize;
  let lr = base.lr;
  let imgSize = base.imgSize;
  let maxSeqLen = base.maxSeqLen;

  if (project.module === "image") {
    if (project.task === "classification") {
      batchSize = n <= 4 ? 2 : n <= 16 ? 4 : n <= 64 ? 8 : 16;
      epochs = n <= 10 ? 15 : n <= 50 ? 25 : 40;
      lr = n <= 10 ? 0.0005 : 0.001;
      imgSize = 224;
    } else {
      batchSize = n <= 4 ? 2 : n <= 12 ? 4 : n <= 40 ? 8 : 16;
      epochs = n <= 5 ? 20 : n <= 25 ? 35 : n <= 150 ? 50 : 80;
      lr = 0.001;
      const maxDim = project.imageItems.reduce(
        (m, it) => Math.max(m, it.width || 0, it.height || 0),
        0
      );
      imgSize = maxDim > 0 && maxDim <= 384 ? 320 : maxDim > 0 && maxDim <= 512 ? 512 : 640;
    }
  } else if (project.module === "audio") {
    if (project.task === "transcription") {
      batchSize = n <= 4 ? 2 : n <= 16 ? 4 : 8;
      epochs = n <= 10 ? 5 : n <= 50 ? 8 : 10;
      lr = 0.0001;
    } else {
      batchSize = n <= 8 ? 4 : n <= 32 ? 8 : 16;
      epochs = n <= 15 ? 20 : 30;
      lr = 0.001;
    }
  } else if (project.task === "intent") {
    batchSize = n <= 8 ? 4 : n <= 32 ? 8 : 16;
    epochs = n <= 20 ? 5 : n <= 100 ? 6 : 8;
    lr = 0.00003;
    const maxChars = project.textItems.reduce((m, it) => Math.max(m, it.prompt.length), 0);
    maxSeqLen = maxChars > 0 && maxChars <= 300 ? 128 : maxChars <= 800 ? 256 : 512;
  } else {
    batchSize = n <= 4 ? 1 : n <= 16 ? 2 : 4;
    epochs = n <= 10 ? 3 : n <= 50 ? 4 : 3;
    lr = 0.0002;
    const maxChars = project.textItems.reduce(
      (m, it) => Math.max(m, it.system.length + it.prompt.length + it.completion.length),
      0
    );
    maxSeqLen = maxChars > 0 && maxChars <= 400 ? 256 : maxChars <= 1200 ? 512 : 1024;
  }

  const summary = `Auto-tuned for ${n} items (bs ${batchSize} · ep ${epochs} · ${splitRatio.train}/${splitRatio.val}/${splitRatio.test})`;
  return { epochs, batchSize, lr, imgSize, maxSeqLen, splitRatio, itemCount: n, summary };
}

function taskDisplayLabel(project: Project): string {
  const mod = MODULE_TASKS[project.module];
  const taskLabel = mod.tasks.find((t) => t.value === project.task)?.label ?? project.task;
  return `${mod.label} · ${taskLabel}`;
}

function customModuleConfigFor(project: Project): {
  allowHf: boolean;
  allowLocal: boolean;
  defaultSource: "huggingface" | "local_weight";
  namePlaceholder: string;
  hfPlaceholder: string;
  extensions: string[];
  helperText: string;
} {
  if (project.module === "image" && project.task === "classification") {
    return {
      allowHf: true,
      allowLocal: true,
      defaultSource: "huggingface",
      namePlaceholder: "e.g. ConvNeXt-Tiny or Custom ResNet",
      hfPlaceholder: "e.g. timm/mobilenetv3_large_100.ra_in1k or google/vit-base-patch16-224",
      extensions: ["pt", "pth", "safetensors", "bin"],
      helperText: "Loads timm HuggingFace hub vision models or local PyTorch state_dict checkpoints.",
    };
  }
  if (project.module === "image") {
    return {
      allowHf: false,
      allowLocal: true,
      defaultSource: "local_weight",
      namePlaceholder: "e.g. Custom YOLO11n Checkpoint",
      hfPlaceholder: "",
      extensions: ["pt", "onnx"],
      helperText: "Imports custom Ultralytics .pt or .onnx weights into Colab for fine-tuning.",
    };
  }
  if (project.module === "audio" && project.task === "transcription") {
    return {
      allowHf: true,
      allowLocal: false,
      defaultSource: "huggingface",
      namePlaceholder: "e.g. Distil-Whisper or Fine-tuned Whisper",
      hfPlaceholder: "e.g. openai/whisper-small or openai/whisper-medium",
      extensions: ["bin", "safetensors"],
      helperText: "Any HuggingFace Whisper seq2seq speech recognition repository.",
    };
  }
  if (project.module === "audio") {
    return {
      allowHf: true,
      allowLocal: true,
      defaultSource: "huggingface",
      namePlaceholder: "e.g. Custom Audio Head or AST Repo",
      hfPlaceholder: "e.g. MIT/ast-finetuned-audioset-10-10-0.4593",
      extensions: ["keras", "h5", "pt", "onnx"],
      helperText: "Supports HuggingFace audio repos or local .keras/.pt classification checkpoints.",
    };
  }
  if (project.task === "intent") {
    return {
      allowHf: true,
      allowLocal: true,
      defaultSource: "huggingface",
      namePlaceholder: "e.g. RoBERTa Base or BGE Classifier",
      hfPlaceholder: "e.g. FacebookAI/roberta-base or microsoft/MiniLM-L12-H384-uncased",
      extensions: ["bin", "safetensors", "pt"],
      helperText: "Any HuggingFace encoder model compatible with AutoModelForSequenceClassification.",
    };
  }
  return {
    allowHf: true,
    allowLocal: true,
    defaultSource: "huggingface",
    namePlaceholder: "e.g. Dolphin 3.0 1B or Qwen2.5 1.5B Uncensored",
    hfPlaceholder: "e.g. dphn/Dolphin3.0-Llama3.2-1B or Qwen/Qwen2.5-1.5B-Instruct",
    extensions: ["safetensors", "bin", "pt"],
    helperText: "HuggingFace causal LLM repo ID (GGUF repos auto-resolve their LoRA base model on Colab).",
  };
}

function openCustomModelModal(project: Project, onDone: () => void): void {
  const cfgSpec = customModuleConfigFor(project);
  openModal(`Add Custom Module · ${taskDisplayLabel(project)}`, (close) => {
    const wrap = el("div", { style: "display:flex;flex-direction:column;gap:12px" });
    const nameIn = el("input", {
      class: "input",
      placeholder: cfgSpec.namePlaceholder,
    }) as HTMLInputElement;

    let kind: "local_weight" | "huggingface" = cfgSpec.defaultSource;
    const kindSel = el("select", { class: "select" }) as HTMLSelectElement;
    if (cfgSpec.allowHf) {
      kindSel.appendChild(el("option", { value: "huggingface" }, ["HuggingFace Repo ID / URL"]));
    }
    if (cfgSpec.allowLocal) {
      kindSel.appendChild(
        el("option", { value: "local_weight" }, [`Local Weight File (${cfgSpec.extensions.map((e) => `.${e}`).join(" / ")})`])
      );
    }
    kindSel.value = kind;

    const hfIn = el("input", {
      class: "input mono",
      placeholder: cfgSpec.hfPlaceholder,
    }) as HTMLInputElement;

    let pickedFile = "";
    const fileLabel = el("div", { class: "dim mono", style: "font-size:11px" }, ["No file selected"]);
    const pickFileBtn = el("button", { class: "btn btn-sm", type: "button" }, ["Choose Weight File"]);
    pickFileBtn.onclick = async () => {
      const sel = await api.pickFiles(
        [{ name: "Model Weights", extensions: cfgSpec.extensions }],
        false
      );
      if (typeof sel === "string") {
        pickedFile = sel;
        fileLabel.textContent = sel;
      }
    };

    const hfField = el("div", { class: "field", style: kind === "huggingface" ? "display:flex" : "display:none" }, [
      el("div", { class: "field-label" }, ["HuggingFace Model ID"]),
      hfIn,
    ]);
    const fileField = el("div", { class: "field", style: kind === "local_weight" ? "display:flex" : "display:none" }, [
      el("div", { class: "field-label" }, ["Weight File"]),
      pickFileBtn,
      fileLabel,
    ]);

    kindSel.onchange = () => {
      kind = kindSel.value as "local_weight" | "huggingface";
      hfField.style.display = kind === "huggingface" ? "flex" : "none";
      fileField.style.display = kind === "local_weight" ? "flex" : "none";
    };

    const saveBtn = el("button", { class: "btn btn-primary", type: "button" }, ["Save Custom Module"]);
    saveBtn.onclick = async () => {
      const name = nameIn.value.trim();
      if (!name) {
        showToast("Enter a model name.", "error");
        return;
      }
      const rawRepo = hfIn.value.trim().replace(/^https?:\/\/(www\.)?huggingface\.co\//i, "").replace(/\/(tree|blob)\/main.*$/i, "").replace(/\/+$/, "");
      if (kind === "huggingface" && !rawRepo) {
        showToast("Enter a HuggingFace repository ID.", "error");
        return;
      }
      if (kind === "local_weight" && !pickedFile) {
        showToast("Select a local weight file.", "error");
        return;
      }
      try {
        const saved = await api.registerCustomModel(
          name,
          project.module,
          project.task,
          kind,
          kind === "local_weight" ? pickedFile : null,
          kind === "huggingface" ? rawRepo : null
        );
        const ws = await api.getWorkspaceInfo();
        selectedModelId = saved.id;
        isCustomSelected = true;
        customHfRepo = saved.hfRepoId;
        customWeightRelPath = saved.localRelPath;
        setState({ workspace: ws });
        close();
        onDone();
        showToast(`Registered ${saved.name}`, "success");
      } catch (err) {
        showToast(String(err), "error");
      }
    };

    wrap.append(
      el("div", { class: "dim", style: "font-size:11px" }, [cfgSpec.helperText]),
      el("div", { class: "field" }, [el("div", { class: "field-label" }, ["Module Name"]), nameIn]),
      el("div", { class: "field" }, [el("div", { class: "field-label" }, ["Source"]), kindSel]),
      hfField,
      fileField,
      el("div", { class: "modal-actions" }, [saveBtn])
    );
    setTimeout(() => nameIn.focus(), 20);
    return wrap;
  });
}

function buildRunConsole(
  run: TrainingRun,
  consoleScroll?: { top: number; pinnedBottom: boolean },
  onRetryRun?: () => void
): HTMLElement {
  const badgeCls =
    run.state === "completed"
      ? "badge-success"
      : run.state === "failed"
      ? "badge-danger"
      : "badge-accent";

  const isActiveRun =
    run.state === "uploading" ||
    run.state === "installing" ||
    run.state === "training" ||
    run.state === "downloading_weights";

  const cancelBtn = isActiveRun
    ? el("button", { class: "btn btn-sm btn-danger", type: "button" }, ["Cancel Run"])
    : null;
  if (cancelBtn) {
    cancelBtn.onclick = async () => {
      cancelBtn.setAttribute("disabled", "true");
      try {
        const updated = await api.cancelTrainingPoll(run.id);
        upsertRunInStore(updated);
        showToast("Colab run cancelled and runtime cleaned up", "info");
      } catch (err) {
        showToast(String(err), "error");
      }
    };
  }

  const renameRunBtn = el(
    "button",
    { class: "btn btn-sm", type: "button", title: "Rename model / trained weights" },
    [svgIcon("edit", 12), "Rename"]
  );
  renameRunBtn.onclick = () => openRenameRunModal(run.id, run.modelLabel);

  const deleteRunBtn = el(
    "button",
    { class: "btn btn-sm btn-danger", type: "button", title: "Delete this training run" },
    [svgIcon("trash", 12), "Delete"]
  );
  deleteRunBtn.onclick = () => {
    openConfirmModal(
      "Delete Training Run",
      `Delete run ${run.id} (${run.modelLabel}) and its local bundle files from disk?`,
      "Delete Run",
      async () => {
        await deleteTrainingRunWithFeedback(run.id);
      }
    );
  };

  const syncBtn = run.driveRunFolderId
    ? el("button", { class: "btn btn-sm", type: "button" }, ["Sync Drive"])
    : null;
  if (syncBtn) {
    syncBtn.onclick = async () => {
      try {
        const updated = await api.syncTrainingRun(run.id);
        upsertRunInStore(updated);
        showToast("Synced run status from Google Drive", "info");
      } catch (err) {
        showToast(String(err), "error");
      }
    };
  }

  const downloadNbBtn = el("button", { class: "btn btn-sm", type: "button" }, [
    svgIcon("download", 12),
    "Download .ipynb",
  ]);
  downloadNbBtn.onclick = async () => {
    try {
      const dest = await api.saveFile(`damnisium_${run.id}.ipynb`, [
        { name: "Jupyter Notebook", extensions: ["ipynb"] },
      ]);
      if (typeof dest === "string" && dest.trim()) {
        const saved = await api.exportRunNotebook(run.id, dest);
        showToast(`Saved ${saved}`, "success");
      }
    } catch (err) {
      showToast(String(err), "error");
    }
  };

  const openWeightsBtn = el("button", { class: "btn btn-sm", type: "button" }, [
    svgIcon("folder", 12),
    "Open Weights Folder",
  ]);
  openWeightsBtn.onclick = () => void api.openLocalPath(run.weightsDir);

  const onnxPath = run.downloadedWeights.find((w) => w.toLowerCase().endsWith(".onnx"));
  const evalReady =
    run.state === "completed" &&
    (Boolean(onnxPath) ||
      run.downloadedWeights.some((w) => w.toLowerCase().includes("onnx_model")));
  const testEvalBtn = evalReady
    ? el("button", { class: "btn btn-sm btn-primary", type: "button" }, [
        svgIcon("flask", 12),
        "Test & Evaluate",
      ])
    : null;
  if (testEvalBtn) {
    testEvalBtn.onclick = async () => {
      try {
        const bundles = await api.listEvalBundles();
        setState({
          evalBundles: bundles,
          selectedEvalBundleId: run.id,
          activeView: "evaluate",
        });
      } catch (err) {
        showToast(String(err), "error");
      }
    };
  }
  const loadOnnxBtn =
    onnxPath && run.module === "image" && run.task === "detection"
      ? el("button", { class: "btn btn-sm", type: "button" }, ["Use ONNX in Auto-Label"])
      : null;
  if (loadOnnxBtn && onnxPath) {
    loadOnnxBtn.onclick = async () => {
      try {
        await api.importCustomAutolabelOnnx(onnxPath);
        const list = await api.listAutolabelModels();
        setState({ autolabelModels: list });
        showToast("Imported trained ONNX model into Auto-Label Assist", "success");
      } catch (err) {
        showToast(String(err), "error");
      }
    };
  }

  const logLines = run.logTail;
  const copyOutputsBtn = el("button", { class: "btn btn-sm", type: "button" }, [
    svgIcon("copy", 12),
    "Copy Outputs",
  ]);
  copyOutputsBtn.onclick = async () => {
    try {
      await navigator.clipboard.writeText(logLines.join("\n"));
      copyOutputsBtn.innerHTML = "";
      copyOutputsBtn.append(svgIcon("check", 12), "Copied");
      window.setTimeout(() => {
        copyOutputsBtn.innerHTML = "";
        copyOutputsBtn.append(svgIcon("copy", 12), "Copy Outputs");
      }, 2000);
    } catch (err) {
      showToast(String(err), "error");
    }
  };

function renderColabConsoleLine(rawLine: string): HTMLElement {
  const trimmed = rawLine.trim();
  if (trimmed.startsWith("[Damnisium Error]")) {
    const body = trimmed.slice("[Damnisium Error]".length).trim();
    return el("div", { class: "console-line console-line-error" }, [
      el("span", { class: "console-tag tag-error" }, ["ERROR"]),
      el("span", { class: "console-msg" }, [body || trimmed]),
    ]);
  }
  if (trimmed.startsWith("[Damnisium]")) {
    const body = trimmed.slice("[Damnisium]".length).trim();
    return el("div", { class: "console-line console-line-sys" }, [
      el("span", { class: "console-tag tag-sys" }, ["DAMNISIUM"]),
      el("span", { class: "console-msg" }, [body || trimmed]),
    ]);
  }
  if (trimmed.startsWith("[Colab Result]")) {
    const body = trimmed.slice("[Colab Result]".length).trim();
    return el("div", { class: "console-line console-line-sys" }, [
      el("span", { class: "console-tag tag-ok" }, ["COLAB"]),
      el("span", { class: "console-msg" }, [body || trimmed]),
    ]);
  }

  const tsMatch = /^\[(\d{2}:\d{2}:\d{2})\]\s*(.*)$/.exec(trimmed);
  if (tsMatch) {
    const stamp = tsMatch[1];
    const body = tsMatch[2].trim();
    const tsNode = el("span", { class: "console-ts" }, [stamp]);

    if (body.startsWith("Error:") || body.startsWith("Failed:")) {
      return el("div", { class: "console-line console-line-error" }, [
        tsNode,
        el("span", { class: "console-tag tag-error" }, ["ERROR"]),
        el("span", { class: "console-msg" }, [body]),
      ]);
    }
    if (body.startsWith("Running:")) {
      const cmdText = body.slice("Running:".length).trim();
      return el("div", { class: "console-line console-line-cmd" }, [
        tsNode,
        el("span", { class: "console-tag tag-cmd" }, ["EXEC"]),
        el("span", { class: "console-msg console-msg-dim" }, [cmdText]),
      ]);
    }
    if (/^(Epoch|Step)\s+/.test(body)) {
      const tokens = body.split(/\s+/);
      const badgeLabel = tokens.slice(0, 2).join(" ");
      const kvNodes = tokens.slice(2).map((tok) => {
        const eqIdx = tok.indexOf("=");
        if (eqIdx > 0) {
          const k = tok.slice(0, eqIdx);
          const v = tok.slice(eqIdx + 1);
          return el("span", { class: "console-kv" }, [
            el("span", { class: "console-kv-k" }, [k]),
            el("span", { class: "console-kv-v" }, [v]),
          ]);
        }
        return el("span", { class: "console-msg" }, [tok]);
      });
      return el("div", { class: "console-line console-line-metric" }, [
        tsNode,
        el("span", { class: "console-tag tag-metric" }, [badgeLabel]),
        el("div", { class: "console-kv-row" }, kvNodes),
      ]);
    }
    return el("div", { class: "console-line" }, [
      tsNode,
      el("span", { class: "console-msg" }, [body]),
    ]);
  }

  if (/\b(Error|Exception|Traceback|Failed):/i.test(trimmed)) {
    return el("div", { class: "console-line console-line-error" }, [
      el("span", { class: "console-tag tag-error" }, ["ERROR"]),
      el("span", { class: "console-msg" }, [trimmed]),
    ]);
  }
  return el("div", { class: "console-line" }, [
    el("span", { class: "console-msg console-msg-dim" }, [trimmed]),
  ]);
}

  const consoleBox = el(
    "div",
    { class: "console-box", "data-run-console": "1", "data-run-id": run.id },
    logLines.length > 0
      ? logLines.map((line) => renderColabConsoleLine(line))
      : [
          el("div", { class: "console-empty" }, [
            run.state === "waiting_for_colab"
              ? "Offline bundle saved locally — download the .ipynb notebook to run manually in Colab."
              : "Allocating headless Colab runtime & streaming MCP kernel output...",
          ]),
        ]
  );
  queueMicrotask(() => {
    if (consoleScroll && !consoleScroll.pinnedBottom) {
      consoleBox.scrollTop = consoleScroll.top;
    } else {
      consoleBox.scrollTop = consoleBox.scrollHeight;
    }
  });

  return el("div", { class: "card", style: "display:flex;flex-direction:column;gap:12px" }, [
    el("div", { class: "card-header", style: "margin-bottom:0" }, [
      el("div", {}, [
        el("div", { class: "card-title" }, [`${run.projectName} · ${run.modelLabel}`]),
        el("div", { class: "dim mono", style: "font-size:11px;margin-top:2px" }, [
          `${run.id} · GPU: ${run.gpuName ?? run.colabAccelerator ?? "Pending"} · ETA: ${formatEta(run.etaSeconds)}`,
        ]),
      ]),
      el("div", { style: "display:flex;align-items:center;gap:6px;flex-wrap:wrap" }, [
        el("span", { class: `badge ${badgeCls}` }, [run.state]),
        cancelBtn,
        syncBtn,
        downloadNbBtn,
        openWeightsBtn,
        testEvalBtn,
        loadOnnxBtn,
        renameRunBtn,
        deleteRunBtn,
      ]),
    ]),
    el("div", {}, [
      el(
        "div",
        { style: "display:flex;justify-content:space-between;font-size:11px;margin-bottom:5px" },
        [
          el("span", { class: "muted" }, [run.message]),
          el("span", { class: "mono" }, [
            `Epoch ${run.currentEpoch.toFixed(1)}/${run.totalEpochs} (${Math.round(run.progressPct)}%)`,
          ]),
        ]
      ),
      el("div", { class: "progress-track" }, [
        el("div", { class: "progress-fill", style: `width:${Math.min(100, run.progressPct)}%` }),
      ]),
    ]),
    run.error
      ? buildColabErrorPanel(run.error, resolveQuotaProjectId(), () => {
          colabSpecsLoadedForEmail = null;
          colabSpecsError = null;
          launchCooldownUntilMs = 0;
          if (onRetryRun) {
            onRetryRun();
          } else if (lastPanelContainer) {
            renderTrainingPanel(lastPanelContainer);
          }
        })
      : null,
    el("div", { class: "grid-2" }, [
      el("div", { class: "chart-box" }, [
        el("div", { class: "chart-title" }, [
          el("span", {}, ["Loss Curve"]),
          el("span", { class: "mono dim" }, ["Crimson: Train · Amber: Val"]),
        ]),
        renderMetricChartSvg(run.metrics, "loss"),
      ]),
      el("div", { class: "chart-box" }, [
        el("div", { class: "chart-title" }, [
          el("span", {}, ["Validation Metric"]),
          el("span", { class: "mono dim" }, [
            run.metrics[run.metrics.length - 1]?.scoreName ?? "Score",
          ]),
        ]),
        renderMetricChartSvg(run.metrics, "score"),
      ]),
    ]),
    run.downloadedWeights.length > 0
      ? el("div", { style: "display:flex;flex-wrap:wrap;gap:6px;align-items:center" }, [
          el("span", { class: "field-label" }, ["Weights:"]),
          ...run.downloadedWeights.map((w) =>
            el("span", { class: "badge badge-success mono" }, [w.split(/[\\/]/).pop() ?? w])
          ),
        ])
      : null,
    el("div", {}, [
      el(
        "div",
        { style: "display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;gap:8px" },
        [
          el("div", { style: "display:flex;align-items:center;gap:8px" }, [
            el("span", { class: "field-label" }, ["Colab Live Outputs"]),
            el("span", { class: "badge mono" }, [`${logLines.length} lines`]),
          ]),
          logLines.length > 0 ? copyOutputsBtn : null,
        ]
      ),
      consoleBox,
    ]),
  ]);
}

function projectDatasetStatsText(project: Project): string {
  if (project.module === "image") {
    return `${project.imageItems.length} images · ${project.classes.length} classes`;
  }
  if (project.module === "audio") {
    const slices = project.audioItems.reduce((acc, a) => acc + a.slices.length, 0);
    return `${project.audioItems.length} clips · ${slices} slices`;
  }
  return `${project.textItems.length} samples · ${project.task === "intent" ? `${project.classes.length} classes` : "prompt/completion"}`;
}

export function renderTrainingPanel(container: HTMLElement): void {
  const prevScroller = container.querySelector(".view-scroll") as HTMLElement | null;
  const savedScrollTop = prevScroller ? prevScroller.scrollTop : 0;
  const prevConsole = container.querySelector('[data-run-console="1"]') as HTMLElement | null;
  const savedConsoleRunId = prevConsole?.getAttribute("data-run-id") ?? null;
  const savedConsoleState = prevConsole
    ? {
        top: prevConsole.scrollTop,
        pinnedBottom:
          prevConsole.scrollHeight - prevConsole.scrollTop - prevConsole.clientHeight <= 24,
      }
    : undefined;

  lastPanelContainer = container;
  const state = getState();
  container.innerHTML = "";
  const root = el("div", { class: "view-scroll" });
  const wrap = el("div", { class: "train-layout" });

  const projectOptions = state.projects;
  const activeProj = state.activeProject;

  if (!activeProj && projectOptions.length > 0) {
    const fallbackId = projectOptions[0].id;
    if (loadingFallbackProjectId !== fallbackId) {
      loadingFallbackProjectId = fallbackId;
      void api.loadProject(fallbackId).then((p) => {
        loadingFallbackProjectId = null;
        if (!getState().activeProject) {
          setState({ activeProject: p });
        }
      });
    }
  } else {
    loadingFallbackProjectId = null;
  }

  const projSel = el("select", { class: "select", style: "max-width:280px" }) as HTMLSelectElement;
  for (const p of projectOptions) {
    const opt = el("option", { value: p.id }, [`${p.name} (${MODULE_TASKS[p.module].label} · ${p.task})`]) as HTMLOptionElement;
    if (p.id === activeProj?.id) opt.selected = true;
    projSel.appendChild(opt);
  }
  projSel.onchange = async () => {
    try {
      const p = await api.loadProject(projSel.value);
      setState({ activeProject: p });
    } catch (err) {
      showToast(String(err), "error");
    }
  };

  const deleteProjBtn = activeProj
    ? el(
        "button",
        {
          class: "btn btn-sm btn-danger",
          type: "button",
          title: `Delete project ${activeProj.name}`,
        },
        [svgIcon("trash", 12), "Delete Project"]
      )
    : null;
  if (deleteProjBtn && activeProj) {
    deleteProjBtn.onclick = () => confirmAndDeleteProject(activeProj.id, activeProj.name);
  }

  wrap.appendChild(
    el("div", { class: "dashboard-header", style: "margin-bottom:4px" }, [
      el("div", {}, [
        el("div", { style: "font-size:17px;font-weight:600;display:flex;align-items:center;gap:8px;flex-wrap:wrap" }, [
          activeProj ? `${activeProj.name} · Colab Training` : "Model Training & Colab Compute",
          activeProj ? el("span", { class: "badge badge-accent" }, [taskDisplayLabel(activeProj)]) : null,
        ]),
        el("div", { class: "dim", style: "font-size:12px" }, [
          activeProj
            ? `${projectDatasetStatsText(activeProj)} · Stream remote GPU training and checkpoints from Google Colab`
            : "Select or create a project to configure pretrained modules and Colab training",
        ]),
      ]),
      el("div", { style: "display:flex;align-items:center;gap:8px;flex-wrap:wrap" }, [
        projectOptions.length > 0 ? projSel : null,
        deleteProjBtn,
      ]),
    ])
  );

  let triggerActiveLaunch: (() => void) | undefined;

  if (activeProj) {
    const ws = state.workspace;
    const compatiblePretrained = (ws?.pretrainedModels ?? []).filter(
      (m) => m.module === activeProj.module && m.tasks.includes(activeProj.task)
    );
    const compatibleCustom = (ws?.customModels ?? []).filter(
      (m) => m.module === activeProj.module && m.task === activeProj.task
    );

    if (lastTrainProjectId !== activeProj.id) {
      lastTrainProjectId = activeProj.id;
      autoTunedSplitProjectId = null;
      selectedModelId = compatiblePretrained[0]?.id ?? compatibleCustom[0]?.id ?? "";
      isCustomSelected = compatiblePretrained.length === 0 && compatibleCustom.length > 0;
      customHfRepo = isCustomSelected ? compatibleCustom[0]?.hfRepoId ?? null : null;
      customWeightRelPath = isCustomSelected ? compatibleCustom[0]?.localRelPath ?? null : null;
      resumeFromRunId = null;
      useQlora = false;
      preflightCacheProjectId = null;
      preflightCacheReport = null;
      draftEpochs = null;
      draftBatchSize = null;
      draftLr = null;
      draftDim = null;
      draftTrainPct = null;
      draftValPct = null;
      draftTestPct = null;
      draftExportOnnx = true;
      draftAutoSplit = true;
      draftModelName = null;
    }

    if (state.hubPendingModelRepo) {
      const targetRepo = state.hubPendingModelRepo.toLowerCase();
      const matched = compatibleCustom.find((cm) => cm.hfRepoId?.toLowerCase() === targetRepo);
      if (matched) {
        selectedModelId = matched.id;
        isCustomSelected = true;
        customHfRepo = matched.hfRepoId;
        customWeightRelPath = matched.localRelPath;
      } else {
        selectedModelId = "custom-hf";
        isCustomSelected = true;
        customHfRepo = state.hubPendingModelRepo;
        customWeightRelPath = null;
      }
      if (activeProj.module === "text" && activeProj.task !== "intent") {
        useQlora = true;
      }
      setState({ hubPendingModelRepo: null });
    }

    if (
      !isCustomSelected &&
      !compatiblePretrained.some((m) => m.id === selectedModelId) &&
      compatiblePretrained.length > 0
    ) {
      selectedModelId = compatiblePretrained[0].id;
      customHfRepo = null;
      customWeightRelPath = null;
    }

    if (preflightCacheProjectId !== activeProj.id) {
      preflightCacheProjectId = activeProj.id;
      preflightCacheReport = null;
      const pid = activeProj.id;
      const projSnapshot = activeProj;
      void api.runPreflightCheck(pid).then(async (rep) => {
        if (preflightCacheProjectId !== pid) return;
        const rec = recommendedDatasetDefaults(projSnapshot, rep);
        const hasNoValWarn = rep.issues.some((i) => i.code === "NO_VAL_SPLIT");
        const splitDiffers =
          projSnapshot.splitRatio.train !== rec.splitRatio.train ||
          projSnapshot.splitRatio.val !== rec.splitRatio.val ||
          projSnapshot.splitRatio.test !== rec.splitRatio.test;
        if (
          (hasNoValWarn || splitDiffers) &&
          draftTrainPct === null &&
          autoTunedSplitProjectId !== pid
        ) {
          autoTunedSplitProjectId = pid;
          try {
            const updatedProj = await api.updateProjectSettings(
              pid,
              projSnapshot.imageAug,
              projSnapshot.audioAug,
              projSnapshot.textAug,
              rec.splitRatio,
              true
            );
            const rep2 = await api.runPreflightCheck(pid);
            if (preflightCacheProjectId === pid) {
              preflightCacheReport = rep2;
              if (getState().activeProject?.id === pid) {
                setState({ activeProject: updatedProj });
              } else {
                renderTrainingPanel(container);
              }
            }
            return;
          } catch {
          }
        }
        preflightCacheReport = rep;
        renderTrainingPanel(container);
      });
    }

    const defaults = recommendedDatasetDefaults(activeProj, preflightCacheReport);
    const epochsIn = el("input", {
      class: "input",
      type: "number",
      min: "1",
      max: "500",
      value: draftEpochs ?? String(defaults.epochs),
    }) as HTMLInputElement;
    epochsIn.oninput = () => {
      draftEpochs = epochsIn.value;
    };
    const batchIn = el("input", {
      class: "input",
      type: "number",
      min: "1",
      max: "128",
      value: draftBatchSize ?? String(defaults.batchSize),
    }) as HTMLInputElement;
    batchIn.oninput = () => {
      draftBatchSize = batchIn.value;
    };
    const lrIn = el("input", {
      class: "input",
      type: "number",
      step: "0.00001",
      value: draftLr ?? String(defaults.lr),
    }) as HTMLInputElement;
    lrIn.oninput = () => {
      draftLr = lrIn.value;
    };
    const dimIn = el("input", {
      class: "input",
      type: "number",
      min: "64",
      max: "2048",
      step: "32",
      value:
        draftDim ?? String(activeProj.module === "text" ? defaults.maxSeqLen : defaults.imgSize),
    }) as HTMLInputElement;
    dimIn.oninput = () => {
      draftDim = dimIn.value;
    };

    const exportOnnxChk = el("input", { type: "checkbox", checked: draftExportOnnx }) as HTMLInputElement;
    exportOnnxChk.onchange = () => {
      draftExportOnnx = exportOnnxChk.checked;
    };
    const autoSplitChk = el("input", { type: "checkbox", checked: draftAutoSplit }) as HTMLInputElement;
    autoSplitChk.onchange = () => {
      draftAutoSplit = autoSplitChk.checked;
    };

    const trainPctIn = el("input", {
      class: "input",
      type: "number",
      min: "10",
      max: "100",
      value: draftTrainPct ?? String(defaults.splitRatio.train),
    }) as HTMLInputElement;
    const valPctIn = el("input", {
      class: "input",
      type: "number",
      min: "0",
      max: "60",
      value: draftValPct ?? String(defaults.splitRatio.val),
    }) as HTMLInputElement;
    const testPctIn = el("input", {
      class: "input",
      type: "number",
      min: "0",
      max: "50",
      value: draftTestPct ?? String(defaults.splitRatio.test),
    }) as HTMLInputElement;

    const splitTrainSeg = el("div", { class: "split-bar-train" });
    const splitValSeg = el("div", { class: "split-bar-val" });
    const splitTestSeg = el("div", { class: "split-bar-test" });
    const splitSummaryLabel = el("span", { class: "mono dim", style: "font-size:11px" });
    const updateSplitBar = (recordDraft: boolean) => {
      if (recordDraft) {
        draftTrainPct = trainPctIn.value;
        draftValPct = valPctIn.value;
        draftTestPct = testPctIn.value;
      }
      const tr = Math.max(0, Number(trainPctIn.value) || 0);
      const va = Math.max(0, Number(valPctIn.value) || 0);
      const te = Math.max(0, Number(testPctIn.value) || 0);
      const sum = Math.max(1, tr + va + te);
      splitTrainSeg.style.width = `${(tr / sum) * 100}%`;
      splitValSeg.style.width = `${(va / sum) * 100}%`;
      splitTestSeg.style.width = `${(te / sum) * 100}%`;
      splitSummaryLabel.textContent = `Train ${Math.round((tr / sum) * 100)}% · Val ${Math.round((va / sum) * 100)}% · Test ${Math.round((te / sum) * 100)}%`;
    };
    trainPctIn.oninput = () => updateSplitBar(true);
    valPctIn.oninput = () => updateSplitBar(true);
    testPctIn.oninput = () => updateSplitBar(true);
    updateSplitBar(false);

    const priorRuns = state.runs.filter((r) => r.projectId === activeProj.id);
    const resumeSel = el("select", { class: "select" }, [
      el("option", { value: "" }, ["None (Fresh Training Run)"]),
    ]) as HTMLSelectElement;
    for (const r of priorRuns) {
      const o = el(
        "option",
        { value: r.id },
        [`${r.id} · ${r.modelLabel} (${r.state} · ep ${r.currentEpoch.toFixed(0)}/${r.totalEpochs})`]
      ) as HTMLOptionElement;
      if (resumeFromRunId === r.id) o.selected = true;
      resumeSel.appendChild(o);
    }
    resumeSel.onchange = () => {
      resumeFromRunId = resumeSel.value || null;
    };

    const qloraChk = el("input", { type: "checkbox", checked: useQlora }) as HTMLInputElement;
    qloraChk.onchange = () => {
      useQlora = qloraChk.checked;
    };

    const hubBrowseBtn = el("button", { class: "btn btn-sm", type: "button" }, [
      svgIcon("globe", 12),
      "Uncensored Hub",
    ]);
    hubBrowseBtn.onclick = () => setState({ activeView: "hub" });

    const addCustomBtn = el("button", { class: "btn btn-sm btn-primary", type: "button" }, [
      svgIcon("plus", 12),
      "Add Custom Module",
    ]);
    addCustomBtn.onclick = () =>
      openCustomModelModal(activeProj, () => renderTrainingPanel(container));

    const modelGrid = el("div", { class: "model-grid" });
    for (const m of compatiblePretrained) {
      const isSelected = !isCustomSelected && selectedModelId === m.id;
      const hfTel = m.hfRepoId ? hfTelemetryCache.get(m.hfRepoId.toLowerCase()) : undefined;
      if (m.hfRepoId && state.connectivity.online && !hfTel) {
        triggerHfTelemetryFetch(m.hfRepoId, container);
      }
      const dispParam = hfTel?.paramLabel ?? m.paramCount;
      const dispSize = formatWeightSizeLabel(hfTel, m.sizeLabel);
      const card = el("div", { class: `model-card${isSelected ? " active" : ""}` }, [
        el("div", { class: "model-card-title" }, [m.name]),
        el("div", { class: "dim", style: "font-size:11px" }, [m.architecture]),
        el("div", { class: "model-card-meta" }, [
          el("span", {}, [dispParam]),
          el("span", {}, [dispSize]),
          el("span", {}, [m.speedTier]),
        ]),
      ]);
      card.onclick = () => {
        selectedModelId = m.id;
        isCustomSelected = false;
        customHfRepo = null;
        customWeightRelPath = null;
        renderTrainingPanel(container);
      };
      modelGrid.appendChild(card);
    }

    for (const cm of compatibleCustom) {
      const isSelected = isCustomSelected && selectedModelId === cm.id;
      const hfTel = cm.hfRepoId ? hfTelemetryCache.get(cm.hfRepoId.toLowerCase()) : undefined;
      if (cm.hfRepoId && state.connectivity.online && !hfTel) {
        triggerHfTelemetryFetch(cm.hfRepoId, container);
      }
      const removeCustomBtn = el(
        "button",
        {
          class: "btn btn-sm btn-ghost",
          type: "button",
          title: "Remove custom module",
          style: "padding:2px 4px;height:20px",
        },
        [svgIcon("trash", 11)]
      );
      removeCustomBtn.onclick = async (e) => {
        e.stopPropagation();
        try {
          await api.deleteCustomModel(cm.id);
          const updatedWs = await api.getWorkspaceInfo();
          if (selectedModelId === cm.id) {
            isCustomSelected = false;
            selectedModelId = compatiblePretrained[0]?.id ?? "";
            customHfRepo = null;
            customWeightRelPath = null;
          }
          setState({ workspace: updatedWs });
        } catch (err) {
          showToast(String(err), "error");
        }
      };

      const card = el("div", { class: `model-card${isSelected ? " active" : ""}` }, [
        el("div", { style: "display:flex;align-items:center;justify-content:space-between;gap:6px" }, [
          el("div", { class: "model-card-title" }, [cm.name]),
          removeCustomBtn,
        ]),
        el("div", { class: "dim mono", style: "font-size:11px;word-break:break-all" }, [
          cm.hfRepoId ?? cm.localRelPath ?? "Custom Weight",
        ]),
        el("div", { class: "model-card-meta" }, [
          el("span", {}, [hfTel?.paramLabel ?? (cm.sourceKind === "huggingface" ? "HF Hub" : "Local")]),
          el("span", {}, [
            hfTel
              ? formatWeightSizeLabel(hfTel, "--")
              : cm.sourceKind === "huggingface"
              ? "Inspecting..."
              : "Custom File",
          ]),
        ]),
      ]);
      card.onclick = () => {
        selectedModelId = cm.id;
        isCustomSelected = true;
        customHfRepo = cm.hfRepoId;
        customWeightRelPath = cm.localRelPath;
        renderTrainingPanel(container);
      };
      modelGrid.appendChild(card);
    }

    const activePretrained = !isCustomSelected
      ? compatiblePretrained.find((m) => m.id === selectedModelId) ?? compatiblePretrained[0]
      : undefined;
    const activeCustom = isCustomSelected
      ? compatibleCustom.find((cm) => cm.id === selectedModelId)
      : undefined;
    const activeRepo = customHfRepo ?? activeCustom?.hfRepoId ?? activePretrained?.hfRepoId ?? null;
    if (activeRepo && state.connectivity.online && !hfTelemetryCache.has(activeRepo.toLowerCase())) {
      triggerHfTelemetryFetch(activeRepo, container);
    }
    const activeHf = activeRepo ? hfTelemetryCache.get(activeRepo.toLowerCase()) : undefined;
    const selName = activeCustom?.name ?? activePretrained?.name ?? customHfRepo ?? selectedModelId;
    const selParams = activeHf?.paramLabel ?? activePretrained?.paramCount ?? "Unknown";
    const selSize = formatWeightSizeLabel(
      activeHf,
      activePretrained?.sizeLabel ?? (customWeightRelPath ? "Local Weight" : "--")
    );
    const isCausalText = activeProj.module === "text" && activeProj.task !== "intent";
    const selVram = estimateTrainingVramLabel(selSize, isCausalText, useQlora);
    const refreshSelTelemetryBtn = activeRepo
      ? el(
          "button",
          {
            class: "btn btn-sm btn-ghost",
            type: "button",
            title: "Refresh Hugging Face parameter and byte size metadata",
            disabled: hfTelemetryInFlight.has(activeRepo.toLowerCase()),
          },
          [svgIcon("refresh", 12)]
        )
      : null;
    if (refreshSelTelemetryBtn && activeRepo) {
      const targetRepo = activeRepo;
      refreshSelTelemetryBtn.onclick = () => triggerHfTelemetryFetch(targetRepo, container, true);
    }
    const selectedTelemetryCard = el(
      "div",
      {
        style:
          "display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;padding:10px 12px;border-radius:var(--radius-md);background:var(--bg-1);border:1px solid var(--border-1)",
      },
      [
        el("div", { style: "display:flex;align-items:center;gap:8px;flex-wrap:wrap" }, [
          el("span", { style: "font-weight:600;font-size:12px" }, [`Selected: ${selName}`]),
          activeRepo ? el("span", { class: "mono dim", style: "font-size:11px" }, [activeRepo]) : null,
        ]),
        el("div", { style: "display:flex;align-items:center;gap:8px;flex-wrap:wrap" }, [
          el("span", { class: "badge badge-accent mono" }, [`Params: ${selParams}`]),
          el("span", { class: "badge mono" }, [`Size: ${selSize}`]),
          el("span", { class: "badge mono" }, [selVram]),
          refreshSelTelemetryBtn,
        ]),
      ]
    );

    const modelNameIn = el("input", {
      class: "input",
      type: "text",
      placeholder: `Auto (${selName} — auto-numbered if repeated)`,
      value: draftModelName ?? "",
    }) as HTMLInputElement;
    modelNameIn.oninput = () => {
      draftModelName = modelNameIn.value;
    };

    const exportZipBtn = el("button", { class: "btn btn-sm", type: "button" }, [
      svgIcon("package", 12),
      "Export Dataset ZIP",
    ]);
    exportZipBtn.onclick = async () => {
      const dest = await api.saveFile(`${activeProj.name}_dataset.zip`, [
        { name: "ZIP Archive", extensions: ["zip"] },
      ]);
      if (!dest) return;
      try {
        const updatedProj = await api.updateProjectSettings(
          activeProj.id,
          activeProj.imageAug,
          activeProj.audioAug,
          activeProj.textAug,
          {
            train: Number(trainPctIn.value) || 80,
            val: Number(valPctIn.value) || 15,
            test: Number(testPctIn.value) || 5,
          },
          autoSplitChk.checked
        );
        const res = await api.exportDatasetZip(updatedProj.id, dest);
        setState({ activeProject: updatedProj });
        showToast(`Exported ${res.itemCount} items (${res.classCount} classes) to ZIP`, "success");
      } catch (err) {
        showToast(String(err), "error");
      }
    };

    const recheckBtn = el("button", { class: "btn btn-sm", type: "button" }, [
      svgIcon("shield", 12),
      "Run Pre-Flight Check",
    ]);
    recheckBtn.onclick = async () => {
      try {
        const rep = await api.runPreflightCheck(activeProj.id);
        preflightCacheProjectId = activeProj.id;
        preflightCacheReport = rep;
        renderTrainingPanel(container);
      } catch (err) {
        showToast(String(err), "error");
      }
    };

    const autoTuneBtn = el("button", { class: "btn btn-sm", type: "button" }, [
      svgIcon("refresh", 12),
      "Auto-Tune Defaults",
    ]);
    autoTuneBtn.onclick = async () => {
      try {
        draftEpochs = null;
        draftBatchSize = null;
        draftLr = null;
        draftDim = null;
        draftTrainPct = null;
        draftValPct = null;
        draftTestPct = null;
        autoTunedSplitProjectId = activeProj.id;
        const rec = recommendedDatasetDefaults(activeProj, preflightCacheReport);
        const updatedProj = await api.updateProjectSettings(
          activeProj.id,
          activeProj.imageAug,
          activeProj.audioAug,
          activeProj.textAug,
          rec.splitRatio,
          true
        );
        const rep = await api.runPreflightCheck(activeProj.id);
        preflightCacheProjectId = activeProj.id;
        preflightCacheReport = rep;
        setState({ activeProject: updatedProj });
        showToast(rec.summary, "success");
      } catch (err) {
        showToast(String(err), "error");
      }
    };

    const pf = preflightCacheReport;
    const pfBox = el("div", { class: "preflight-box" }, [
      el("div", { style: "display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap" }, [
        el("div", { style: "display:flex;align-items:center;gap:8px;flex-wrap:wrap" }, [
          el("span", { style: "font-weight:600;font-size:12px" }, ["Pre-Flight Dataset Check"]),
          pf
            ? el("span", { class: `badge ${pf.ok ? "badge-success" : "badge-danger"}` }, [
                pf.ok ? "Ready" : "Action Needed",
              ])
            : el("span", { class: "badge" }, ["Scanning..."]),
          pf
            ? el("span", { class: "mono dim", style: "font-size:11px" }, [
                `Total ${pf.totalItems} · Labeled ${pf.labeledItems} · Splits ${pf.splitCounts.train}/${pf.splitCounts.val}/${pf.splitCounts.test}`,
              ])
            : null,
          el("span", { class: "badge mono", style: "font-size:10px" }, [defaults.summary]),
        ]),
        el("div", { style: "display:flex;gap:6px;flex-wrap:wrap" }, [autoTuneBtn, exportZipBtn, recheckBtn]),
      ]),
      ...(pf?.issues.map((iss) =>
        el("div", { style: "display:flex;align-items:center;gap:8px;font-size:11px" }, [
          el("span", { class: `badge ${iss.severity === "error" ? "badge-danger" : "badge-warn"}` }, [
            iss.severity.toUpperCase(),
          ]),
          el("span", {}, [
            iss.itemIds.length > 0 ? `${iss.message} (${iss.itemIds.slice(0, 5).join(", ")})` : iss.message,
          ]),
        ])
      ) ?? []),
    ]);

    const pfBlocked = pf !== null && !pf.ok;
    const isOffline = !state.connectivity.online;
    const uploadingRun = state.runs.some(
      (r) => r.projectId === activeProj.id && r.state === "uploading"
    );
    const cdSec = launchCooldownSeconds();
    const launchLocked = launchInFlight || uploadingRun || cdSec > 0;
    const baseLaunchLabel = `Launch ${taskDisplayLabel(activeProj)} on Colab`;
    const launchLabel =
      launchInFlight || uploadingRun
        ? "Uploading Run..."
        : cdSec > 0
        ? `Cooldown (${cdSec}s)...`
        : baseLaunchLabel;
    const launchBtn = el(
      "button",
      {
        class: "btn btn-primary",
        type: "button",
        disabled: pfBlocked || isOffline || launchLocked,
        "data-launch-btn": "1",
        "data-base-label": baseLaunchLabel,
        "data-base-disabled": pfBlocked || isOffline ? "1" : "0",
      },
      [svgIcon("cpu", 14), el("span", { "data-btn-text": "1" }, [launchLabel])]
    );
    const offlineBtn = el(
      "button",
      {
        class: "btn",
        type: "button",
        disabled: pfBlocked || launchLocked,
        "data-offline-btn": "1",
        "data-base-disabled": pfBlocked ? "1" : "0",
      },
      [svgIcon("download", 14), "Export Offline Colab Bundle"]
    );

    const runTraining = async (offlineOnly: boolean) => {
      if (
        launchInFlight ||
        getState().runs.some((r) => r.projectId === activeProj.id && r.state === "uploading") ||
        launchCooldownSeconds() > 0
      ) {
        return;
      }
      if (!offlineOnly && !state.connectivity.online) {
        const check = await refreshConnectivity();
        if (!check.online) {
          showToast("Offline Mode detected — use Export Offline Colab Bundle or Retry Connection.", "error");
          return;
        }
      }
      if (!offlineOnly && (!getState().workspace?.auth.connected || !getState().workspace?.auth.colabAuthorized)) {
        openGoogleSignInModal(() => {
          colabSpecsLoadedForEmail = null;
          renderTrainingPanel(container);
        });
        return;
      }
      lastPanelContainer = container;
      launchInFlight = true;
      launchBtn.setAttribute("disabled", "true");
      offlineBtn.setAttribute("disabled", "true");
      if (!offlineOnly) {
        const span = launchBtn.querySelector('[data-btn-text="1"]');
        if (span) {
          span.textContent = "Starting Colab Run...";
        }
      }
      let triggeredRun = false;
      try {
        const rep = await api.runPreflightCheck(activeProj.id);
        preflightCacheProjectId = activeProj.id;
        preflightCacheReport = rep;
        if (!rep.ok) {
          renderTrainingPanel(container);
          showToast(rep.issues.find((i) => i.severity === "error")?.message ?? "Pre-flight check failed", "error");
          return;
        }
        triggeredRun = true;
        launchCooldownUntilMs = Date.now() + 8000;
        scheduleCooldownTick();
        const updatedProj = await api.updateProjectSettings(
          activeProj.id,
          activeProj.imageAug,
          activeProj.audioAug,
          activeProj.textAug,
          {
            train: Number(trainPctIn.value) || 80,
            val: Number(valPctIn.value) || 15,
            test: Number(testPctIn.value) || 5,
          },
          autoSplitChk.checked
        );
        const [chosenVariant, chosenAccel] = hwSel.value ? hwSel.value.split("::") : ["", ""];
        const cfg: TrainingConfig = {
          projectId: updatedProj.id,
          modelId: selectedModelId,
          isCustomModel: isCustomSelected,
          customHfRepo,
          customWeightRelPath,
          epochs: Math.max(1, Number(epochsIn.value) || 10),
          batchSize: Math.max(1, Number(batchIn.value) || 8),
          learningRate: Number(lrIn.value) || 0.001,
          imgSize: activeProj.module === "image" ? Math.max(64, Number(dimIn.value) || 640) : 640,
          maxSeqLen: activeProj.module === "text" ? Math.max(64, Number(dimIn.value) || 512) : 512,
          exportOnnx: exportOnnxChk.checked,
          autoApplySplit: autoSplitChk.checked,
          resumeFromRunId,
          useQlora,
          preferredAccelerator: chosenAccel || null,
          runtimeVariant: chosenVariant || null,
          customModelName: modelNameIn.value.trim() || null,
        };
        const run = await api.startTrainingRun(cfg, offlineOnly);
        draftModelName = null;
        setState({ activeProject: updatedProj, selectedRunId: run.id });
        upsertRunInStore(run);
        showToast(
          offlineOnly
            ? "Offline Colab bundle saved locally"
            : "Headless Colab GPU run started — streaming progress below",
          "success"
        );
      } catch (err) {
        showToast(String(err), "error");
      } finally {
        launchInFlight = false;
        if (triggeredRun) {
          launchCooldownUntilMs = Date.now() + 8000;
          scheduleCooldownTick();
        }
        renderTrainingPanel(container);
      }
    };

    launchBtn.onclick = () => void runTraining(false);
    offlineBtn.onclick = () => void runTraining(true);
    triggerActiveLaunch = () => {
      launchCooldownUntilMs = 0;
      void runTraining(false);
    };

    const authEmail = ws?.auth.user?.email ?? null;
    const canFetchSpecs = Boolean(
      ws?.auth.connected && ws?.auth.colabAuthorized && state.connectivity.online
    );
    if (canFetchSpecs && !colabSpecsLoading && colabSpecsLoadedForEmail !== authEmail) {
      colabSpecsLoadedForEmail = authEmail;
      colabSpecsLoading = true;
      colabSpecsError = null;
      void api
        .listColabRuntimeSpecs()
        .then((specs) => {
          colabSpecs = specs;
          if (!selectedHardwareKey || !specs.some((s) => runtimeSpecKey(s) === selectedHardwareKey)) {
            selectedHardwareKey = pickDefaultHardwareKey(specs);
          }
        })
        .catch((err) => {
          colabSpecsError = String(err);
        })
        .finally(() => {
          colabSpecsLoading = false;
          if (getState().activeView === "train") {
            renderTrainingPanel(container);
          }
        });
    }

    const hwSel = el("select", { class: "select" }) as HTMLSelectElement;
    if (colabSpecs.length > 0) {
      const seen = new Set<string>();
      const sortedSpecs = [...colabSpecs].sort(
        (a, b) => Number(b.eligible !== false) - Number(a.eligible !== false)
      );
      for (const s of sortedSpecs) {
        const key = runtimeSpecKey(s);
        if (seen.has(key)) continue;
        seen.add(key);
        const opt = el("option", { value: key }, [formatRuntimeSpecLabel(s)]) as HTMLOptionElement;
        if (s.eligible === false) opt.disabled = true;
        if (key === selectedHardwareKey) opt.selected = true;
        hwSel.appendChild(opt);
      }
    } else {
      hwSel.appendChild(
        el("option", { value: "VARIANT_GPU::T4" }, [
          colabSpecsLoading
            ? "Loading Colab hardware specs..."
            : !ws?.auth.connected
            ? "Auto GPU (Sign in to list account specs)"
            : !ws?.auth.colabAuthorized
            ? "Re-sign in required for Colab specs"
            : "Auto GPU (T4 Preferred)",
        ])
      );
    }
    hwSel.onchange = () => {
      selectedHardwareKey = hwSel.value;
    };

    const refreshHwBtn = el(
      "button",
      {
        class: "btn btn-sm",
        type: "button",
        title: "Refresh Colab GPU/TPU specs",
        disabled: colabSpecsLoading || !canFetchSpecs,
      },
      [svgIcon("refresh", 12)]
    );
    refreshHwBtn.onclick = () => {
      colabSpecsLoadedForEmail = null;
      renderTrainingPanel(container);
    };

    const thirdHyperparamField =
      activeProj.module === "audio"
        ? el("div", { class: "field" }, [
            el("div", { class: "field-label" }, ["Audio Sample Rate"]),
            el("input", {
              class: "input mono",
              type: "text",
              value: "16000 Hz (Mono PCM)",
              readonly: true,
            }),
          ])
        : el("div", { class: "field" }, [
            el("div", { class: "field-label" }, [
              activeProj.module === "text" ? "Max Seq Length" : "Input Image Size",
            ]),
            dimIn,
          ]);

    const stepCard = el("div", { class: "card", style: "display:flex;flex-direction:column;gap:14px" }, [
      el("div", { class: "card-header", style: "margin-bottom:0" }, [
        el("div", { style: "display:flex;align-items:center;gap:8px;flex-wrap:wrap" }, [
          el("div", { class: "card-title" }, [`1. Select ${taskDisplayLabel(activeProj)} Module`]),
          customHfRepo ? el("span", { class: "badge badge-accent mono" }, [customHfRepo]) : null,
        ]),
        el("div", { style: "display:flex;gap:6px;align-items:center" }, [hubBrowseBtn, addCustomBtn]),
      ]),
      modelGrid,
      selectedTelemetryCard,
      el("div", { class: "card-title" }, ["2. Dataset Split, Checkpoints & Pre-Flight"]),
      el("div", { class: "grid-3" }, [
        el("div", { class: "field" }, [el("div", { class: "field-label" }, ["Train %"]), trainPctIn]),
        el("div", { class: "field" }, [el("div", { class: "field-label" }, ["Val %"]), valPctIn]),
        el("div", { class: "field" }, [el("div", { class: "field-label" }, ["Test %"]), testPctIn]),
      ]),
      el("div", { style: "display:flex;flex-direction:column;gap:5px" }, [
        el("div", { style: "display:flex;justify-content:space-between;align-items:center" }, [
          el("span", { class: "field-label" }, ["Dataset Split Distribution"]),
          splitSummaryLabel,
        ]),
        el("div", { class: "split-bar-track" }, [splitTrainSeg, splitValSeg, splitTestSeg]),
      ]),
      pfBox,
      el("div", { class: "grid-3" }, [
        el("div", { class: "field" }, [el("div", { class: "field-label" }, ["Epochs"]), epochsIn]),
        el("div", { class: "field" }, [el("div", { class: "field-label" }, ["Batch Size"]), batchIn]),
        el("div", { class: "field" }, [el("div", { class: "field-label" }, ["Learning Rate"]), lrIn]),
      ]),
      el("div", { class: "grid-3" }, [
        thirdHyperparamField,
        el("div", { class: "field" }, [
          el("div", { class: "field-label" }, ["Colab Runtime Hardware"]),
          el("div", { style: "display:flex;gap:6px;align-items:center" }, [hwSel, refreshHwBtn]),
        ]),
        el("div", { class: "field" }, [
          el("div", { class: "field-label" }, ["Resume From Checkpoint"]),
          resumeSel,
        ]),
      ]),
      el("div", { class: "field" }, [
        el("div", { class: "field-label" }, ["Trained Model / Weights Name (optional)"]),
        modelNameIn,
      ]),
      colabSpecsError
        ? buildColabErrorPanel(
            `Hardware lookup error: ${colabSpecsError}`,
            resolveQuotaProjectId(),
            () => {
              colabSpecsLoadedForEmail = null;
              colabSpecsError = null;
              renderTrainingPanel(container);
            }
          )
        : null,
      el("div", { style: "display:flex;flex-wrap:wrap;gap:14px" }, [
        el("label", { class: "check-row" }, [
          el("span", {}, ["Auto-apply split before launch"]),
          autoSplitChk,
        ]),
        el("label", { class: "check-row" }, [
          el("span", {}, [
            activeProj.module === "text" && activeProj.task !== "intent"
              ? "Export merged ONNX"
              : "Export ONNX weights",
          ]),
          exportOnnxChk,
        ]),
        activeProj.module === "text" && activeProj.task !== "intent"
          ? el("label", { class: "check-row" }, [
              el("span", {}, ["Use 4-bit QLoRA (Colab T4)"]),
              qloraChk,
            ])
          : null,
      ]),
      el("div", {
        style: "display:flex;align-items:center;justify-content:space-between;gap:10px;padding-top:6px;border-top:1px solid var(--border-1);flex-wrap:wrap",
      }, [
        el("div", { class: "dim", style: "font-size:12px" }, [
          isOffline
            ? "Offline Mode active — Retry Connection in top bar or export Colab bundle offline."
            : ws?.auth.connected && ws?.auth.colabAuthorized
            ? `${selName} (${selParams} · ${selSize} · ${selVram}) · Colab API · ${ws.auth.user?.email ?? ""}`
            : ws?.auth.connected
            ? `${selName} (${selParams} · ${selSize}) · Re-sign in for Colab API scope.`
            : `${selName} (${selParams} · ${selSize}) · Link Google in Settings for Colab compute`,
        ]),
        el("div", { style: "display:flex;gap:8px" }, [offlineBtn, launchBtn]),
      ]),
    ]);

    wrap.appendChild(stepCard);
  } else {
    const goDashBtn = el("button", { class: "btn btn-primary", type: "button" }, [
      svgIcon("plus", 14),
      "Create Project",
    ]);
    goDashBtn.onclick = () => setState({ activeView: "dashboard" });
    wrap.appendChild(
      el("div", { class: "card empty-state" }, [
        el("div", { style: "font-weight:600;color:var(--text-1)" }, ["No project selected"]),
        el("div", { class: "dim" }, ["Create or open an Image ML, Audio ML, or Text / Q&A project to train models on Colab."]),
        goDashBtn,
      ])
    );
  }

  const scopedRuns =
    !showAllProjectRuns && activeProj
      ? state.runs.filter((r) => r.projectId === activeProj.id)
      : state.runs;
  const activeRun =
    scopedRuns.find((r) => r.id === state.selectedRunId) ?? scopedRuns[0] ?? null;

  if (activeRun) {
    const canRetryInPlace = Boolean(
      activeProj && activeRun.projectId === activeProj.id && triggerActiveLaunch
    );
    wrap.appendChild(
      buildRunConsole(
        activeRun,
        savedConsoleRunId === activeRun.id ? savedConsoleState : undefined,
        canRetryInPlace ? triggerActiveLaunch : undefined
      )
    );
  }

  if (state.runs.length > 0) {
    const scopeBtn = el("button", { class: "btn btn-sm", type: "button" }, [
      showAllProjectRuns ? "This Project Only" : `Show All Projects (${state.runs.length})`,
    ]);
    scopeBtn.onclick = () => {
      showAllProjectRuns = !showAllProjectRuns;
      renderTrainingPanel(container);
    };
    const list = el("div", { class: "runs-list" });
    for (const run of scopedRuns) {
      const renameRowBtn = el(
        "button",
        {
          class: "btn btn-sm btn-ghost",
          type: "button",
          title: "Rename model weights",
          style: "padding:3px 6px",
        },
        [svgIcon("edit", 12)]
      );
      renameRowBtn.onclick = (e) => {
        e.stopPropagation();
        openRenameRunModal(run.id, run.modelLabel);
      };
      const delRowBtn = el(
        "button",
        {
          class: "btn btn-sm btn-ghost",
          type: "button",
          title: "Delete training run",
          style: "padding:3px 6px",
        },
        [svgIcon("trash", 12)]
      );
      delRowBtn.onclick = (e) => {
        e.stopPropagation();
        openConfirmModal(
          "Delete Training Run",
          `Delete run ${run.id} (${run.modelLabel}) and its local bundle files from disk?`,
          "Delete Run",
          async () => {
            await deleteTrainingRunWithFeedback(run.id);
          }
        );
      };
      const badgeCls =
        run.state === "completed"
          ? "badge-success"
          : run.state === "failed"
          ? "badge-danger"
          : "badge-accent";
      const row = el(
        "div",
        { class: `run-row${activeRun?.id === run.id ? " active" : ""}` },
        [
          el("span", { style: "font-weight:500" }, [`${run.projectName} · ${run.modelLabel}`]),
          el("div", { style: "display:flex;align-items:center;gap:8px" }, [
            el("span", { class: `badge ${badgeCls}` }, [run.state]),
            el("span", { class: "mono dim" }, [`${Math.round(run.progressPct)}%`]),
            renameRowBtn,
            delRowBtn,
          ]),
        ]
      );
      row.onclick = () => setState({ selectedRunId: run.id });
      list.appendChild(row);
    }
    if (scopedRuns.length > 0 || state.runs.length > scopedRuns.length) {
      wrap.appendChild(
        el("div", { class: "card" }, [
          el("div", { class: "card-header", style: "margin-bottom:10px" }, [
            el("div", { class: "card-title" }, [
              showAllProjectRuns ? "All Training Runs" : `${activeProj?.name ?? "Project"} Training Runs`,
            ]),
            state.runs.length > scopedRuns.length || showAllProjectRuns ? scopeBtn : null,
          ]),
          scopedRuns.length > 0
            ? list
            : el("div", { class: "dim", style: "font-size:12px" }, [
                "No training runs recorded for this project yet.",
              ]),
        ])
      );
    }
  }

  root.appendChild(wrap);
  container.appendChild(root);
  if (savedScrollTop > 0) {
    root.scrollTop = savedScrollTop;
    window.requestAnimationFrame(() => {
      if (root.isConnected) {
        root.scrollTop = savedScrollTop;
      }
    });
  }
}
