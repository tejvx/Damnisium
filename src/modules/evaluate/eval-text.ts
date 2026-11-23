import { api } from "../../api";
import type { EvalBundleInfo, LlamaRuntimeStatus, TextEvalRow } from "../../types";
import { el, showToast, svgIcon } from "../../utils/dom";
import {
  type EvalClassificationScore,
  runDistilBertIntent,
  runQwenGeneration,
  setGgufRuntimeActive,
} from "../../utils/test-runtime";

interface ChatTurn {
  role: "user" | "assistant";
  content: string;
  meta: string;
}

interface BatchEvalRowResult {
  prompt: string;
  expected: string;
  predicted: string;
  matched: boolean;
  latencyMs: number;
}

let chatTurns: ChatTurn[] = [];
let intentSnippet = "";
let intentScores: EvalClassificationScore[] = [];
let intentLatency = 0;
let batchRows: BatchEvalRowResult[] = [];
let batchRunning = false;
let batchCancel = false;
let llamaStatusCache: LlamaRuntimeStatus | null = null;
let installingLlama = false;
let draftSystemPrompt = "";
let draftTemp = 0.2;
let draftMaxTok = 128;
let activeRenderBundleId: string | null = null;

export function resetTextEvalState(): void {
  chatTurns = [];
  intentScores = [];
  batchRows = [];
  batchRunning = false;
  batchCancel = true;
  draftSystemPrompt = "";
  draftTemp = 0.2;
  draftMaxTok = 128;
  activeRenderBundleId = null;
}

function answersMatch(expected: string, predicted: string, isIntent: boolean): boolean {
  const e = expected.trim().toLowerCase();
  const p = predicted.trim().toLowerCase();
  if (!e) return false;
  if (isIntent) {
    return e === p;
  }
  return p === e || p.includes(e) || e.includes(p);
}

export function renderTextEvalPane(
  container: HTMLElement,
  bundle: EvalBundleInfo
): void {
  container.innerHTML = "";
  activeRenderBundleId = bundle.id;
  const isIntent = bundle.runtimeKind === "hf_text_cls";
  const isGguf = bundle.runtimeKind === "llama_gguf";
  const onnxDir = bundle.onnxDir ?? "";

  if (isGguf && !llamaStatusCache) {
    const requestedBundleId = bundle.id;
    void api.getLlamaRuntimeStatus().then((st) => {
      llamaStatusCache = st;
      if (container.isConnected && activeRenderBundleId === requestedBundleId) {
        renderTextEvalPane(container, bundle);
      }
    });
  }

  const leftCol = el("div", { class: "eval-pane-col" });
  const leftBody = el("div", { class: "eval-scroll" });

  if (isIntent) {
    const inputArea = el("textarea", {
      class: "textarea",
      style: "min-height:110px",
      placeholder: "Enter text snippet to classify intent...",
    }) as HTMLTextAreaElement;
    inputArea.value = intentSnippet;

    const runBtn = el("button", { class: "btn btn-primary", type: "button" }, [
      "Classify Intent",
    ]);
    inputArea.oninput = () => {
      intentSnippet = inputArea.value;
    };
    inputArea.onkeydown = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        runBtn.click();
      }
    };
    runBtn.onclick = async () => {
      if (!onnxDir) {
        showToast("No ONNX bundle directory found for DistilBERT.", "error");
        return;
      }
      const txt = inputArea.value.trim();
      if (!txt) return;
      intentSnippet = txt;
      runBtn.setAttribute("disabled", "true");
      try {
        const res = await runDistilBertIntent(onnxDir, bundle.onnxPath, txt);
        intentScores = res.scores;
        intentLatency = res.latencyMs;
        renderTextEvalPane(container, bundle);
      } catch (err) {
        showToast(String(err), "error");
      } finally {
        runBtn.removeAttribute("disabled");
      }
    };

    const confList = el("div", { class: "eval-conf-list" });
    for (const s of intentScores) {
      const pct = Math.round(s.score * 100);
      confList.appendChild(
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

    leftBody.append(
      el("div", { class: "panel-section-title" }, [
        "Interactive Intent Playground",
        intentLatency > 0 ? el("span", { class: "mono" }, [`${intentLatency} ms`]) : null,
      ]),
      inputArea,
      el("div", { style: "display:flex;justify-content:flex-end" }, [runBtn]),
      confList.children.length > 0
        ? confList
        : el("div", { class: "dim", style: "font-size:12px" }, ["Enter text and click Classify Intent."])
    );
  } else {
    const sysIn = el("input", {
      class: "input",
      placeholder: "Optional system prompt",
    }) as HTMLInputElement;
    sysIn.value = draftSystemPrompt;
    sysIn.oninput = () => {
      draftSystemPrompt = sysIn.value;
    };
    const promptIn = el("textarea", {
      class: "textarea",
      style: "min-height:84px",
      placeholder: "Ask a question or enter a prompt... (Ctrl+Enter to generate)",
    }) as HTMLTextAreaElement;

    const tempVal = el("span", { class: "mono" }, [draftTemp.toFixed(2)]);
    const tempSl = el("input", {
      type: "range",
      min: "0",
      max: "1.2",
      step: "0.05",
      value: String(draftTemp),
    }) as HTMLInputElement;
    tempSl.oninput = () => {
      draftTemp = Number(tempSl.value);
      tempVal.textContent = draftTemp.toFixed(2);
    };

    const tokIn = el("input", {
      class: "input mono",
      type: "number",
      min: "16",
      max: "512",
      value: String(draftMaxTok),
    }) as HTMLInputElement;
    tokIn.oninput = () => {
      const parsed = Number(tokIn.value);
      if (Number.isFinite(parsed) && parsed >= 16) {
        draftMaxTok = parsed;
      }
    };

    const sendBtn = el("button", { class: "btn btn-primary", type: "button" }, ["Generate"]);
    promptIn.onkeydown = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        sendBtn.click();
      }
    };
    const clearBtn = el("button", { class: "btn btn-sm", type: "button" }, ["Clear"]);
    clearBtn.onclick = () => {
      chatTurns = [];
      renderTextEvalPane(container, bundle);
    };

    sendBtn.onclick = async () => {
      if (!isGguf && !onnxDir) {
        showToast("No ONNX bundle directory found for text generation.", "error");
        return;
      }
      const prompt = promptIn.value.trim();
      if (!prompt) return;
      draftSystemPrompt = sysIn.value;
      draftMaxTok = Number(tokIn.value) || 128;
      sendBtn.setAttribute("disabled", "true");
      chatTurns.push({ role: "user", content: prompt, meta: "" });
      let pushedUser = true;
      try {
        if (isGguf) {
          const res = await api.runGgufChat(
            bundle.onnxPath,
            bundle.onnxDir,
            draftSystemPrompt,
            prompt,
            draftMaxTok,
            draftTemp
          );
          setGgufRuntimeActive(true);
          chatTurns.push({
            role: "assistant",
            content: res.output || "(Empty output)",
            meta: `${res.durationMs} ms · ${res.tokensPerSec.toFixed(1)} tok/s · GGUF`,
          });
        } else {
          const res = await runQwenGeneration(
            onnxDir,
            bundle.onnxPath,
            prompt,
            draftSystemPrompt,
            draftMaxTok,
            draftTemp
          );
          chatTurns.push({
            role: "assistant",
            content: res.text || "(Empty output)",
            meta: `${res.latencyMs} ms · ${res.tokPerSec} tok/s`,
          });
        }
        pushedUser = false;
        renderTextEvalPane(container, bundle);
      } catch (err) {
        if (pushedUser) {
          chatTurns.pop();
        }
        showToast(String(err), "error");
      } finally {
        sendBtn.removeAttribute("disabled");
      }
    };

    let ggufSetupCard: HTMLElement | null = null;
    if (isGguf && llamaStatusCache && !llamaStatusCache.installed) {
      const installBtn = el(
        "button",
        { class: "btn btn-sm btn-primary", type: "button", disabled: installingLlama },
        [
          svgIcon("download", 12),
          installingLlama
            ? "Installing llama.cpp..."
            : `Install llama.cpp Runtime (~${llamaStatusCache.downloadSizeMb} MB)`,
        ]
      );
      installBtn.onclick = async () => {
        installingLlama = true;
        renderTextEvalPane(container, bundle);
        try {
          llamaStatusCache = await api.installLlamaRuntime();
          showToast(`Installed llama.cpp (${llamaStatusCache.releaseTag ?? "CPU x64"})`, "success");
        } catch (err) {
          showToast(String(err), "error");
        } finally {
          installingLlama = false;
          renderTextEvalPane(container, bundle);
        }
      };
      ggufSetupCard = el("div", { class: "preflight-box" }, [
        el("div", { style: "font-weight:600;font-size:12px" }, ["On-Demand GGUF CPU Runtime"]),
        el("div", { class: "dim", style: "font-size:11px" }, [
          "Downloads official Windows x64 llama-server.exe into ./damnisium-data/runtime/llama/.",
        ]),
        installBtn,
      ]);
      sendBtn.setAttribute("disabled", "true");
    }

    const chatList = el("div", { style: "display:flex;flex-direction:column;gap:8px" });
    for (const t of chatTurns) {
      chatList.appendChild(
        el("div", { class: `chat-bubble${t.role === "user" ? " user" : ""}` }, [
          el("div", {}, [t.content]),
          t.meta
            ? el("div", { class: "dim mono", style: "font-size:10px;margin-top:5px" }, [t.meta])
            : null,
        ])
      );
    }

    leftBody.append(
      el("div", { class: "panel-section-title" }, [
        isGguf ? "Local GGUF Console (llama.cpp)" : "Interactive Prompt / Q&A Console",
        clearBtn,
      ])
    );
    if (ggufSetupCard) {
      leftBody.appendChild(ggufSetupCard);
    }
    leftBody.append(
      el("div", { class: "field" }, [el("div", { class: "field-label" }, ["System"]), sysIn]),
      el("div", { class: "grid-2" }, [
        el("div", { class: "slider-row" }, [
          el("div", { class: "slider-head" }, [el("span", {}, ["Temperature"]), tempVal]),
          tempSl,
        ]),
        el("div", { class: "field" }, [
          el("div", { class: "field-label" }, ["Max Tokens"]),
          tokIn,
        ]),
      ]),
      promptIn,
      el("div", { style: "display:flex;justify-content:flex-end" }, [sendBtn]),
      chatList
    );
  }
  leftCol.appendChild(leftBody);

  const rightCol = el("div", { class: "eval-pane-col" });
  const rightBody = el("div", { class: "eval-scroll" });

  const uploadBtn = el("button", { class: "btn btn-sm btn-primary", type: "button" }, [
    svgIcon("upload", 12),
    "Run Batch CSV / JSONL",
  ]);
  const stopBtn = el("button", {
    class: "btn btn-sm btn-danger",
    type: "button",
    style: batchRunning ? "" : "display:none",
  }, ["Stop"]);
  stopBtn.onclick = () => {
    batchCancel = true;
  };
  const progLabel = el("span", { class: "mono dim", style: "font-size:11px" });

  const runBatchEval = async (rows: TextEvalRow[]) => {
    if (!isGguf && !onnxDir) {
      showToast("No ONNX bundle directory found for this model.", "error");
      return;
    }
    batchRunning = true;
    batchCancel = false;
    batchRows = [];
    stopBtn.style.display = "inline-flex";
    try {
      for (let i = 0; i < rows.length; i++) {
        if (batchCancel) break;
        progLabel.textContent = `${i + 1}/${rows.length}`;
        const row = rows[i];
        let predicted = "";
        let latencyMs = 0;
        if (isIntent) {
          const out = await runDistilBertIntent(onnxDir, bundle.onnxPath, row.prompt);
          predicted = out.scores[0]?.label ?? "";
          latencyMs = out.latencyMs;
        } else if (isGguf) {
          const out = await api.runGgufChat(
            bundle.onnxPath,
            bundle.onnxDir,
            row.system,
            row.prompt,
            96,
            0.05
          );
          setGgufRuntimeActive(true);
          predicted = out.output;
          latencyMs = out.durationMs;
        } else {
          const out = await runQwenGeneration(
            onnxDir,
            bundle.onnxPath,
            row.prompt,
            row.system,
            96,
            0.05
          );
          predicted = out.text;
          latencyMs = out.latencyMs;
        }
        batchRows.push({
          prompt: row.prompt,
          expected: row.expected,
          predicted,
          matched: answersMatch(row.expected, predicted, isIntent),
          latencyMs,
        });
      }
    } catch (err) {
      showToast(String(err), "error");
    } finally {
      batchRunning = false;
      if (container.isConnected && activeRenderBundleId === bundle.id) {
        renderTextEvalPane(container, bundle);
      }
    }
  };

  uploadBtn.onclick = async () => {
    const sel = await api.pickFiles(
      [{ name: "Eval Dataset", extensions: ["csv", "jsonl"] }],
      false
    );
    if (typeof sel !== "string") return;
    try {
      const rows = await api.readTextEvalFile(sel);
      if (rows.length === 0) {
        showToast("No valid rows found in evaluation file.", "info");
        return;
      }
      void runBatchEval(rows);
    } catch (err) {
      showToast(String(err), "error");
    }
  };

  const matchedCount = batchRows.filter((r) => r.matched).length;
  const accuracyPct =
    batchRows.length > 0 ? ((matchedCount / batchRows.length) * 100).toFixed(1) : "--";

  const table = el("table", { class: "data-table" });
  table.appendChild(
    el("thead", {}, [
      el("tr", {}, [
        el("th", {}, ["Prompt / Input"]),
        el("th", {}, ["Expected"]),
        el("th", {}, ["Predicted"]),
        el("th", { style: "width:70px" }, ["Match"]),
      ]),
    ])
  );
  const tbody = el("tbody");
  for (const r of batchRows) {
    tbody.appendChild(
      el("tr", {}, [
        el("td", {}, [el("div", { class: "text-cell-clip" }, [r.prompt])]),
        el("td", {}, [el("div", { class: "text-cell-clip" }, [r.expected])]),
        el("td", {}, [el("div", { class: "text-cell-clip" }, [r.predicted])]),
        el("td", {}, [
          el(
            "span",
            { class: `badge ${r.matched ? "badge-success" : "badge-danger"}` },
            [r.matched ? "PASS" : "MISS"]
          ),
        ]),
      ])
    );
  }
  table.appendChild(tbody);

  rightBody.append(
    el("div", { class: "panel-section-title" }, [
      "Batch Evaluation",
      el("div", { style: "display:flex;gap:6px;align-items:center" }, [
        progLabel,
        stopBtn,
        uploadBtn,
      ]),
    ]),
    el("div", { style: "display:flex;gap:8px;flex-wrap:wrap" }, [
      el("div", { class: "metric-chip" }, [
        el("span", { class: "dim" }, ["Accuracy"]),
        el("span", { class: "metric-chip-val" }, [`${accuracyPct}%`]),
      ]),
      el("div", { class: "metric-chip" }, [
        el("span", { class: "dim" }, ["Matched"]),
        el("span", { class: "metric-chip-val" }, [`${matchedCount}/${batchRows.length}`]),
      ]),
    ]),
    table
  );
  rightCol.appendChild(rightBody);

  container.appendChild(el("div", { class: "eval-text-grid" }, [leftCol, rightCol]));
}
