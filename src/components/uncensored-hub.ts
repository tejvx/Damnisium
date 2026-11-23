import { api } from "../api";
import { MODULE_TASKS } from "../constants";
import { getState, refreshConnectivity, setState } from "../state";
import type { HfDownloadProgressPayload, HfHubModel, HfHubResponse, LocalGgufModel, ProjectTask } from "../types";
import { el, openModal, showToast, svgIcon } from "../utils/dom";
import { formatBytes } from "../utils/format";

let moduleFilter: "all" | "text" | "image" | "audio" = "all";
let formatFilter: "all" | "GGUF" | "Safetensors" | "ONNX" = "all";
let sortBy: "trending" | "downloads" | "recent" = "trending";
let searchQuery = "";
let hideOver4gb = true;
let hubResponse: HfHubResponse | null = null;
let localGgufs: LocalGgufModel[] = [];
let loadingHub = false;

function defaultTaskForModule(mod: "image" | "audio" | "text"): ProjectTask {
  if (mod === "image") return "classification";
  if (mod === "audio") return "transcription";
  return "qa";
}

async function loadHubData(container: HTMLElement): Promise<void> {
  if (loadingHub) return;
  loadingHub = true;
  renderUncensoredHub(container);
  try {
    const [resp, ggufs] = await Promise.all([
      api.fetchUncensoredHub(moduleFilter, formatFilter, sortBy, searchQuery, hideOver4gb),
      api.listLocalGgufModels(),
    ]);
    hubResponse = resp;
    localGgufs = ggufs;
  } catch (err) {
    showToast(String(err), "error");
  } finally {
    loadingHub = false;
    renderUncensoredHub(container);
  }
}

async function useRepoInColab(repoId: string, moduleHint: string): Promise<void> {
  const state = getState();
  const active = state.activeProject;
  const mod =
    active?.module ??
    (moduleHint === "image" || moduleHint === "audio" || moduleHint === "text" ? moduleHint : "text");
  const task = active?.task ?? defaultTaskForModule(mod);
  const existing = state.workspace?.customModels.find(
    (m) => m.hfRepoId?.toLowerCase() === repoId.toLowerCase() && m.module === mod && m.task === task
  );
  try {
    if (!existing) {
      const leaf = repoId.split("/")[1] ?? repoId;
      await api.registerCustomModel(leaf, mod, task, "huggingface", null, repoId);
      const ws = await api.getWorkspaceInfo();
      setState({ workspace: ws });
    }
    setState({ hubPendingModelRepo: repoId, activeView: "train" });
    showToast(`Selected ${repoId} for Colab training (${MODULE_TASKS[mod].label} · ${task})`, "success");
  } catch (err) {
    showToast(String(err), "error");
  }
}

function openInspectRepoModal(initialModelOrRepo: HfHubModel | string, hostContainer: HTMLElement): void {
  const repoLabel = typeof initialModelOrRepo === "string" ? initialModelOrRepo : initialModelOrRepo.repoId;
  let modalDisposed = false;
  let unlisten: (() => void) | null = null;
  const setDlBtnLabel = (btn: HTMLElement, iconName: string, text: string) => {
    btn.innerHTML = "";
    btn.append(svgIcon(iconName, 12), text);
  };

  openModal(
    `Hugging Face · ${repoLabel}`,
    (close) => {
      const wrap = el("div", { style: "display:flex;flex-direction:column;gap:12px" });
      const body = el("div", { style: "display:flex;flex-direction:column;gap:8px" }, [
        el("div", { class: "dim", style: "font-size:12px" }, ["Inspecting repository files and exact sizes..."]),
      ]);
      const progWrap = el("div", { style: "display:none;flex-direction:column;gap:5px" });
      const progText = el("div", { class: "mono dim", style: "font-size:11px" });
      const progBar = el("div", { class: "progress-fill", style: "width:0%" });
      progWrap.append(progText, el("div", { class: "progress-track" }, [progBar]));
      wrap.append(body, progWrap);

      void api
        .onHfDownloadProgress((p: HfDownloadProgressPayload) => {
          if (modalDisposed || p.repoId.toLowerCase() !== repoLabel.toLowerCase()) return;
          progWrap.style.display = "flex";
          const pct = p.totalBytes ? Math.min(100, Math.round((p.downloadedBytes / p.totalBytes) * 100)) : 50;
          progBar.style.width = `${pct}%`;
          progText.textContent = p.done
            ? `Downloaded ${p.filename} (${formatBytes(p.downloadedBytes)})`
            : `Downloading ${p.filename}: ${formatBytes(p.downloadedBytes)}${p.totalBytes ? ` / ${formatBytes(p.totalBytes)}` : ""}`;
        })
        .then((fn) => {
          if (modalDisposed) {
            fn();
          } else {
            unlisten = fn;
          }
        });

      void api
        .inspectHfRepo(repoLabel)
        .then((model) => {
          if (modalDisposed) return;
          body.innerHTML = "";
          const colabQuickBtn = el(
            "button",
            { class: "btn btn-sm btn-primary", type: "button" },
            [svgIcon("cpu", 12), "Use in Colab"]
          );
          colabQuickBtn.onclick = async () => {
            close();
            await useRepoInColab(model.repoId, model.moduleHint);
          };
          body.appendChild(
            el("div", { style: "display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap" }, [
              el("div", { style: "display:flex;gap:6px;flex-wrap:wrap" }, [
                el("span", { class: "badge badge-accent" }, [model.moduleHint.toUpperCase()]),
                ...model.formatTags.map((ft) => el("span", { class: "badge" }, [ft])),
                model.paramLabel ? el("span", { class: "badge mono" }, [model.paramLabel]) : null,
                model.estimatedGb !== null
                  ? el("span", { class: `badge ${model.over4gb ? "badge-warn" : "badge-success"}` }, [
                      `${model.estimatedGb.toFixed(2)} GB${model.over4gb ? " (>4GB)" : ""}`,
                    ])
                  : null,
              ]),
              colabQuickBtn,
            ])
          );

          if (model.files.length === 0) {
            body.appendChild(
              el("div", { class: "dim", style: "font-size:12px" }, [
                "No downloadable .gguf / .onnx / .safetensors weight files detected in this repository.",
              ])
            );
            return;
          }

          const list = el("div", { style: "display:flex;flex-direction:column;gap:6px;max-height:320px;overflow:auto" });
          for (const f of model.files) {
            const isLarge = f.sizeBytes !== null && f.sizeBytes > 4_294_967_296;
            const dlBtn = el("button", { class: "btn btn-sm", type: "button" }, [
              svgIcon("download", 12),
              "Download",
            ]);
            dlBtn.onclick = async () => {
              dlBtn.setAttribute("disabled", "true");
              setDlBtnLabel(dlBtn, "download", "Downloading...");
              try {
                const savedPath = await api.downloadHfFile(model.repoId, f.rfilename);
                localGgufs = await api.listLocalGgufModels();
                const bundles = await api.listEvalBundles();
                setState({ evalBundles: bundles });
                showToast(`Saved ${f.rfilename}`, "success");
                if (f.format === "GGUF" && !f.isMmproj) {
                  const match = bundles.find((b) => b.onnxPath === savedPath);
                  if (match) {
                    setDlBtnLabel(dlBtn, "flask", "Test Now");
                    dlBtn.removeAttribute("disabled");
                    dlBtn.onclick = () => {
                      close();
                      setState({ selectedEvalBundleId: match.id, activeView: "evaluate" });
                    };
                    return;
                  }
                }
                renderUncensoredHub(hostContainer);
                setDlBtnLabel(dlBtn, "check", "Saved");
              } catch (err) {
                showToast(String(err), "error");
                dlBtn.removeAttribute("disabled");
                setDlBtnLabel(dlBtn, "download", "Download");
              }
            };

            list.appendChild(
              el("div", { class: "hub-file-row" }, [
                el("div", { style: "min-width:0;flex:1" }, [
                  el("div", { class: "mono", style: "font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" }, [
                    f.rfilename,
                  ]),
                  el("div", { class: "dim mono", style: "font-size:10px" }, [
                    `${f.format}${f.isMmproj ? " · mmproj" : ""} · ${f.sizeBytes ? formatBytes(f.sizeBytes) : "Size unknown"}${isLarge ? " · >4GB" : ""}`,
                  ]),
                ]),
                dlBtn,
              ])
            );
          }
          body.appendChild(list);
        })
        .catch((err) => {
          if (modalDisposed) return;
          body.innerHTML = "";
          body.appendChild(
            el("div", { class: "dim", style: "color:#fca5a5;font-size:12px" }, [String(err)])
          );
        });

      return wrap;
    },
    () => {
      modalDisposed = true;
      if (unlisten) {
        unlisten();
        unlisten = null;
      }
    }
  );
}

export function renderUncensoredHub(container: HTMLElement): void {
  const prevScroller = container.querySelector(".view-scroll") as HTMLElement | null;
  const savedScrollTop = prevScroller ? prevScroller.scrollTop : 0;
  const state = getState();
  if (!hubResponse && !loadingHub) {
    void loadHubData(container);
  }

  container.innerHTML = "";
  const root = el("div", { class: "view-scroll" });
  const wrap = el("div", { class: "hub-wrap" });

  const connBadge = el(
    "span",
    { class: `badge ${state.connectivity.online ? "badge-success" : "badge-warn"}` },
    [state.connectivity.online ? "Live Hub Online" : "Offline Presets Only"]
  );
  const reloadBtn = el("button", { class: "btn btn-sm", type: "button", disabled: loadingHub }, [
    svgIcon("refresh", 12),
    loadingHub ? "Syncing..." : "Refresh Feed",
  ]);
  reloadBtn.onclick = async () => {
    await refreshConnectivity();
    void loadHubData(container);
  };

  wrap.appendChild(
    el("div", { class: "hub-header" }, [
      el("div", {}, [
        el("div", { style: "font-size:17px;font-weight:600" }, ["Uncensored Model Hub"]),
        el("div", { class: "dim", style: "font-size:12px" }, [
          "huggingface.co/models?other=uncensored · Trending default with lightweight 4 GB / 4B hardware protection",
        ]),
      ]),
      el("div", { style: "display:flex;align-items:center;gap:8px" }, [connBadge, reloadBtn]),
    ])
  );

  const customInput = el("input", {
    class: "input mono",
    placeholder: "Paste any Hugging Face owner/repo or URL (e.g. bartowski/Dolphin3.0-Llama3.2-1B-GGUF)",
  }) as HTMLInputElement;
  const inspectCustomBtn = el("button", { class: "btn btn-primary", type: "button" }, ["Inspect Repo"]);
  inspectCustomBtn.onclick = () => {
    const val = customInput.value.trim();
    if (!val) {
      showToast("Enter a Hugging Face repo ID or URL.", "info");
      return;
    }
    openInspectRepoModal(val, container);
  };
  customInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") inspectCustomBtn.click();
  });

  const modSegmented = el("div", { class: "segmented" });
  for (const [val, label] of [
    ["all", "All"],
    ["text", "Text / Q&A"],
    ["image", "Image"],
    ["audio", "Audio"],
  ] as const) {
    const b = el("button", { class: moduleFilter === val ? "active" : "", type: "button" }, [label]);
    b.onclick = () => {
      moduleFilter = val;
      void loadHubData(container);
    };
    modSegmented.appendChild(b);
  }

  const fmtSel = el("select", { class: "select", style: "width:130px" }, [
    el("option", { value: "all" }, ["All Formats"]),
    el("option", { value: "GGUF" }, ["GGUF"]),
    el("option", { value: "Safetensors" }, ["Safetensors"]),
    el("option", { value: "ONNX" }, ["ONNX"]),
  ]) as HTMLSelectElement;
  fmtSel.value = formatFilter;
  fmtSel.onchange = () => {
    formatFilter = fmtSel.value as typeof formatFilter;
    void loadHubData(container);
  };

  const sortSel = el("select", { class: "select", style: "width:140px" }, [
    el("option", { value: "trending" }, ["Trending"]),
    el("option", { value: "downloads" }, ["Most Downloads"]),
    el("option", { value: "recent" }, ["Recently Updated"]),
  ]) as HTMLSelectElement;
  sortSel.value = sortBy;
  sortSel.onchange = () => {
    sortBy = sortSel.value as typeof sortBy;
    void loadHubData(container);
  };

  const searchIn = el("input", {
    class: "input",
    style: "width:210px",
    value: searchQuery,
    placeholder: "Filter keyword...",
  }) as HTMLInputElement;
  const searchBtn = el("button", { class: "btn btn-sm", type: "button" }, ["Filter"]);
  const runSearch = () => {
    searchQuery = searchIn.value.trim();
    void loadHubData(container);
  };
  searchBtn.onclick = runSearch;
  searchIn.addEventListener("keydown", (e) => {
    if (e.key === "Enter") runSearch();
  });

  const hide4gbChk = el("input", { type: "checkbox", checked: hideOver4gb }) as HTMLInputElement;
  hide4gbChk.onchange = () => {
    hideOver4gb = hide4gbChk.checked;
    void loadHubData(container);
  };

  wrap.appendChild(
    el("div", { class: "hub-toolbar" }, [
      el("div", { class: "hub-custom-row" }, [customInput, inspectCustomBtn]),
      el("div", { class: "hub-filter-row" }, [
        el("div", { style: "display:flex;align-items:center;gap:8px;flex-wrap:wrap" }, [
          modSegmented,
          fmtSel,
          sortSel,
          searchIn,
          searchBtn,
        ]),
        el("label", { class: "check-row", style: "gap:8px" }, [
          hide4gbChk,
          el("span", {}, ["Hide > 4GB Models"]),
        ]),
      ]),
    ])
  );

  if (hubResponse?.warning) {
    wrap.appendChild(
      el("div", { class: "badge badge-warn", style: "padding:8px 12px;border-radius:var(--radius-md)" }, [
        hubResponse.warning,
      ])
    );
  }

  const models = hubResponse?.models ?? [];
  if (loadingHub && models.length === 0) {
    wrap.appendChild(
      el("div", { class: "empty-state" }, [
        el("div", { class: "spinner" }),
        el("div", { class: "dim" }, ["Fetching live uncensored Hugging Face catalog and inspecting model sizes..."]),
      ])
    );
  } else if (models.length === 0) {
    wrap.appendChild(
      el("div", { class: "empty-state" }, [
        el("div", { style: "font-weight:600" }, ["No models match active filters"]),
        el("div", { class: "dim" }, ["Try unchecking 'Hide > 4GB Models' or clearing the keyword filter."]),
      ])
    );
  } else {
    const grid = el("div", { class: "hub-grid" });
    for (const m of models) {
      const hasGguf = m.formatTags.some((t) => t.toUpperCase() === "GGUF");
      const localMatch = localGgufs.find((g) => g.repoId.toLowerCase() === m.repoId.toLowerCase());

      const inspectBtn = el("button", { class: "btn btn-sm", type: "button" }, [
        svgIcon("download", 12),
        "Files & Download",
      ]);
      inspectBtn.onclick = () => openInspectRepoModal(m, container);

      const colabBtn = el("button", { class: "btn btn-sm btn-primary", type: "button" }, [
        svgIcon("cpu", 12),
        "Use in Colab",
      ]);
      colabBtn.onclick = () => void useRepoInColab(m.repoId, m.moduleHint);

      const testBtn = localMatch
        ? el("button", { class: "btn btn-sm", type: "button" }, [svgIcon("flask", 12), "Test GGUF"])
        : null;
      if (testBtn && localMatch) {
        testBtn.onclick = async () => {
          const bundles = await api.listEvalBundles();
          const b = bundles.find((x) => x.onnxPath === localMatch.localPath);
          setState({
            evalBundles: bundles,
            selectedEvalBundleId: b?.id ?? localMatch.id,
            activeView: "evaluate",
          });
        };
      }

      grid.appendChild(
        el("div", { class: "hub-card" }, [
          el("div", { class: "hub-card-badges" }, [
            m.isPinned ? el("span", { class: "badge badge-accent" }, ["Pinned"]) : null,
            el("span", { class: "badge" }, [m.moduleHint]),
            ...m.formatTags.map((f) => el("span", { class: "badge" }, [f])),
            m.paramLabel ? el("span", { class: "badge mono" }, [m.paramLabel]) : null,
            m.estimatedGb !== null
              ? el("span", { class: `badge ${m.over4gb ? "badge-warn" : "badge-success"}` }, [
                  `~${m.estimatedGb.toFixed(2)} GB`,
                ])
              : null,
            localMatch ? el("span", { class: "badge badge-success" }, ["Local Ready"]) : null,
          ]),
          el("div", { class: "hub-card-repo mono" }, [m.repoId]),
          el("div", { class: "hub-card-summary" }, [m.summary]),
          el("div", { class: "hub-card-footer" }, [
            el("span", { class: "mono dim", style: "font-size:11px" }, [
              hasGguf || m.downloads > 0 ? `${m.downloads.toLocaleString()} dl · ${m.likes} likes` : "Curated preset",
            ]),
            el("div", { style: "display:flex;gap:6px;flex-wrap:wrap" }, [inspectBtn, testBtn, colabBtn]),
          ]),
        ])
      );
    }
    wrap.appendChild(grid);
  }

  if (localGgufs.length > 0) {
    const localCard = el("div", { class: "card", style: "display:flex;flex-direction:column;gap:8px" }, [
      el("div", { class: "card-title" }, ["Downloaded Local GGUF Models (./damnisium-data/models/gguf)"]),
    ]);
    for (const g of localGgufs) {
      const runBtn = el("button", { class: "btn btn-sm btn-primary", type: "button" }, [
        svgIcon("flask", 12),
        "Open in Test & Evaluate",
      ]);
      runBtn.onclick = async () => {
        const bundles = await api.listEvalBundles();
        setState({
          evalBundles: bundles,
          selectedEvalBundleId: g.id,
          activeView: "evaluate",
        });
      };
      localCard.appendChild(
        el("div", { class: "hub-file-row" }, [
          el("div", {}, [
            el("div", { class: "mono", style: "font-size:12px;font-weight:600" }, [g.name]),
            el("div", { class: "dim mono", style: "font-size:11px" }, [
              `${formatBytes(g.sizeBytes)}${g.mmprojPath ? " · mmproj attached" : ""}`,
            ]),
          ]),
          runBtn,
        ])
      );
    }
    wrap.appendChild(localCard);
  }

  root.appendChild(wrap);
  container.appendChild(root);
  if (savedScrollTop > 0) {
    queueMicrotask(() => {
      root.scrollTop = savedScrollTop;
    });
  }
}
