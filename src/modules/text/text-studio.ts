import { api } from "../../api";
import { getState, setState } from "../../state";
import type { Project, SplitType, TextItem } from "../../types";
import { el, openModal, showToast, svgIcon } from "../../utils/dom";
import { estimateTokens } from "../../utils/format";

let lastTextProjectId: string | null = null;
let activeTextId: string | null = null;
let editingIntentClassId: string | null = null;
let searchTerm = "";
let splitFilter: "all" | SplitType = "all";
let currentPage = 0;
let pendingSearchFocus: { start: number; end: number } | null = null;
const PAGE_SIZE = 100;

function openCsvMappingModal(project: Project, filePath: string, headers: string[], onDone: () => void): void {
  openModal("Import CSV Dataset", (close) => {
    const wrap = el("div", { style: "display:flex;flex-direction:column;gap:12px" });
    const buildSelect = (allowNone: boolean, autoMatch: string[]) => {
      const sel = el("select", { class: "select" }) as HTMLSelectElement;
      if (allowNone) sel.appendChild(el("option", { value: "" }, ["(None)"]));
      for (const h of headers) {
        const opt = el("option", { value: h }, [h]) as HTMLOptionElement;
        if (autoMatch.some((m) => h.toLowerCase().includes(m))) {
          opt.selected = true;
        }
        sel.appendChild(opt);
      }
      return sel;
    };

    const promptSel = buildSelect(false, ["prompt", "question", "text", "input"]);
    const compSel = buildSelect(true, ["completion", "answer", "response", "output"]);
    const sysSel = buildSelect(true, ["system"]);
    const lblSel = buildSelect(true, ["label", "intent", "category", "class"]);

    const importBtn = el("button", { class: "btn btn-primary", type: "button" }, ["Import Rows"]);
    importBtn.onclick = async () => {
      importBtn.setAttribute("disabled", "true");
      try {
        const updated = await api.importTextCsv(
          project.id,
          filePath,
          promptSel.value,
          compSel.value || null,
          sysSel.value || null,
          lblSel.value || null
        );
        setState({ activeProject: updated, projects: await api.listProjects() });
        close();
        onDone();
        showToast("Imported CSV rows", "success");
      } catch (err) {
        showToast(String(err), "error");
        importBtn.removeAttribute("disabled");
      }
    };

    wrap.append(
      el("div", { class: "field" }, [
        el("div", { class: "field-label" }, [
          project.task === "intent" ? "Text Column" : "Prompt / Question Column",
        ]),
        promptSel,
      ])
    );
    if (project.task !== "intent") {
      wrap.append(
        el("div", { class: "field" }, [
          el("div", { class: "field-label" }, ["Completion / Answer Column"]),
          compSel,
        ])
      );
    }
    if (project.task === "qa") {
      wrap.append(
        el("div", { class: "field" }, [
          el("div", { class: "field-label" }, ["System Column"]),
          sysSel,
        ])
      );
    }
    if (project.task === "intent") {
      wrap.append(
        el("div", { class: "field" }, [
          el("div", { class: "field-label" }, ["Intent Label Column"]),
          lblSel,
        ])
      );
    }
    wrap.append(el("div", { class: "modal-actions" }, [importBtn]));
    return wrap;
  });
}

export function renderTextStudio(container: HTMLElement): void {
  const state = getState();
  const project = state.activeProject;
  if (!project) return;

  if (lastTextProjectId !== project.id) {
    lastTextProjectId = project.id;
    activeTextId = null;
    editingIntentClassId = null;
    searchTerm = "";
    splitFilter = "all";
    currentPage = 0;
    pendingSearchFocus = null;
  }

  const filtered = project.textItems.filter((row) => {
    if (splitFilter !== "all" && row.split !== splitFilter) return false;
    if (!searchTerm.trim()) return true;
    const q = searchTerm.toLowerCase();
    return (
      row.prompt.toLowerCase().includes(q) ||
      row.completion.toLowerCase().includes(q) ||
      row.system.toLowerCase().includes(q)
    );
  });

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  currentPage = Math.min(currentPage, totalPages - 1);
  const pageRows = filtered.slice(currentPage * PAGE_SIZE, (currentPage + 1) * PAGE_SIZE);

  if (activeTextId && !project.textItems.some((r) => r.id === activeTextId)) {
    activeTextId = null;
  }
  const editingRow: TextItem =
    project.textItems.find((r) => r.id === activeTextId) ?? {
      id: "",
      prompt: "",
      completion: "",
      system: "",
      intentClassId: project.classes[0]?.id ?? null,
      split: "train",
      createdAt: "",
    };

  container.innerHTML = "";
  const layout = el("div", { class: "text-layout" });

  const mainPane = el("div", { class: "text-main" });
  const newBtn = el("button", { class: "btn btn-sm btn-primary", type: "button" }, [
    svgIcon("plus", 12),
    "New Row",
  ]);
  newBtn.onclick = () => {
    activeTextId = null;
    renderTextStudio(container);
  };

  const impCsvBtn = el("button", { class: "btn btn-sm", type: "button" }, ["Import CSV"]);
  impCsvBtn.onclick = async () => {
    const sel = await api.pickFiles([{ name: "CSV", extensions: ["csv"] }], false);
    if (typeof sel !== "string") return;
    try {
      const headers = await api.previewCsvHeaders(sel);
      openCsvMappingModal(project, sel, headers, () => renderTextStudio(container));
    } catch (err) {
      showToast(String(err), "error");
    }
  };

  const impJsonlBtn = el("button", { class: "btn btn-sm", type: "button" }, ["Import JSONL"]);
  impJsonlBtn.onclick = async () => {
    const sel = await api.pickFiles([{ name: "JSONL", extensions: ["jsonl", "json"] }], false);
    if (typeof sel !== "string") return;
    try {
      const updated = await api.importTextJsonl(project.id, sel);
      setState({ activeProject: updated, projects: await api.listProjects() });
      renderTextStudio(container);
      showToast("Imported JSONL dataset", "success");
    } catch (err) {
      showToast(String(err), "error");
    }
  };

  const expBtn = el("button", { class: "btn btn-sm", type: "button" }, ["Export JSONL"]);
  expBtn.onclick = async () => {
    const dest = await api.saveFile(`${project.name}.jsonl`, [
      { name: "JSONL", extensions: ["jsonl"] },
      { name: "CSV", extensions: ["csv"] },
    ]);
    if (typeof dest !== "string") return;
    const fmt = dest.toLowerCase().endsWith(".csv") ? "csv" : "jsonl";
    try {
      const count = await api.exportTextDataset(project.id, fmt, dest);
      showToast(`Exported ${count} rows`, "success");
    } catch (err) {
      showToast(String(err), "error");
    }
  };

  const searchIn = el("input", {
    class: "input",
    style: "max-width:200px",
    placeholder: "Search rows...",
    value: searchTerm,
  }) as HTMLInputElement;
  searchIn.oninput = () => {
    pendingSearchFocus = {
      start: searchIn.selectionStart ?? searchIn.value.length,
      end: searchIn.selectionEnd ?? searchIn.value.length,
    };
    searchTerm = searchIn.value;
    currentPage = 0;
    renderTextStudio(container);
  };

  const splitSel = el("select", { class: "select", style: "max-width:105px" }) as HTMLSelectElement;
  for (const optVal of ["all", "train", "val", "test"] as const) {
    const o = el("option", { value: optVal }, [optVal === "all" ? "All Splits" : optVal]) as HTMLOptionElement;
    if (splitFilter === optVal) o.selected = true;
    splitSel.appendChild(o);
  }
  splitSel.onchange = () => {
    splitFilter = splitSel.value as "all" | SplitType;
    currentPage = 0;
    renderTextStudio(container);
  };

  const toolbar = el("div", { class: "text-toolbar" }, [
    el("div", { style: "display:flex;gap:6px;align-items:center" }, [
      newBtn,
      impCsvBtn,
      impJsonlBtn,
      expBtn,
    ]),
    el("div", { style: "display:flex;gap:6px;align-items:center" }, [searchIn, splitSel]),
  ]);

  const table = el("table", { class: "data-table" });
  const thead = el("thead", {}, [
    el("tr", {}, [
      el("th", { style: "width:46px" }, ["#"]),
      el("th", { style: "width:64px" }, ["Split"]),
      el("th", {}, [project.task === "intent" ? "Input Text" : "Prompt / Question"]),
      el("th", {}, [project.task === "intent" ? "Intent Class" : "Completion / Answer"]),
      el("th", { style: "width:64px" }, ["Tokens"]),
    ]),
  ]);
  const tbody = el("tbody");

  pageRows.forEach((row, idx) => {
    const clsName =
      project.classes.find((c) => c.id === row.intentClassId)?.name ?? "Unassigned";
    const tokens = estimateTokens(`${row.system} ${row.prompt} ${row.completion}`);
    const tr = el("tr", { class: row.id === activeTextId ? "active" : "" }, [
      el("td", { class: "mono dim" }, [String(currentPage * PAGE_SIZE + idx + 1)]),
      el("td", {}, [el("span", { class: "badge" }, [row.split])]),
      el("td", {}, [el("div", { class: "text-cell-clip" }, [row.prompt])]),
      el("td", {}, [
        el("div", { class: "text-cell-clip" }, [
          project.task === "intent" ? clsName : row.completion,
        ]),
      ]),
      el("td", { class: "mono dim" }, [String(tokens)]),
    ]);
    tr.onclick = () => {
      activeTextId = row.id;
      renderTextStudio(container);
    };
    tbody.appendChild(tr);
  });

  table.append(thead, tbody);
  const tableWrap = el("div", { class: "text-table-wrap" }, [table]);

  const prevBtn = el("button", {
    class: "btn btn-sm",
    type: "button",
    disabled: currentPage === 0,
  }, ["Prev"]);
  prevBtn.onclick = () => {
    if (currentPage > 0) {
      currentPage--;
      renderTextStudio(container);
    }
  };
  const nextBtn = el("button", {
    class: "btn btn-sm",
    type: "button",
    disabled: currentPage + 1 >= totalPages,
  }, ["Next"]);
  nextBtn.onclick = () => {
    if (currentPage + 1 < totalPages) {
      currentPage++;
      renderTextStudio(container);
    }
  };

  const pagBar = el("div", { class: "text-pagination" }, [
    el("span", { class: "mono" }, [`${filtered.length} rows`]),
    el("div", { style: "display:flex;align-items:center;gap:8px" }, [
      prevBtn,
      el("span", { class: "mono" }, [`Page ${currentPage + 1} / ${totalPages}`]),
      nextBtn,
    ]),
  ]);

  mainPane.append(toolbar, tableWrap, pagBar);

  const editorPane = el("div", { class: "text-editor-pane" });
  const sysIn = el("textarea", {
    class: "textarea",
    style: "min-height:58px",
    placeholder: "Optional system instruction",
  }) as HTMLTextAreaElement;
  sysIn.value = editingRow.system;

  const promptIn = el("textarea", {
    class: "textarea",
    style: "min-height:110px",
    placeholder:
      project.task === "intent"
        ? "Enter user utterance or text sample"
        : project.task === "qa"
        ? "Enter question"
        : "Enter prompt",
  }) as HTMLTextAreaElement;
  promptIn.value = editingRow.prompt;

  const compIn = el("textarea", {
    class: "textarea",
    style: "min-height:140px",
    placeholder: project.task === "qa" ? "Enter ideal answer" : "Enter completion",
  }) as HTMLTextAreaElement;
  compIn.value = editingRow.completion;

  const intentSel = el("select", { class: "select" }) as HTMLSelectElement;
  intentSel.appendChild(el("option", { value: "" }, ["Select Intent"]));
  for (const c of project.classes) {
    const o = el("option", { value: c.id }, [c.name]) as HTMLOptionElement;
    if (editingRow.intentClassId === c.id) o.selected = true;
    intentSel.appendChild(o);
  }

  const rowSplitSel = el("select", { class: "select" }) as HTMLSelectElement;
  for (const sp of ["train", "val", "test"] as SplitType[]) {
    const o = el("option", { value: sp }, [sp]) as HTMLOptionElement;
    if (editingRow.split === sp) o.selected = true;
    rowSplitSel.appendChild(o);
  }

  const saveRowBtn = el("button", { class: "btn btn-primary", type: "button" }, [
    activeTextId ? "Update Row (Ctrl+Enter)" : "Add Row (Ctrl+Enter)",
  ]);
  const submitRow = async () => {
    if (!promptIn.value.trim()) {
      showToast("Prompt / text cannot be empty.", "error");
      return;
    }
    try {
      const updated = await api.upsertTextItem(project.id, {
        id: editingRow.id,
        prompt: promptIn.value,
        completion: compIn.value,
        system: sysIn.value,
        intentClassId: project.task === "intent" ? intentSel.value || null : null,
        split: rowSplitSel.value as SplitType,
        createdAt: editingRow.createdAt,
      });
      activeTextId = null;
      setState({ activeProject: updated, projects: await api.listProjects() });
      renderTextStudio(container);
      showToast("Saved row", "success");
    } catch (err) {
      showToast(String(err), "error");
    }
  };
  saveRowBtn.onclick = () => void submitRow();
  const handleSaveShortcut = (e: KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      void submitRow();
    }
  };
  promptIn.onkeydown = handleSaveShortcut;
  compIn.onkeydown = handleSaveShortcut;
  sysIn.onkeydown = handleSaveShortcut;

  const delRowBtn = el("button", { class: "btn btn-danger", type: "button" }, ["Delete Row"]);
  delRowBtn.onclick = async () => {
    if (!activeTextId) return;
    try {
      const updated = await api.deleteTextItem(project.id, activeTextId);
      activeTextId = null;
      setState({ activeProject: updated, projects: await api.listProjects() });
      renderTextStudio(container);
    } catch (err) {
      showToast(String(err), "error");
    }
  };

  editorPane.appendChild(
    el("div", { class: "panel-section" }, [
      el("div", { class: "panel-section-title" }, [activeTextId ? "Edit Sample" : "New Sample"]),
      project.task === "qa"
        ? el("div", { class: "field" }, [
            el("div", { class: "field-label" }, ["System Prompt"]),
            sysIn,
          ])
        : null,
      el("div", { class: "field" }, [
        el("div", { class: "field-label" }, [
          project.task === "intent" ? "Text" : project.task === "qa" ? "Question" : "Prompt",
        ]),
        promptIn,
      ]),
      project.task !== "intent"
        ? el("div", { class: "field" }, [
            el("div", { class: "field-label" }, [
              project.task === "qa" ? "Answer" : "Completion",
            ]),
            compIn,
          ])
        : el("div", { class: "field" }, [
            el("div", { class: "field-label" }, ["Intent Class"]),
            intentSel,
          ]),
      el("div", { class: "field" }, [el("div", { class: "field-label" }, ["Split"]), rowSplitSel]),
      el("div", { style: "display:flex;gap:8px" }, [
        saveRowBtn,
        activeTextId ? delRowBtn : null,
      ]),
    ])
  );

  if (project.task === "intent") {
    const clsIn = el("input", { class: "input", placeholder: "New intent label" }) as HTMLInputElement;
    const addClsBtn = el("button", { class: "btn btn-sm", type: "button" }, ["Add"]);
    const addIntent = async () => {
      if (!clsIn.value.trim()) return;
      try {
        const updated = await api.addLabelClass(project.id, clsIn.value.trim());
        setState({ activeProject: updated, projects: await api.listProjects() });
        renderTextStudio(container);
      } catch (err) {
        showToast(String(err), "error");
      }
    };
    addClsBtn.onclick = () => void addIntent();
    clsIn.onkeydown = (e) => {
      if (e.key === "Enter") void addIntent();
    };

    const balanceWrap = el("div", { style: "display:flex;flex-direction:column;gap:8px" });
    const maxCount = Math.max(
      1,
      ...project.classes.map(
        (c) => project.textItems.filter((t) => t.intentClassId === c.id).length
      )
    );
    for (const c of project.classes) {
      const cnt = project.textItems.filter((t) => t.intentClassId === c.id).length;
      const pct = Math.round((cnt / maxCount) * 100);
      if (editingIntentClassId === c.id) {
        const renameIn = el("input", {
          class: "input",
          style: "height:24px;padding:2px 6px;font-size:12px;flex:1;min-width:0",
          value: c.name,
        }) as HTMLInputElement;
        const saveRename = async () => {
          const nextName = renameIn.value.trim();
          if (!nextName) {
            showToast("Class name cannot be empty.", "error");
            return;
          }
          try {
            const updated = await api.updateLabelClass(project.id, c.id, nextName, c.color);
            editingIntentClassId = null;
            setState({ activeProject: updated, projects: await api.listProjects() });
            renderTextStudio(container);
            showToast(`Renamed intent class to ${nextName}`, "success");
          } catch (err) {
            showToast(String(err), "error");
          }
        };
        const saveBtn = el(
          "button",
          { class: "btn btn-sm btn-ghost", type: "button", title: "Save class name" },
          [svgIcon("check", 11)]
        );
        saveBtn.onclick = () => void saveRename();
        renameIn.onkeydown = (e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            void saveRename();
          } else if (e.key === "Escape") {
            e.preventDefault();
            editingIntentClassId = null;
            renderTextStudio(container);
          }
        };
        balanceWrap.appendChild(
          el("div", { style: "display:flex;align-items:center;gap:6px" }, [renameIn, saveBtn])
        );
        setTimeout(() => {
          renameIn.focus();
          renameIn.select();
        }, 10);
        continue;
      }

      const editBtn = el(
        "button",
        { class: "btn btn-sm btn-ghost", type: "button", title: "Rename intent class" },
        [svgIcon("edit", 11)]
      );
      editBtn.onclick = () => {
        editingIntentClassId = c.id;
        renderTextStudio(container);
      };

      const delBtn = el(
        "button",
        { class: "btn btn-sm btn-ghost", type: "button", title: "Delete intent class" },
        [svgIcon("trash", 11)]
      );
      delBtn.onclick = async () => {
        try {
          const updated = await api.deleteLabelClass(project.id, c.id);
          setState({ activeProject: updated, projects: await api.listProjects() });
          renderTextStudio(container);
        } catch (err) {
          showToast(String(err), "error");
        }
      };

      const labelSpan = el("span", { title: "Double-click to rename", style: "cursor:pointer" }, [
        c.name,
      ]);
      labelSpan.ondblclick = () => {
        editingIntentClassId = c.id;
        renderTextStudio(container);
      };

      balanceWrap.appendChild(
        el("div", {}, [
          el("div", { style: "display:flex;justify-content:space-between;align-items:center;font-size:11px;gap:6px" }, [
            labelSpan,
            el("div", { style: "display:flex;align-items:center;gap:4px" }, [
              el("span", { class: "mono dim" }, [String(cnt)]),
              editBtn,
              delBtn,
            ]),
          ]),
          el("div", { class: "balance-bar" }, [
            el("div", {
              class: "balance-fill",
              style: `width:${pct}%;background:${c.color}`,
            }),
          ]),
        ])
      );
    }

    editorPane.appendChild(
      el("div", { class: "panel-section" }, [
        el("div", { class: "panel-section-title" }, ["Intent Classes & Balance"]),
        el("div", { style: "display:flex;gap:5px" }, [clsIn, addClsBtn]),
        balanceWrap,
      ])
    );
  }

  const textAug = { ...project.textAug };
  const shufChk = el("input", {
    type: "checkbox",
    checked: textAug.shufflePromptOrder,
  }) as HTMLInputElement;
  const caseChk = el("input", {
    type: "checkbox",
    checked: textAug.caseJitter,
  }) as HTMLInputElement;
  shufChk.onchange = () => {
    textAug.shufflePromptOrder = shufChk.checked;
  };
  caseChk.onchange = () => {
    textAug.caseJitter = caseChk.checked;
  };

  const saveTextAugBtn = el("button", { class: "btn btn-sm", type: "button" }, [
    "Save Augmentations",
  ]);
  saveTextAugBtn.onclick = async () => {
    try {
      const updated = await api.updateProjectSettings(
        project.id,
        project.imageAug,
        project.audioAug,
        textAug,
        project.splitRatio,
        false
      );
      setState({ activeProject: updated });
      showToast("Saved text augmentations", "success");
    } catch (err) {
      showToast(String(err), "error");
    }
  };

  editorPane.appendChild(
    el("div", { class: "panel-section" }, [
      el("div", { class: "panel-section-title" }, ["Text Augmentations"]),
      el("label", { class: "check-row" }, [el("span", {}, ["Shuffle Prompt Order"]), shufChk]),
      el("label", { class: "check-row" }, [el("span", {}, ["Case Normalization Jitter"]), caseChk]),
      saveTextAugBtn,
    ])
  );

  layout.append(mainPane, editorPane);
  container.appendChild(layout);
  if (pendingSearchFocus) {
    const sel = pendingSearchFocus;
    pendingSearchFocus = null;
    searchIn.focus();
    searchIn.setSelectionRange(sel.start, sel.end);
  }
}
