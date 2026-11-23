import { api } from "../api";
import { MODULE_TASKS } from "../constants";
import { getState, removeRunFromStore, setState, upsertRunInStore } from "../state";
import type { ProjectModule, ProjectSummary, ProjectTask, TrainingRun } from "../types";
import { el, openConfirmModal, openModal, showToast, svgIcon } from "../utils/dom";
import { formatIsoShort } from "../utils/format";

export function openRenameRunModal(
  runId: string,
  currentLabel: string,
  onRenamed?: (updated: TrainingRun) => void
): void {
  openModal("Rename Model / Trained Weights", (close) => {
    const wrap = el("div", { style: "display:flex;flex-direction:column;gap:12px" });
    const nameInput = el("input", {
      class: "input",
      type: "text",
      value: currentLabel,
      placeholder: "e.g. bottle-detector-v2",
    }) as HTMLInputElement;

    const cancelBtn = el("button", { class: "btn", type: "button" }, ["Cancel"]);
    cancelBtn.onclick = close;
    const saveBtn = el("button", { class: "btn btn-primary", type: "button" }, ["Save Name"]);

    const submit = async () => {
      const next = nameInput.value.trim();
      if (!next) {
        showToast("Enter a model name.", "error");
        return;
      }
      saveBtn.setAttribute("disabled", "true");
      try {
        const updated = await api.renameTrainingRun(runId, next);
        upsertRunInStore(updated);
        const evalBundles = await api.listEvalBundles().catch(() => getState().evalBundles);
        setState({ evalBundles });
        onRenamed?.(updated);
        close();
        showToast(`Renamed model to ${updated.modelLabel}`, "success");
      } catch (err) {
        showToast(String(err), "error");
      } finally {
        saveBtn.removeAttribute("disabled");
      }
    };

    saveBtn.onclick = () => void submit();
    nameInput.onkeydown = (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        void submit();
      }
    };

    wrap.append(
      el("div", { class: "field" }, [
        el("div", { class: "field-label" }, ["Model / Weights Name"]),
        nameInput,
      ]),
      el("div", { class: "modal-actions" }, [cancelBtn, saveBtn])
    );
    setTimeout(() => {
      nameInput.focus();
      nameInput.select();
    }, 20);
    return wrap;
  });
}

export function confirmAndDeleteProject(projectId: string, projectName: string): void {
  openConfirmModal(
    "Delete Project",
    `Permanently delete project "${projectName}", all its dataset media, and associated training runs from disk?`,
    "Delete Project",
    async () => {
      await api.deleteProject(projectId);
      const [projects, runs, evalBundles] = await Promise.all([
        api.listProjects(),
        api.listTrainingRuns(),
        api.listEvalBundles(),
      ]);
      const state = getState();
      const deletedActive = state.activeProject?.id === projectId;
      setState({
        projects,
        runs,
        evalBundles,
        activeProject: deletedActive ? null : state.activeProject,
        activeView: deletedActive && state.activeView === "workspace" ? "dashboard" : state.activeView,
        selectedRunId: runs.some((r) => r.id === state.selectedRunId) ? state.selectedRunId : (runs[0]?.id ?? null),
      });
      showToast(`Deleted project ${projectName}`, "info");
    }
  );
}

export async function deleteTrainingRunWithFeedback(runId: string): Promise<void> {
  try {
    await api.deleteTrainingRun(runId);
    removeRunFromStore(runId);
    const evalBundles = await api.listEvalBundles().catch(() => getState().evalBundles);
    setState({ evalBundles });
    showToast(`Deleted run ${runId}`, "info");
  } catch (err) {
    showToast(String(err), "error");
  }
}

function openCreateProjectModal(): void {
  openModal("Create Project", (close) => {
    const wrap = el("div", { style: "display:flex;flex-direction:column;gap:12px" });

    const nameInput = el("input", {
      class: "input",
      type: "text",
      placeholder: "e.g. warehouse_safety_v1",
    }) as HTMLInputElement;

    let selectedModule: ProjectModule = "image";
    let selectedTask: ProjectTask = MODULE_TASKS.image.tasks[0].value;

    const moduleSelect = el("select", { class: "select" }) as HTMLSelectElement;
    for (const modKey of ["image", "audio", "text"] as ProjectModule[]) {
      moduleSelect.appendChild(el("option", { value: modKey }, [MODULE_TASKS[modKey].label]));
    }

    const taskSelect = el("select", { class: "select" }) as HTMLSelectElement;
    const hintText = el("div", { class: "dim", style: "font-size:11px" });

    const refreshTasks = () => {
      taskSelect.innerHTML = "";
      const list = MODULE_TASKS[selectedModule].tasks;
      for (const t of list) {
        taskSelect.appendChild(el("option", { value: t.value }, [t.label]));
      }
      selectedTask = list[0].value;
      hintText.textContent = list[0].hint;
    };

    moduleSelect.onchange = () => {
      selectedModule = moduleSelect.value as ProjectModule;
      refreshTasks();
    };
    taskSelect.onchange = () => {
      selectedTask = taskSelect.value as ProjectTask;
      const found = MODULE_TASKS[selectedModule].tasks.find((t) => t.value === selectedTask);
      hintText.textContent = found?.hint ?? "";
    };
    refreshTasks();

    const createBtn = el("button", { class: "btn btn-primary", type: "button" }, ["Create Project"]);
    const cancelBtn = el("button", { class: "btn", type: "button" }, ["Cancel"]);
    cancelBtn.onclick = close;

    createBtn.onclick = async () => {
      const name = nameInput.value.trim();
      if (!name) {
        showToast("Enter a project name.", "error");
        return;
      }
      createBtn.setAttribute("disabled", "true");
      try {
        const created = await api.createProject(name, selectedModule, selectedTask);
        const projects = await api.listProjects();
        setState({ projects, activeProject: created, activeView: "workspace" });
        close();
        showToast(`Created ${created.name}`, "success");
      } catch (err) {
        showToast(String(err), "error");
      } finally {
        createBtn.removeAttribute("disabled");
      }
    };

    wrap.append(
      el("div", { class: "field" }, [el("div", { class: "field-label" }, ["Project Name"]), nameInput]),
      el("div", { class: "field" }, [el("div", { class: "field-label" }, ["Module"]), moduleSelect]),
      el("div", { class: "field" }, [el("div", { class: "field-label" }, ["Task Type"]), taskSelect, hintText]),
      el("div", { class: "modal-actions" }, [cancelBtn, createBtn])
    );
    setTimeout(() => nameInput.focus(), 20);
    return wrap;
  });
}

function buildProjectCard(summary: ProjectSummary): HTMLElement {
  const pct =
    summary.itemCount > 0 ? Math.round((summary.labeledCount / summary.itemCount) * 100) : 0;

  const delBtn = el(
    "button",
    {
      class: "btn btn-sm btn-ghost",
      type: "button",
      title: "Delete Project",
    },
    [svgIcon("trash", 13)]
  );

  delBtn.onclick = (e) => {
    e.stopPropagation();
    confirmAndDeleteProject(summary.id, summary.name);
  };

  const card = el("div", { class: "project-card" }, [
    el("div", { class: "project-card-top" }, [
      el("div", {}, [
        el("div", { class: "project-card-name" }, [summary.name]),
        el("div", { style: "display:flex;gap:6px;margin-top:5px" }, [
          el("span", { class: "badge badge-accent" }, [MODULE_TASKS[summary.module].label]),
          el("span", { class: "badge" }, [summary.task]),
        ]),
      ]),
      delBtn,
    ]),
    el("div", { class: "progress-track" }, [
      el("div", { class: "progress-fill", style: `width:${pct}%` }),
    ]),
    el("div", { class: "project-card-stats" }, [
      el("span", { class: "mono" }, [
        `${summary.labeledCount}/${summary.itemCount} labeled · ${summary.classCount} classes`,
      ]),
      el("span", { class: "dim" }, [formatIsoShort(summary.updatedAt)]),
    ]),
  ]);

  card.onclick = async () => {
    try {
      const proj = await api.loadProject(summary.id);
      setState({ activeProject: proj, activeView: "workspace" });
    } catch (err) {
      showToast(String(err), "error");
    }
  };

  return card;
}

export function renderDashboard(container: HTMLElement): void {
  const state = getState();
  container.innerHTML = "";
  const root = el("div", { class: "view-scroll workspace-enter" });

  const newBtn = el("button", { class: "btn btn-primary", type: "button" }, [
    svgIcon("plus", 14),
    "New Project",
  ]);
  newBtn.onclick = openCreateProjectModal;

  const header = el("div", { class: "dashboard-header" }, [
    el("div", {}, [
      el("div", { style: "font-size:17px;font-weight:600" }, ["Projects"]),
      el("div", { class: "dim", style: "font-size:12px" }, [
        "Image ML, Audio Training, and Text / Q&A workspaces",
      ]),
    ]),
    newBtn,
  ]);
  root.appendChild(header);

  if (state.projects.length === 0) {
    const createEmptyBtn = el("button", { class: "btn btn-primary", type: "button" }, [
      svgIcon("plus", 14),
      "Create Project",
    ]);
    createEmptyBtn.onclick = openCreateProjectModal;

    const evalEmptyBtn = el("button", { class: "btn", type: "button" }, [
      svgIcon("flask", 14),
      "Test & Evaluate Model",
    ]);
    evalEmptyBtn.onclick = () => setState({ activeView: "evaluate" });

    root.appendChild(
      el("div", { class: "card empty-state" }, [
        el("img", {
          src: "/assets/damnisium.png",
          alt: "Damnisium",
          class: "welcome-emblem",
        }),
        el("div", { style: "color:var(--text-1);font-weight:600;font-size:15px" }, [
          "Damnisium v1-beta",
        ]),
        el("div", { class: "dim", style: "max-width:380px" }, [
          "Select Image, Audio, or Text / Q&A to label datasets, train on Google Colab, and evaluate ONNX models locally.",
        ]),
        el("div", { style: "display:flex;gap:8px;margin-top:4px" }, [
          createEmptyBtn,
          evalEmptyBtn,
        ]),
      ])
    );
  } else {
    const grid = el("div", { class: "dashboard-grid" });
    for (const p of state.projects) {
      grid.appendChild(buildProjectCard(p));
    }
    root.appendChild(grid);
  }

  if (state.runs.length > 0) {
    const viewAllBtn = el("button", { class: "btn btn-sm", type: "button" }, ["View All"]);
    viewAllBtn.onclick = () => setState({ activeView: "train" });
    const runsCard = el("div", { class: "card" }, [
      el("div", { class: "card-header" }, [
        el("div", { class: "card-title" }, ["Recent Training Runs"]),
        viewAllBtn,
      ]),
    ]);

    const list = el("div", { class: "runs-list" });
    for (const run of state.runs.slice(0, 6)) {
      const badgeCls =
        run.state === "completed"
          ? "badge-success"
          : run.state === "failed"
          ? "badge-danger"
          : "badge-accent";
      const renameRunBtn = el(
        "button",
        {
          class: "btn btn-sm btn-ghost",
          type: "button",
          title: "Rename trained model weights",
          style: "padding:3px 6px",
        },
        [svgIcon("edit", 12)]
      );
      renameRunBtn.onclick = (e) => {
        e.stopPropagation();
        openRenameRunModal(run.id, run.modelLabel);
      };
      const delRunBtn = el(
        "button",
        {
          class: "btn btn-sm btn-ghost",
          type: "button",
          title: "Delete training run",
          style: "padding:3px 6px",
        },
        [svgIcon("trash", 12)]
      );
      delRunBtn.onclick = (e) => {
        e.stopPropagation();
        void deleteTrainingRunWithFeedback(run.id);
      };
      const row = el("div", { class: "run-row" }, [
        el("div", {}, [
          el("div", { style: "font-weight:600;font-size:12px" }, [
            `${run.projectName} · ${run.modelLabel}`,
          ]),
          el("div", { class: "dim mono", style: "font-size:11px" }, [
            `${run.id} · ${run.message}`,
          ]),
        ]),
        el("div", { style: "display:flex;align-items:center;gap:8px" }, [
          el("span", { class: `badge ${badgeCls}` }, [run.state]),
          el("span", { class: "mono", style: "font-size:11px" }, [
            `${Math.round(run.progressPct)}%`,
          ]),
          renameRunBtn,
          delRunBtn,
        ]),
      ]);
      row.onclick = () => setState({ selectedRunId: run.id, activeView: "train" });
      list.appendChild(row);
    }
    runsCard.appendChild(list);
    root.appendChild(runsCard);
  }

  container.appendChild(root);
}
