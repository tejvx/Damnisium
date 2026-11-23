import { api } from "../../api";
import { getState, setState } from "../../state";
import type { BoundingBox, ImageItem, PolygonShape, Project, SplitType } from "../../types";
import { runAutoLabelOnImage } from "../../utils/autolabel";
import { el, showToast, svgIcon } from "../../utils/dom";
import { renderImageAugPreview } from "./aug-preview";
import {
  type CanvasTool,
  type CanvasViewState,
  ImageCanvasEngine,
  type SelectedShape,
} from "./canvas-engine";

interface Snapshot {
  boxes: BoundingBox[];
  polygons: PolygonShape[];
}

let lastImageProjectId: string | null = null;
let activeImageId: string | null = null;
let activeClassId = "";
let editingClassId: string | null = null;
let filterMode: "all" | "labeled" | "unlabeled" = "all";
let currentTool: CanvasTool = "box";
let selectedShape: SelectedShape | null = null;
let selectedAutoModelId = "yolov8n";
let autoConf = 0.35;
let autoIou = 0.45;
let engine: ImageCanvasEngine | null = null;
let savedCanvasView: { imageId: string; view: CanvasViewState } | null = null;
let undoStack: Snapshot[] = [];
let redoStack: Snapshot[] = [];
let keyHandler: ((e: KeyboardEvent) => void) | null = null;
let batchRunning = false;

function isImageLabeled(item: ImageItem): boolean {
  return item.boxes.length > 0 || item.polygons.length > 0 || item.imageLabels.length > 0;
}

function cloneSnap(item: ImageItem): Snapshot {
  return {
    boxes: item.boxes.map((b) => ({ ...b })),
    polygons: item.polygons.map((p) => ({ ...p, points: p.points.map((pt) => ({ ...pt })) })),
  };
}

export function cleanupImageStudio(): void {
  if (keyHandler) {
    window.removeEventListener("keydown", keyHandler);
    keyHandler = null;
  }
  if (engine) {
    engine.destroy();
    engine = null;
  }
}

export function renderImageStudio(container: HTMLElement): void {
  if (engine && activeImageId) {
    const view = engine.getViewState();
    if (view) {
      savedCanvasView = { imageId: activeImageId, view };
    }
  }
  cleanupImageStudio();
  const state = getState();
  const project = state.activeProject;
  const dataDir = state.workspace?.dataDir ?? "";
  if (!project) return;

  if (lastImageProjectId !== project.id) {
    lastImageProjectId = project.id;
    activeImageId = null;
    activeClassId = "";
    editingClassId = null;
    savedCanvasView = null;
    filterMode = "all";
    undoStack = [];
    redoStack = [];
    selectedShape = null;
  }

  const needsClassGate = project.imageItems.length > 0 && project.classes.length === 0;

  if (project.task === "segmentation" && currentTool === "box") {
    currentTool = "polygon";
  } else if (project.task === "detection" && currentTool === "polygon") {
    currentTool = "box";
  } else if (project.task === "classification") {
    currentTool = "select";
  }

  if (!activeClassId || !project.classes.some((c) => c.id === activeClassId)) {
    activeClassId = project.classes[0]?.id ?? "";
  }

  const filteredItems = project.imageItems.filter((item) => {
    if (filterMode === "labeled") return isImageLabeled(item);
    if (filterMode === "unlabeled") return !isImageLabeled(item);
    return true;
  });

  if (!activeImageId || !filteredItems.some((i) => i.id === activeImageId)) {
    activeImageId = filteredItems[0]?.id ?? project.imageItems[0]?.id ?? null;
    undoStack = [];
    redoStack = [];
    selectedShape = null;
  }

  const activeItem = project.imageItems.find((i) => i.id === activeImageId) ?? null;
  const curIdx = filteredItems.findIndex((i) => i.id === activeImageId);

  container.innerHTML = "";
  const layout = el("div", { class: "studio-layout" });

  const leftPanel = el("div", { class: "studio-left" });
  const importFilesBtn = el("button", { class: "btn btn-sm btn-primary", type: "button" }, [
    svgIcon("upload", 12),
    "Images",
  ]);
  const importFolderBtn = el("button", { class: "btn btn-sm", type: "button" }, [
    svgIcon("folder", 12),
    "Folder",
  ]);

  importFilesBtn.onclick = async () => {
    const sel = await api.pickFiles(
      [{ name: "Images", extensions: ["png", "jpg", "jpeg", "webp", "bmp", "gif"] }],
      true
    );
    const paths = Array.isArray(sel) ? sel : typeof sel === "string" ? [sel] : [];
    if (paths.length === 0) return;
    try {
      const updated = await api.importImages(project.id, paths);
      setState({ activeProject: updated, projects: await api.listProjects() });
      renderImageStudio(container);
      showToast(
        updated.classes.length === 0
          ? `Imported ${paths.length} image(s) — create at least one class to start labeling`
          : `Imported ${paths.length} image(s)`,
        updated.classes.length === 0 ? "info" : "success"
      );
    } catch (err) {
      showToast(String(err), "error");
    }
  };

  importFolderBtn.onclick = async () => {
    const folder = await api.pickFolder();
    if (typeof folder !== "string") return;
    try {
      const updated = await api.importImageFolder(project.id, folder);
      setState({ activeProject: updated, projects: await api.listProjects() });
      renderImageStudio(container);
      showToast(
        updated.classes.length === 0
          ? "Imported folder images — create at least one class to start labeling"
          : "Imported folder images",
        updated.classes.length === 0 ? "info" : "success"
      );
    } catch (err) {
      showToast(String(err), "error");
    }
  };

  const classInput = el("input", {
    class: "input",
    placeholder: "New class name",
  }) as HTMLInputElement;
  const addClassBtn = el("button", { class: "btn btn-sm", type: "button" }, ["Add"]);
  const submitClass = async () => {
    const name = classInput.value.trim();
    if (!name) return;
    try {
      const updated = await api.addLabelClass(project.id, name);
      activeClassId = updated.classes[updated.classes.length - 1]?.id ?? activeClassId;
      setState({ activeProject: updated, projects: await api.listProjects() });
      renderImageStudio(container);
    } catch (err) {
      showToast(String(err), "error");
    }
  };
  addClassBtn.onclick = () => void submitClass();
  classInput.onkeydown = (e) => {
    if (e.key === "Enter") void submitClass();
  };

  const classesList = el("div", { style: "display:flex;flex-direction:column;gap:3px" });
  project.classes.forEach((cls, idx) => {
    if (editingClassId === cls.id) {
      const renameIn = el("input", {
        class: "input class-rename-input",
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
          editingClassId = null;
          setState({ activeProject: updated, projects: await api.listProjects() });
          renderImageStudio(container);
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
          editingClassId = null;
          renderImageStudio(container);
        }
      };
      const row = el("div", { class: `class-row${cls.id === activeClassId ? " active" : ""}` }, [
        el("span", { class: "class-swatch", style: `background:${cls.color}` }),
        renameIn,
        saveBtn,
      ]);
      classesList.appendChild(row);
      setTimeout(() => {
        renameIn.focus();
        renameIn.select();
      }, 10);
      return;
    }

    const editCls = el(
      "button",
      { class: "btn btn-sm btn-ghost", type: "button", title: "Rename class" },
      [svgIcon("edit", 11)]
    );
    editCls.onclick = (e) => {
      e.stopPropagation();
      editingClassId = cls.id;
      renderImageStudio(container);
    };

    const delCls = el(
      "button",
      { class: "btn btn-sm btn-ghost", type: "button", title: "Delete class" },
      [svgIcon("trash", 11)]
    );
    delCls.onclick = async (e) => {
      e.stopPropagation();
      try {
        const updated = await api.deleteLabelClass(project.id, cls.id);
        setState({ activeProject: updated, projects: await api.listProjects() });
        renderImageStudio(container);
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
      editingClassId = cls.id;
      renderImageStudio(container);
    };
    const row = el(
      "div",
      { class: `class-row${cls.id === activeClassId ? " active" : ""}` },
      [
        el("span", { class: "class-swatch", style: `background:${cls.color}` }),
        nameSpan,
        idx < 9 ? el("span", { class: "dim mono", style: "font-size:10px" }, [String(idx + 1)]) : null,
        editCls,
        delCls,
      ]
    );
    row.onclick = () => {
      activeClassId = cls.id;
      engine?.setActiveClass(cls.id);
      renderImageStudio(container);
    };
    classesList.appendChild(row);
  });

  const filterBar = el("div", { class: "segmented", style: "width:100%" });
  for (const mode of ["all", "labeled", "unlabeled"] as const) {
    const b = el("button", {
      class: filterMode === mode ? "active" : "",
      style: "flex:1",
      type: "button",
    }, [mode === "all" ? `All (${project.imageItems.length})` : mode]);
    b.onclick = () => {
      filterMode = mode;
      renderImageStudio(container);
    };
    filterBar.appendChild(b);
  }

  leftPanel.append(
    el("div", { class: "panel-section" }, [
      el("div", { class: "panel-section-title" }, ["Dataset Media"]),
      el("div", { style: "display:flex;gap:6px" }, [importFilesBtn, importFolderBtn]),
      filterBar,
    ]),
    el("div", { class: "panel-section" }, [
      el("div", { class: "panel-section-title" }, ["Classes (Keys 1-9)"]),
      el("div", { style: "display:flex;gap:5px" }, [classInput, addClassBtn]),
      classesList,
    ])
  );

  const filmstrip = el("div", { class: "filmstrip" });
  for (const item of filteredItems) {
    const count =
      project.task === "classification"
        ? item.imageLabels.length
        : item.boxes.length + item.polygons.length;
    const card = el(
      "div",
      { class: `thumb-card${item.id === activeImageId ? " active" : ""}` },
      [
        el("img", {
          src: api.mediaSrc(dataDir, item.thumbRelPath),
          loading: "lazy",
          alt: item.filename,
        }),
        count > 0 ? el("span", { class: "thumb-badge" }, [String(count)]) : null,
      ]
    );
    card.onclick = () => {
      activeImageId = item.id;
      savedCanvasView = null;
      undoStack = [];
      redoStack = [];
      selectedShape = null;
      renderImageStudio(container);
    };
    filmstrip.appendChild(card);
  }
  leftPanel.appendChild(filmstrip);

  const centerPanel = el("div", { class: "studio-center" });
  const toolbar = el("div", { class: "canvas-toolbar" });
  const toolLeft = el("div", { class: "toolbar-group" });
  const toolRight = el("div", { class: "toolbar-group" });

  const makeToolBtn = (label: string, tool: CanvasTool, icon: string) => {
    const b = el(
      "button",
      { class: `tool-btn${currentTool === tool ? " active" : ""}`, type: "button", disabled: needsClassGate },
      [svgIcon(icon, 13), label]
    );
    b.onclick = () => {
      currentTool = tool;
      engine?.setTool(tool);
      renderImageStudio(container);
    };
    return b;
  };

  toolLeft.appendChild(makeToolBtn("Select (V)", "select", "pointer"));
  if (project.task === "detection") {
    toolLeft.appendChild(makeToolBtn("Box (B)", "box", "box"));
  }
  if (project.task === "segmentation") {
    toolLeft.appendChild(makeToolBtn("Polygon (P)", "polygon", "polygon"));
  }

  const fitBtn = el("button", { class: "tool-btn", type: "button", disabled: needsClassGate }, ["Fit (F)"]);
  fitBtn.onclick = () => {
    savedCanvasView = null;
    engine?.fitView();
  };
  const undoBtn = el("button", { class: "tool-btn", type: "button", disabled: needsClassGate || undoStack.length === 0 }, ["Undo"]);
  const redoBtn = el("button", { class: "tool-btn", type: "button", disabled: needsClassGate || redoStack.length === 0 }, ["Redo"]);
  const delShapeBtn = el(
    "button",
    { class: "tool-btn", type: "button", disabled: needsClassGate || !selectedShape },
    [svgIcon("trash", 12), "Delete"]
  );
  const posCounter =
    filteredItems.length > 0 && curIdx >= 0
      ? el("span", { class: "mono dim", style: "font-size:11px;padding:0 6px" }, [
        `${curIdx + 1} / ${filteredItems.length}`,
      ])
      : null;
  const saveNextBtn = el(
    "button",
    {
      class: "btn btn-sm btn-primary",
      type: "button",
      disabled: !activeItem || needsClassGate,
      title: "Save current labels and advance to next image (Ctrl+S / Enter)",
    },
    [
      svgIcon("check", 12),
      curIdx >= 0 && curIdx + 1 < filteredItems.length ? "Save & Next" : "Save Label",
    ]
  );

  toolRight.append(undoBtn, redoBtn, delShapeBtn, fitBtn);
  if (posCounter) {
    toolRight.appendChild(posCounter);
  }
  toolRight.appendChild(saveNextBtn);
  toolbar.append(toolLeft, toolRight);

  const canvasStage = el("div", { class: "canvas-stage" });
  const canvasEl = el("canvas") as HTMLCanvasElement;
  canvasStage.appendChild(canvasEl);

  const persistActiveItem = async (
    nextBoxes: BoundingBox[],
    nextPolys: PolygonShape[],
    nextLabels: string[],
    nextSplit: SplitType,
    pushUndo = true,
    advanceDelta = 0
  ) => {
    if (!activeItem) return;
    if (pushUndo) {
      undoStack.push(cloneSnap(activeItem));
      if (undoStack.length > 40) undoStack.shift();
      redoStack = [];
    }
    try {
      const updated = await api.saveImageAnnotations(
        project.id,
        activeItem.id,
        nextSplit,
        nextBoxes,
        nextPolys,
        nextLabels
      );
      if (advanceDelta !== 0) {
        const nextIdx = curIdx + advanceDelta;
        if (nextIdx >= 0 && nextIdx < filteredItems.length) {
          activeImageId = filteredItems[nextIdx].id;
          savedCanvasView = null;
          undoStack = [];
          redoStack = [];
          selectedShape = null;
        }
      }
      setState({ activeProject: updated, projects: await api.listProjects() });
      renderImageStudio(container);
    } catch (err) {
      showToast(String(err), "error");
    }
  };

  const saveAndAdvance = async (delta: number, showSavedToast = false) => {
    if (!activeItem || needsClassGate) return;
    if (engine?.hasReadyPolygon()) {
      engine.finishPolygonIfReady();
      return;
    }
    const targetIdx = curIdx + delta;
    const willAdvance = targetIdx >= 0 && targetIdx < filteredItems.length;
    await persistActiveItem(
      activeItem.boxes,
      activeItem.polygons,
      activeItem.imageLabels,
      activeItem.split,
      false,
      willAdvance ? delta : 0
    );
    if (showSavedToast) {
      showToast(
        delta > 0
          ? willAdvance
            ? "Saved & next image"
            : "Saved — last image reached"
          : "Saved",
        "success"
      );
    }
  };

  saveNextBtn.onclick = () => void saveAndAdvance(1, true);

  if (needsClassGate) {
    const gateIn = el("input", {
      class: "input",
      placeholder: "e.g. person, defect, product",
    }) as HTMLInputElement;
    const gateBtn = el("button", { class: "btn btn-primary", type: "button" }, [
      svgIcon("plus", 13),
      "Create Class & Start Labeling",
    ]);
    const submitGate = async () => {
      const name = gateIn.value.trim();
      if (!name) {
        showToast("Enter a class name to continue.", "error");
        gateIn.focus();
        return;
      }
      gateBtn.setAttribute("disabled", "true");
      try {
        const updated = await api.addLabelClass(project.id, name);
        activeClassId = updated.classes[updated.classes.length - 1]?.id ?? "";
        setState({ activeProject: updated, projects: await api.listProjects() });
        renderImageStudio(container);
        showToast(`Created class "${name}"`, "success");
      } catch (err) {
        showToast(String(err), "error");
        gateBtn.removeAttribute("disabled");
      }
    };
    gateBtn.onclick = () => void submitGate();
    gateIn.onkeydown = (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        void submitGate();
      }
    };

    const gateOverlay = el("div", { class: "class-gate-overlay" }, [
      el("div", { class: "class-gate-card" }, [
        el("div", { style: "font-size:15px;font-weight:600;color:var(--text-1)" }, [
          "Create a Class to Continue",
        ]),
        el("div", { class: "dim", style: "font-size:12px;line-height:1.45" }, [
          `${project.imageItems.length} image(s) uploaded. Add at least one class label before drawing or tagging images.`,
        ]),
        gateIn,
        gateBtn,
      ]),
    ]);
    canvasStage.appendChild(gateOverlay);
    setTimeout(() => gateIn.focus(), 30);
  } else if (filteredItems.length > 1) {
    const prevStageBtn = el(
      "button",
      {
        class: "stage-nav-btn stage-nav-prev",
        type: "button",
        title: "Previous image (←)",
        disabled: curIdx <= 0,
      },
      [svgIcon("chevronLeft", 20)]
    );
    prevStageBtn.onclick = () => void saveAndAdvance(-1, false);

    const nextStageBtn = el(
      "button",
      {
        class: "stage-nav-btn stage-nav-next",
        type: "button",
        title: "Next image (→)",
        disabled: curIdx < 0 || curIdx >= filteredItems.length - 1,
      },
      [svgIcon("chevronRight", 20)]
    );
    nextStageBtn.onclick = () => void saveAndAdvance(1, false);

    canvasStage.append(prevStageBtn, nextStageBtn);
  }

  centerPanel.append(toolbar, canvasStage);

  if (project.task === "classification" && activeItem && !needsClassGate) {
    const clsBar = el("div", { class: "cls-bar" });
    for (const cls of project.classes) {
      const isTagged = activeItem.imageLabels.includes(cls.id);
      const chip = el(
        "button",
        { class: `cls-chip${isTagged ? " active" : ""}`, type: "button" },
        [
          el("span", { class: "class-swatch", style: `background:${cls.color}` }),
          cls.name,
        ]
      );
      chip.onclick = () => {
        const next = isTagged
          ? activeItem.imageLabels.filter((id) => id !== cls.id)
          : [cls.id];
        const advance = isTagged ? 0 : 1;
        void persistActiveItem(activeItem.boxes, activeItem.polygons, next, activeItem.split, false, advance);
      };
      clsBar.appendChild(chip);
    }
    centerPanel.appendChild(clsBar);
  }

  const doUndo = async () => {
    if (!activeItem || undoStack.length === 0) return;
    const prev = undoStack.pop()!;
    redoStack.push(cloneSnap(activeItem));
    try {
      const updated = await api.saveImageAnnotations(
        project.id,
        activeItem.id,
        activeItem.split,
        prev.boxes,
        prev.polygons,
        activeItem.imageLabels
      );
      setState({ activeProject: updated });
      renderImageStudio(container);
    } catch (err) {
      undoStack.push(prev);
      redoStack.pop();
      showToast(String(err), "error");
    }
  };

  const doRedo = async () => {
    if (!activeItem || redoStack.length === 0) return;
    const next = redoStack.pop()!;
    undoStack.push(cloneSnap(activeItem));
    try {
      const updated = await api.saveImageAnnotations(
        project.id,
        activeItem.id,
        activeItem.split,
        next.boxes,
        next.polygons,
        activeItem.imageLabels
      );
      setState({ activeProject: updated });
      renderImageStudio(container);
    } catch (err) {
      redoStack.push(next);
      undoStack.pop();
      showToast(String(err), "error");
    }
  };

  const deleteSelected = () => {
    if (!activeItem || !selectedShape) return;
    const nextBoxes =
      selectedShape.kind === "box"
        ? activeItem.boxes.filter((b) => b.id !== selectedShape!.id)
        : activeItem.boxes;
    const nextPolys =
      selectedShape.kind === "polygon"
        ? activeItem.polygons.filter((p) => p.id !== selectedShape!.id)
        : activeItem.polygons;
    selectedShape = null;
    void persistActiveItem(nextBoxes, nextPolys, activeItem.imageLabels, activeItem.split, true);
  };

  undoBtn.onclick = () => void doUndo();
  redoBtn.onclick = () => void doRedo();
  delShapeBtn.onclick = deleteSelected;

  const rightPanel = el("div", { class: "studio-right" });

  if (activeItem) {
    const splitSel = el("select", { class: "select" }) as HTMLSelectElement;
    for (const sp of ["train", "val", "test"] as SplitType[]) {
      const o = el("option", { value: sp }, [sp]) as HTMLOptionElement;
      if (activeItem.split === sp) o.selected = true;
      splitSel.appendChild(o);
    }
    splitSel.onchange = () => {
      void persistActiveItem(
        activeItem.boxes,
        activeItem.polygons,
        activeItem.imageLabels,
        splitSel.value as SplitType,
        false
      );
    };

    const delImgBtn = el("button", { class: "btn btn-sm btn-danger", type: "button" }, [
      svgIcon("trash", 12),
      "Remove Image",
    ]);
    delImgBtn.onclick = async () => {
      try {
        const updated = await api.deleteImageItem(project.id, activeItem.id);
        activeImageId = null;
        savedCanvasView = null;
        setState({ activeProject: updated, projects: await api.listProjects() });
        renderImageStudio(container);
      } catch (err) {
        showToast(String(err), "error");
      }
    };

    const annoList = el("div", { style: "display:flex;flex-direction:column;gap:4px" });
    for (const b of activeItem.boxes) {
      const cls = project.classes.find((c) => c.id === b.classId);
      const rmBtn = el("button", { class: "btn btn-sm btn-ghost", type: "button" }, [
        svgIcon("trash", 11),
      ]);
      rmBtn.onclick = (e) => {
        e.stopPropagation();
        void persistActiveItem(
          activeItem.boxes.filter((x) => x.id !== b.id),
          activeItem.polygons,
          activeItem.imageLabels,
          activeItem.split,
          true
        );
      };
      const row = el(
        "div",
        {
          class: `anno-item${selectedShape?.id === b.id ? " active" : ""}`,
        },
        [
          el("span", { class: "class-swatch", style: `background:${cls?.color ?? "#38bdf8"}` }),
          el("span", { style: "flex:1" }, [cls?.name ?? "Unknown"]),
          b.confidence ? el("span", { class: "mono dim" }, [`${Math.round(b.confidence * 100)}%`]) : null,
          rmBtn,
        ]
      );
      row.onclick = () => {
        selectedShape = { kind: "box", id: b.id };
        currentTool = "select";
        engine?.setTool("select");
        engine?.setData(project.classes, activeItem.boxes, activeItem.polygons, selectedShape);
      };
      annoList.appendChild(row);
    }
    for (const poly of activeItem.polygons) {
      const cls = project.classes.find((c) => c.id === poly.classId);
      const rmBtn = el("button", { class: "btn btn-sm btn-ghost", type: "button" }, [
        svgIcon("trash", 11),
      ]);
      rmBtn.onclick = (e) => {
        e.stopPropagation();
        void persistActiveItem(
          activeItem.boxes,
          activeItem.polygons.filter((x) => x.id !== poly.id),
          activeItem.imageLabels,
          activeItem.split,
          true
        );
      };
      const row = el(
        "div",
        { class: `anno-item${selectedShape?.id === poly.id ? " active" : ""}` },
        [
          el("span", { class: "class-swatch", style: `background:${cls?.color ?? "#a855f7"}` }),
          el("span", { style: "flex:1" }, [`${cls?.name ?? "Poly"} (${poly.points.length} pts)`]),
          rmBtn,
        ]
      );
      row.onclick = () => {
        selectedShape = { kind: "polygon", id: poly.id };
        currentTool = "select";
        engine?.setTool("select");
        engine?.setData(project.classes, activeItem.boxes, activeItem.polygons, selectedShape);
        delShapeBtn.disabled = false;
      };
      annoList.appendChild(row);
    }

    rightPanel.append(
      el("div", { class: "panel-section" }, [
        el("div", { class: "panel-section-title" }, ["Item Inspector", delImgBtn]),
        el("div", { class: "mono", style: "font-size:11px;word-break:break-all" }, [
          `${activeItem.filename} (${activeItem.width}×${activeItem.height})`,
        ]),
        el("div", { class: "field" }, [el("div", { class: "field-label" }, ["Split"]), splitSel]),
      ]),
      el("div", { class: "panel-section" }, [
        el("div", { class: "panel-section-title" }, [
          `Annotations (${activeItem.boxes.length + activeItem.polygons.length})`,
        ]),
        annoList.children.length > 0
          ? annoList
          : el("div", { class: "dim", style: "font-size:11px" }, ["No annotations on this image"]),
      ])
    );
  }

  if (project.task === "detection") {
    const models = state.autolabelModels;
    const autoSel = el("select", { class: "select" }) as HTMLSelectElement;
    for (const m of models) {
      const label = m.downloaded ? `${m.name} (Ready)` : `${m.name} (${m.approxMb} MB)`;
      const opt = el("option", { value: m.id }, [label]) as HTMLOptionElement;
      if (m.id === selectedAutoModelId) opt.selected = true;
      autoSel.appendChild(opt);
    }
    autoSel.onchange = () => {
      selectedAutoModelId = autoSel.value;
      renderImageStudio(container);
    };

    const selectedModel = models.find((m) => m.id === selectedAutoModelId) ?? models[0];
    const dlBtn = el("button", { class: "btn btn-sm btn-primary", type: "button" }, [
      svgIcon("download", 12),
      selectedModel?.downloaded ? "Model Ready" : `Download (${selectedModel?.approxMb ?? 12} MB)`,
    ]);
    if (selectedModel?.downloaded) dlBtn.setAttribute("disabled", "true");
    dlBtn.onclick = async () => {
      if (!selectedModel) return;
      dlBtn.setAttribute("disabled", "true");
      dlBtn.textContent = "Downloading...";
      try {
        await api.downloadAutolabelModel(selectedModel.id);
        setState({ autolabelModels: await api.listAutolabelModels() });
        renderImageStudio(container);
        showToast(`${selectedModel.name} downloaded`, "success");
      } catch (err) {
        showToast(String(err), "error");
        dlBtn.removeAttribute("disabled");
      }
    };

    const importOnnxBtn = el("button", { class: "btn btn-sm", type: "button" }, ["Import .onnx"]);
    importOnnxBtn.onclick = async () => {
      const picked = await api.pickFiles([{ name: "ONNX Model", extensions: ["onnx"] }], false);
      if (typeof picked !== "string") return;
      try {
        const added = await api.importCustomAutolabelOnnx(picked);
        selectedAutoModelId = added.id;
        setState({ autolabelModels: await api.listAutolabelModels() });
        renderImageStudio(container);
        showToast(`Imported ${added.name}`, "success");
      } catch (err) {
        showToast(String(err), "error");
      }
    };

    const confVal = el("span", { class: "mono" }, [autoConf.toFixed(2)]);
    const confSlider = el("input", {
      type: "range",
      min: "0.1",
      max: "0.9",
      step: "0.05",
      value: String(autoConf),
    }) as HTMLInputElement;
    confSlider.oninput = () => {
      autoConf = Number(confSlider.value);
      confVal.textContent = autoConf.toFixed(2);
    };

    const iouVal = el("span", { class: "mono" }, [autoIou.toFixed(2)]);
    const iouSlider = el("input", {
      type: "range",
      min: "0.1",
      max: "0.9",
      step: "0.05",
      value: String(autoIou),
    }) as HTMLInputElement;
    iouSlider.oninput = () => {
      autoIou = Number(iouSlider.value);
      iouVal.textContent = autoIou.toFixed(2);
    };

    const runAutoForItem = async (target: ImageItem): Promise<Project> => {
      let curProj = getState().activeProject!;
      const modelInfo = getState().autolabelModels.find((m) => m.id === selectedAutoModelId);
      if (!modelInfo || !modelInfo.downloaded) {
        throw new Error("Download or import an ONNX auto-label model first.");
      }
      const res = await runAutoLabelOnImage(
        modelInfo.localPath,
        api.mediaSrc(dataDir, target.relPath),
        modelInfo.inputSize,
        autoConf,
        autoIou,
        curProj.classes,
        modelInfo.classes ?? []
      );
      for (const missingName of res.missingClassNames) {
        curProj = await api.addLabelClass(curProj.id, missingName);
      }
      const classByName = new Map(
        curProj.classes.map((c) => [c.name.trim().toLowerCase(), c.id])
      );
      const resolvedBoxes = res.mappedBoxes.map((b) => {
        if (b.classId.startsWith("pending:")) {
          const rawName = b.classId.slice("pending:".length).toLowerCase();
          return { ...b, classId: classByName.get(rawName) ?? curProj.classes[0]?.id ?? b.classId };
        }
        return b;
      });
      return api.saveImageAnnotations(
        curProj.id,
        target.id,
        target.split,
        [...target.boxes, ...resolvedBoxes],
        target.polygons,
        target.imageLabels
      );
    };

    const runCurrentBtn = el(
      "button",
      { class: "btn btn-sm", type: "button", disabled: !activeItem || !selectedModel?.downloaded },
      [svgIcon("wand", 12), "Run Current"]
    );
    runCurrentBtn.onclick = async () => {
      if (!activeItem) return;
      runCurrentBtn.setAttribute("disabled", "true");
      try {
        const updated = await runAutoForItem(activeItem);
        setState({ activeProject: updated, projects: await api.listProjects() });
        renderImageStudio(container);
        showToast("Auto-labeled current image", "success");
      } catch (err) {
        showToast(String(err), "error");
      } finally {
        runCurrentBtn.removeAttribute("disabled");
      }
    };

    const runBatchBtn = el(
      "button",
      { class: "btn btn-sm", type: "button", disabled: !selectedModel?.downloaded },
      ["Batch Unlabeled"]
    );
    runBatchBtn.onclick = async () => {
      if (batchRunning) return;
      batchRunning = true;
      runBatchBtn.setAttribute("disabled", "true");
      try {
        const targets = getState().activeProject!.imageItems.filter((i) => i.boxes.length === 0);
        if (targets.length === 0) {
          showToast("All images already have bounding boxes.", "info");
          return;
        }
        let done = 0;
        for (const item of targets) {
          runBatchBtn.textContent = `${done}/${targets.length}`;
          const updated = await runAutoForItem(item);
          setState({ activeProject: updated });
          done++;
        }
        setState({ projects: await api.listProjects() });
        renderImageStudio(container);
        showToast(`Auto-labeled ${done} image(s)`, "success");
      } catch (err) {
        showToast(String(err), "error");
      } finally {
        batchRunning = false;
        runBatchBtn.textContent = "Batch Unlabeled";
        if (selectedModel?.downloaded) {
          runBatchBtn.removeAttribute("disabled");
        }
      }
    };

    rightPanel.appendChild(
      el("div", { class: "panel-section" }, [
        el("div", { class: "panel-section-title" }, ["Auto-Label Assist (ONNX)"]),
        autoSel,
        el("div", { style: "display:flex;gap:6px" }, [dlBtn, importOnnxBtn]),
        el("div", { class: "slider-row" }, [
          el("div", { class: "slider-head" }, [el("span", {}, ["Confidence"]), confVal]),
          confSlider,
        ]),
        el("div", { class: "slider-row" }, [
          el("div", { class: "slider-head" }, [el("span", {}, ["IoU NMS"]), iouVal]),
          iouSlider,
        ]),
        el("div", { style: "display:flex;gap:6px" }, [runCurrentBtn, runBatchBtn]),
      ])
    );
  }

  const aug = { ...project.imageAug };
  const augCanvas = el("canvas", {
    class: "aug-preview-canvas",
    width: "260",
    height: "140",
  }) as HTMLCanvasElement;
  const activeSrc = activeItem ? api.mediaSrc(dataDir, activeItem.relPath) : null;
  const redrawAug = () => renderImageAugPreview(augCanvas, activeSrc, aug);

  const flipHChk = el("input", { type: "checkbox", checked: aug.flipHorizontal }) as HTMLInputElement;
  const flipVChk = el("input", { type: "checkbox", checked: aug.flipVertical }) as HTMLInputElement;
  const mosaicChk = el("input", { type: "checkbox", checked: aug.mosaic }) as HTMLInputElement;

  const rotVal = el("span", { class: "mono" }, [`${aug.rotateDeg}°`]);
  const rotSlider = el("input", {
    type: "range",
    min: "0",
    max: "45",
    step: "1",
    value: String(aug.rotateDeg),
  }) as HTMLInputElement;
  const brightVal = el("span", { class: "mono" }, [`${aug.brightnessPct}%`]);
  const brightSlider = el("input", {
    type: "range",
    min: "-40",
    max: "40",
    step: "2",
    value: String(aug.brightnessPct),
  }) as HTMLInputElement;
  const contrastVal = el("span", { class: "mono" }, [`${aug.contrastPct}%`]);
  const contrastSlider = el("input", {
    type: "range",
    min: "-40",
    max: "40",
    step: "2",
    value: String(aug.contrastPct),
  }) as HTMLInputElement;

  flipHChk.onchange = () => { aug.flipHorizontal = flipHChk.checked; redrawAug(); };
  flipVChk.onchange = () => { aug.flipVertical = flipVChk.checked; redrawAug(); };
  mosaicChk.onchange = () => { aug.mosaic = mosaicChk.checked; };
  rotSlider.oninput = () => {
    aug.rotateDeg = Number(rotSlider.value);
    rotVal.textContent = `${aug.rotateDeg}°`;
    redrawAug();
  };
  brightSlider.oninput = () => {
    aug.brightnessPct = Number(brightSlider.value);
    brightVal.textContent = `${aug.brightnessPct}%`;
    redrawAug();
  };
  contrastSlider.oninput = () => {
    aug.contrastPct = Number(contrastSlider.value);
    contrastVal.textContent = `${aug.contrastPct}%`;
    redrawAug();
  };

  const saveAugBtn = el("button", { class: "btn btn-sm", type: "button" }, ["Save Augmentations"]);
  saveAugBtn.onclick = async () => {
    try {
      const updated = await api.updateProjectSettings(
        project.id,
        aug,
        project.audioAug,
        project.textAug,
        project.splitRatio,
        false
      );
      setState({ activeProject: updated });
      showToast("Saved image augmentation config", "success");
    } catch (err) {
      showToast(String(err), "error");
    }
  };

  rightPanel.appendChild(
    el("div", { class: "panel-section" }, [
      el("div", { class: "panel-section-title" }, ["Augmentations"]),
      augCanvas,
      el("label", { class: "check-row" }, [el("span", {}, ["Flip Horizontal"]), flipHChk]),
      el("label", { class: "check-row" }, [el("span", {}, ["Flip Vertical"]), flipVChk]),
      el("label", { class: "check-row" }, [el("span", {}, ["YOLO Mosaic"]), mosaicChk]),
      el("div", { class: "slider-row" }, [
        el("div", { class: "slider-head" }, [el("span", {}, ["Rotation"]), rotVal]),
        rotSlider,
      ]),
      el("div", { class: "slider-row" }, [
        el("div", { class: "slider-head" }, [el("span", {}, ["Brightness"]), brightVal]),
        brightSlider,
      ]),
      el("div", { class: "slider-row" }, [
        el("div", { class: "slider-head" }, [el("span", {}, ["Contrast"]), contrastVal]),
        contrastSlider,
      ]),
      saveAugBtn,
    ])
  );

  layout.append(leftPanel, centerPanel, rightPanel);
  container.appendChild(layout);

  engine = new ImageCanvasEngine(
    canvasEl,
    (boxes, polys) => {
      if (activeItem) {
        void persistActiveItem(boxes, polys, activeItem.imageLabels, activeItem.split, true);
      }
    },
    (sel) => {
      selectedShape = sel;
      delShapeBtn.disabled = !sel;
    }
  );
  engine.setTool(currentTool);
  engine.setActiveClass(activeClassId);
  if (activeItem) {
    const reuseView =
      savedCanvasView && savedCanvasView.imageId === activeItem.id ? savedCanvasView.view : null;
    engine.setData(project.classes, activeItem.boxes, activeItem.polygons, selectedShape);
    engine.loadImage(api.mediaSrc(dataDir, activeItem.relPath), reuseView);
  } else {
    engine.loadImage(null);
  }

  redrawAug();

  keyHandler = (e: KeyboardEvent) => {
    const tag = (e.target as HTMLElement | null)?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (needsClassGate) return;

    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
      e.preventDefault();
      void saveAndAdvance(1, true);
      return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
      e.preventDefault();
      void doUndo();
      return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") {
      e.preventDefault();
      void doRedo();
      return;
    }
    if (e.key === "Delete" || e.key === "Backspace") {
      deleteSelected();
      return;
    }
    if (e.key.toLowerCase() === "v") {
      currentTool = "select";
      engine?.setTool("select");
      renderImageStudio(container);
    } else if (e.key.toLowerCase() === "b" && project.task === "detection") {
      currentTool = "box";
      engine?.setTool("box");
      renderImageStudio(container);
    } else if (e.key.toLowerCase() === "p" && project.task === "segmentation") {
      currentTool = "polygon";
      engine?.setTool("polygon");
      renderImageStudio(container);
    } else if (e.key.toLowerCase() === "f") {
      savedCanvasView = null;
      engine?.fitView();
    } else if (e.key === "Enter") {
      if (engine?.hasReadyPolygon()) {
        engine.finishPolygonIfReady();
      } else {
        e.preventDefault();
        void saveAndAdvance(1, true);
      }
    } else if (e.key === "Escape") {
      engine?.cancelDraft();
    } else if (e.key >= "1" && e.key <= "9") {
      const idx = Number(e.key) - 1;
      const cls = project.classes[idx];
      if (cls) {
        activeClassId = cls.id;
        engine?.setActiveClass(activeClassId);
        if (project.task === "classification" && activeItem) {
          void persistActiveItem(
            activeItem.boxes,
            activeItem.polygons,
            [cls.id],
            activeItem.split,
            false,
            1
          );
        } else {
          renderImageStudio(container);
        }
      }
    } else if (e.key === "ArrowRight" || (e.key.toLowerCase() === "d" && !e.ctrlKey && !e.metaKey)) {
      e.preventDefault();
      void saveAndAdvance(1, false);
    } else if (e.key === "ArrowLeft" || (e.key.toLowerCase() === "a" && !e.ctrlKey && !e.metaKey)) {
      e.preventDefault();
      void saveAndAdvance(-1, false);
    }
  };
  window.addEventListener("keydown", keyHandler);

  const activeThumb = filmstrip.querySelector(".thumb-card.active");
  if (activeThumb) {
    activeThumb.scrollIntoView({ block: "nearest" });
  }
}
