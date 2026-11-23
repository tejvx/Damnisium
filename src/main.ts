import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/components.css";
import "./styles/image-studio.css";
import "./styles/audio-studio.css";
import "./styles/text-studio.css";
import "./styles/train-drawer.css";
import "./styles/evaluate-studio.css";
import "./styles/hub.css";

import { api } from "./api";
import { renderDashboard } from "./components/dashboard";
import { renderSettingsView } from "./components/settings-view";
import { renderSidebar } from "./components/sidebar";
import { renderTopbar } from "./components/topbar";
import { renderTrainingPanel } from "./components/training-panel";
import { renderUncensoredHub } from "./components/uncensored-hub";
import { cleanupAudioStudio, renderAudioStudio } from "./modules/audio/audio-studio";
import { cleanupEvaluateStudio, renderEvaluateStudio } from "./modules/evaluate/evaluate-studio";
import { cleanupImageStudio, renderImageStudio } from "./modules/image/image-studio";
import { renderTextStudio } from "./modules/text/text-studio";
import { getState, refreshConnectivity, setState, subscribe, upsertRunInStore } from "./state";
import type { AppView } from "./types";
import { el, showToast } from "./utils/dom";

const appHost = document.getElementById("app") as HTMLElement;
const railEl = el("aside", { class: "app-rail" });
const topbarEl = el("header", { class: "app-topbar" });
const mainEl = el("main", { class: "app-main" });
appHost.append(railEl, topbarEl, mainEl);

let lastView: AppView | null = null;
let lastProjectId: string | null = null;
let lastEvalBundleId: string | null = null;
let lastEvalBundlesSig = "";

function mountMainView(forceWorkspace = false): void {
  const state = getState();
  renderSidebar(railEl);
  renderTopbar(topbarEl);

  const view = state.activeView;
  const proj = state.activeProject;
  const evalBundlesSig = state.evalBundles.map((b) => `${b.id}:${b.label}`).join(",");

  if (
    view === "workspace" &&
    !forceWorkspace &&
    lastView === "workspace" &&
    lastProjectId === (proj?.id ?? null)
  ) {
    return;
  }

  if (
    view === "evaluate" &&
    lastView === "evaluate" &&
    lastEvalBundleId === state.selectedEvalBundleId &&
    lastEvalBundlesSig === evalBundlesSig
  ) {
    return;
  }

  if (lastView === "evaluate" && view !== "evaluate") {
    cleanupEvaluateStudio();
  }
  cleanupImageStudio();
  cleanupAudioStudio();
  lastView = view;
  lastProjectId = proj?.id ?? null;
  lastEvalBundleId = state.selectedEvalBundleId;
  lastEvalBundlesSig = evalBundlesSig;

  if (view === "dashboard") {
    renderDashboard(mainEl);
  } else if (view === "workspace") {
    if (!proj) {
      setState({ activeView: "dashboard" });
      return;
    }
    if (proj.module === "image") {
      renderImageStudio(mainEl);
    } else if (proj.module === "audio") {
      renderAudioStudio(mainEl);
    } else {
      renderTextStudio(mainEl);
    }
  } else if (view === "train") {
    renderTrainingPanel(mainEl);
  } else if (view === "hub") {
    renderUncensoredHub(mainEl);
  } else if (view === "evaluate") {
    renderEvaluateStudio(mainEl);
  } else if (view === "settings") {
    renderSettingsView(mainEl);
  }
}

function dismissSplash(): void {
  const splash = document.getElementById("splash");
  if (!splash) return;
  splash.classList.add("hidden");
  setTimeout(() => splash.remove(), 260);
}

async function bootstrap(): Promise<void> {
  subscribe(() => {
    mountMainView(false);
  });

  try {
    const [workspace, projects, runs, autolabelModels, evalBundles] = await Promise.all([
      api.getWorkspaceInfo(),
      api.listProjects(),
      api.listTrainingRuns(),
      api.listAutolabelModels(),
      api.listEvalBundles(),
    ]);
    setState({ workspace, projects, runs, autolabelModels, evalBundles });
    void refreshConnectivity();

    await api.onTrainingUpdate(async (run) => {
      upsertRunInStore(run);
      if (run.state === "completed") {
        const updatedBundles = await api.listEvalBundles().catch(() => getState().evalBundles);
        setState({
          evalBundles: updatedBundles,
          selectedEvalBundleId: run.id,
          activeView: "evaluate",
        });
        showToast(`Training completed — opened ${run.modelLabel} in Test & Evaluate`, "success");
      } else if (run.state === "failed") {
        showToast(`Training failed: ${run.message}`, "error");
      }
    });

    await api.onAutoLabelProgress((prog) => {
      if (prog.done) {
        void api.listAutolabelModels().then((list) => {
          setState({ autolabelModels: list });
        });
      }
    });
  } catch (err) {
    showToast(`Initialization failed: ${String(err)}`, "error");
  } finally {
    dismissSplash();
  }
}

void bootstrap();
