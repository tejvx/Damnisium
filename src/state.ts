import { api } from "./api";
import type {
  AppView,
  AutoLabelModelInfo,
  ConnectivityStatus,
  EvalBundleInfo,
  Project,
  ProjectSummary,
  TrainingRun,
  WorkspaceInfo,
} from "./types";

export interface AppStoreState {
  workspace: WorkspaceInfo | null;
  projects: ProjectSummary[];
  activeProject: Project | null;
  activeView: AppView;
  runs: TrainingRun[];
  selectedRunId: string | null;
  autolabelModels: AutoLabelModelInfo[];
  evalBundles: EvalBundleInfo[];
  selectedEvalBundleId: string | null;
  sidebarExpanded: boolean;
  connectivity: ConnectivityStatus;
  connectivityChecking: boolean;
  hubPendingModelRepo: string | null;
}

type Listener = (state: AppStoreState) => void;

const listeners = new Set<Listener>();

const state: AppStoreState = {
  workspace: null,
  projects: [],
  activeProject: null,
  activeView: "dashboard",
  runs: [],
  selectedRunId: null,
  autolabelModels: [],
  evalBundles: [],
  selectedEvalBundleId: null,
  sidebarExpanded: true,
  connectivity: {
    online: false,
    latencyMs: null,
    message: "Checking network...",
    checkedAt: "",
  },
  connectivityChecking: false,
  hubPendingModelRepo: null,
};

export function getState(): AppStoreState {
  return state;
}

export function setState(patch: Partial<AppStoreState>): void {
  Object.assign(state, patch);
  for (const listener of listeners) {
    listener(state);
  }
}

export function subscribe(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export async function refreshConnectivity(): Promise<ConnectivityStatus> {
  setState({ connectivityChecking: true });
  try {
    const status = await api.checkConnectivity();
    setState({ connectivity: status, connectivityChecking: false });
    return status;
  } catch {
    const fallback: ConnectivityStatus = {
      online: false,
      latencyMs: null,
      message: "Offline Mode",
      checkedAt: new Date().toISOString(),
    };
    setState({ connectivity: fallback, connectivityChecking: false });
    return fallback;
  }
}

export function upsertRunInStore(run: TrainingRun): void {
  const idx = state.runs.findIndex((r) => r.id === run.id);
  const next = [...state.runs];
  if (idx >= 0) {
    next[idx] = run;
  } else {
    next.unshift(run);
  }
  setState({ runs: next, selectedRunId: idx < 0 ? run.id : (state.selectedRunId ?? run.id) });
}

export function removeRunFromStore(runId: string): void {
  const next = state.runs.filter((r) => r.id !== runId);
  const nextSelected =
    state.selectedRunId === runId ? (next[0]?.id ?? null) : state.selectedRunId;
  setState({ runs: next, selectedRunId: nextSelected });
}
