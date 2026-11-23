import { api } from "../api";
import { APP_NAME, APP_VERSION } from "../constants";
import { getState, refreshConnectivity, setState } from "../state";
import { disposeAutoLabelSession } from "../utils/autolabel";
import { el, showToast, svgIcon } from "../utils/dom";
import { openGoogleSignInModal } from "./google-signin-modal";

export function renderSettingsView(container: HTMLElement): void {
  const state = getState();
  const ws = state.workspace;
  const conn = state.connectivity;
  container.innerHTML = "";
  const root = el("div", { class: "view-scroll" });
  const wrap = el("div", { class: "settings-wrap" });

  wrap.appendChild(
    el("div", {}, [
      el("div", { style: "font-size:17px;font-weight:600" }, ["Settings"]),
      el("div", { class: "dim", style: "font-size:12px" }, [
        "Network status, Hugging Face Read Token, Google OAuth for Colab Compute, and portable local storage",
      ]),
    ])
  );

  const retryConnBtn = el(
    "button",
    { class: "btn btn-sm", type: "button", disabled: state.connectivityChecking },
    [svgIcon("refresh", 12), state.connectivityChecking ? "Checking..." : "Retry Connection"]
  );
  retryConnBtn.onclick = async () => {
    const res = await refreshConnectivity();
    showToast(
      res.online ? `Connected (${res.latencyMs ?? 0} ms)` : "Offline Mode active",
      res.online ? "success" : "info"
    );
  };

  const netCard = el("div", { class: "card", style: "display:flex;flex-direction:column;gap:10px" }, [
    el("div", { class: "card-header", style: "margin-bottom:0" }, [
      el("div", { class: "card-title" }, ["Connectivity & Offline Mode"]),
      el("span", { class: `badge ${conn.online ? "badge-success" : "badge-warn"}` }, [
        conn.online
          ? conn.latencyMs !== null
            ? `Online · ${conn.latencyMs}ms`
            : "Online"
          : "Offline Mode",
      ]),
    ]),
    el("div", { class: "dim", style: "font-size:12px" }, [
      "Dataset creation, labeling, ZIP export, and local ONNX / GGUF Test & Evaluate always work offline without signing in.",
    ]),
    el("div", { style: "display:flex;align-items:center;justify-content:space-between;gap:8px" }, [
      el("span", { class: "mono dim", style: "font-size:11px" }, [conn.message]),
      retryConnBtn,
    ]),
  ]);

  const clientIdInput = el("input", {
    class: "input mono",
    type: "text",
    value: ws?.config.oauthClientId ?? "",
    placeholder: "xxxxxxxxxxxx-xxxxxxxxxxxxxxxx.apps.googleusercontent.com",
  }) as HTMLInputElement;

  const clientSecretInput = el("input", {
    class: "input mono",
    type: "password",
    value: ws?.config.oauthClientSecret ?? "",
    placeholder: "Optional for Desktop OAuth client (recommended if issued)",
  }) as HTMLInputElement;

  const gcpProjectInput = el("input", {
    class: "input mono",
    type: "text",
    value: ws?.config.gcpProjectId ?? "",
    placeholder: "Optional GCP quota project number",
  }) as HTMLInputElement;

  const driveBackupChk = el("input", {
    type: "checkbox",
    checked: Boolean(ws?.config.enableDriveBackup),
  }) as HTMLInputElement;

  const hfTokenInput = el("input", {
    class: "input mono",
    type: "password",
    value: ws?.config.hfToken ?? "",
    placeholder: "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  }) as HTMLInputElement;

  const persistConfig = async (toastMsg: string) => {
    const prevDrive = Boolean(getState().workspace?.config.enableDriveBackup);
    await api.saveAppConfig(
      clientIdInput.value,
      clientSecretInput.value,
      hfTokenInput.value,
      gcpProjectInput.value,
      driveBackupChk.checked
    );
    const updated = await api.getWorkspaceInfo();
    setState({ workspace: updated });
    if (prevDrive !== driveBackupChk.checked && updated.auth.connected) {
      showToast("Saved — click Switch / Reconnect Google Account to update OAuth scopes", "info");
    } else {
      showToast(toastMsg, "success");
    }
  };

  const saveHfBtn = el("button", { class: "btn", type: "button" }, ["Save HF Token"]);
  saveHfBtn.onclick = async () => {
    try {
      await persistConfig("Saved Hugging Face Read Token locally");
    } catch (err) {
      showToast(String(err), "error");
    }
  };

  const hfCard = el("div", { class: "card", style: "display:flex;flex-direction:column;gap:10px" }, [
    el("div", { class: "card-header", style: "margin-bottom:0" }, [
      el("div", { class: "card-title" }, ["Hugging Face Read Token"]),
      el("span", { class: `badge ${ws?.config.hfToken ? "badge-success" : "badge"}` }, [
        ws?.config.hfToken ? "Configured" : "Optional",
      ]),
    ]),
    el("div", { class: "dim", style: "font-size:12px" }, [
      "Stored locally in ./damnisium-data/config.json to avoid API rate limits and access gated Hugging Face repositories.",
    ]),
    el("div", { style: "display:flex;gap:8px;align-items:center" }, [hfTokenInput, saveHfBtn]),
  ]);

  const saveCredsBtn = el("button", { class: "btn", type: "button" }, ["Save Colab & OAuth Settings"]);
  saveCredsBtn.onclick = async () => {
    try {
      await persistConfig("Saved Colab project & OAuth settings");
    } catch (err) {
      showToast(String(err), "error");
    }
  };

  driveBackupChk.onchange = async () => {
    try {
      await persistConfig(
        driveBackupChk.checked
          ? "Enabled optional Drive checkpoint backup"
          : "Disabled Drive checkpoint backup"
      );
    } catch (err) {
      showToast(String(err), "error");
    }
  };

  const signInModalBtn = el("button", { class: "btn btn-primary", type: "button" }, [
    svgIcon("external", 13),
    ws?.auth.connected
      ? ws.auth.colabAuthorized
        ? "Switch / Reconnect Google Account"
        : "Re-sign in for Colab Scope"
      : "Sign in with Google",
  ]);
  signInModalBtn.onclick = () => openGoogleSignInModal(() => renderSettingsView(container));

  const disconnectBtn = el("button", { class: "btn btn-danger", type: "button" }, [
    svgIcon("logout", 13),
    "Logout of Colab",
  ]);
  disconnectBtn.onclick = async () => {
    try {
      await api.disconnectGoogleAccount();
      const updated = await api.getWorkspaceInfo();
      setState({ workspace: updated });
      showToast("Google account disconnected", "info");
    } catch (err) {
      showToast(String(err), "error");
    }
  };

  const authBadge = ws?.auth.connected
    ? ws.auth.colabAuthorized
      ? el("span", { class: "badge badge-success" }, [
          `Colab Ready: ${ws.auth.user?.email || "Google Account"}`,
        ])
      : el("span", { class: "badge badge-warn" }, [
          `Re-sign in needed: ${ws.auth.user?.email || "Missing Colab Scope"}`,
        ])
    : el("span", { class: "badge badge-warn" }, ["Not Linked"]);

  const advancedOverride = el("details", { class: "advanced-details" }, [
    el("summary", {}, ["Advanced Colab & OAuth Override"]),
    el("div", { style: "display:flex;flex-direction:column;gap:10px;margin-top:10px" }, [
      el("div", { class: "dim", style: "font-size:11px" }, [
        "Uses Damnisium built-in desktop OAuth credentials by default. Override below if using your own GCP project.",
      ]),
      el("div", { class: "grid-2" }, [
        el("div", { class: "field" }, [
          el("div", { class: "field-label" }, ["OAuth Client ID"]),
          clientIdInput,
        ]),
        el("div", { class: "field" }, [
          el("div", { class: "field-label" }, ["OAuth Client Secret"]),
          clientSecretInput,
        ]),
      ]),
      el("div", { class: "field" }, [
        el("div", { class: "field-label" }, ["GCP Quota Project ID (x-goog-user-project)"]),
        gcpProjectInput,
      ]),
      el("div", { style: "display:flex;gap:8px" }, [saveCredsBtn]),
    ]),
  ]);

  const oauthCard = el("div", { class: "card", style: "display:flex;flex-direction:column;gap:12px" }, [
    el("div", { class: "card-header", style: "margin-bottom:0" }, [
      el("div", { class: "card-title" }, ["Google Account & Headless Colab Compute"]),
      authBadge,
    ]),
    el("div", { class: "dim", style: "font-size:12px" }, [
      ws?.auth.connected
        ? ws.auth.colabAuthorized
          ? `Headless Colab REST & MCP API authorized as ${ws.auth.user?.email ?? "Google Account"}. No browser session required during training.`
          : `Signed in as ${ws.auth.user?.email ?? "Google Account"}, but your cached token lacks the colaboratory scope. Click Re-sign in for Colab Scope.`
        : "Sign in over localhost loopback OAuth with Google Colaboratory scope to provision GPU runtimes and stream datasets/weights natively in Rust.",
    ]),
    el("label", { style: "display:flex;align-items:center;gap:8px;font-size:12px;cursor:pointer" }, [
      driveBackupChk,
      "Enable optional Google Drive checkpoint/output backup (requests drive.file on next sign-in)",
    ]),
    el("div", { style: "display:flex;gap:8px;flex-wrap:wrap" }, [
      signInModalBtn,
      ws?.auth.connected ? disconnectBtn : null,
    ]),
    advancedOverride,
  ]);

  const dataDirInput = el("input", {
    class: "input mono",
    type: "text",
    value: ws?.dataDir ?? "",
    readonly: true,
  }) as HTMLInputElement;

  const openDataBtn = el("button", { class: "btn", type: "button" }, [
    svgIcon("folder", 13),
    "Open Data Folder",
  ]);
  openDataBtn.onclick = () => {
    if (ws?.dataDir) {
      void api.openLocalPath(ws.dataDir);
    }
  };

  const changeDataBtn = el("button", { class: "btn", type: "button" }, ["Change Data Folder"]);
  changeDataBtn.onclick = async () => {
    const picked = await api.pickFolder();
    if (typeof picked === "string") {
      try {
        const updated = await api.changeDataDirectory(picked);
        const [projects, runs, autolabelModels, evalBundles] = await Promise.all([
          api.listProjects(),
          api.listTrainingRuns(),
          api.listAutolabelModels(),
          api.listEvalBundles(),
        ]);
        await disposeAutoLabelSession();
        setState({
          workspace: updated,
          projects,
          runs,
          autolabelModels,
          evalBundles,
          activeProject: null,
          selectedRunId: runs[0]?.id ?? null,
          selectedEvalBundleId: null,
        });
        showToast(`Data folder set to ${picked}`, "success");
      } catch (err) {
        showToast(String(err), "error");
      }
    }
  };

  const storageCard = el("div", { class: "card", style: "display:flex;flex-direction:column;gap:10px" }, [
    el("div", { class: "card-title" }, ["Portable Storage Location"]),
    el("div", { class: "dim", style: "font-size:12px" }, [
      "Defaults to ./damnisium-data beside Damnisium.exe for full zero-dependency portability.",
    ]),
    dataDirInput,
    el("div", { style: "display:flex;gap:8px" }, [openDataBtn, changeDataBtn]),
  ]);

  const modelsCard = el("div", { class: "card", style: "display:flex;flex-direction:column;gap:10px" }, [
    el("div", { class: "card-title" }, ["Custom Model Registry"]),
  ]);
  const customModels = ws?.customModels ?? [];
  if (customModels.length === 0) {
    modelsCard.appendChild(
      el("div", { class: "dim", style: "font-size:12px" }, [
        "No custom models registered yet. Register local weights or HuggingFace IDs in Colab Training.",
      ])
    );
  } else {
    for (const cm of customModels) {
      const del = el("button", { class: "btn btn-sm btn-danger", type: "button" }, ["Remove"]);
      del.onclick = async () => {
        try {
          await api.deleteCustomModel(cm.id);
          const updated = await api.getWorkspaceInfo();
          setState({ workspace: updated });
        } catch (err) {
          showToast(String(err), "error");
        }
      };
      modelsCard.appendChild(
        el("div", { class: "run-row" }, [
          el("div", {}, [
            el("div", { style: "font-weight:600;font-size:12px" }, [cm.name]),
            el("div", { class: "dim mono", style: "font-size:11px" }, [
              `${cm.module} · ${cm.task} · ${cm.hfRepoId ?? cm.localRelPath ?? ""}`,
            ]),
          ]),
          del,
        ])
      );
    }
  }

  wrap.append(
    netCard,
    hfCard,
    oauthCard,
    storageCard,
    modelsCard,
    el("div", { class: "settings-footer" }, [`${APP_NAME} ${APP_VERSION} · Tej`])
  );
  root.appendChild(wrap);
  container.appendChild(root);
}
