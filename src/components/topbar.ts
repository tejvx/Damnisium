import { APP_NAME, APP_VERSION, MODULE_TASKS } from "../constants";
import { getState, refreshConnectivity, setState } from "../state";
import { el, showToast, svgIcon } from "../utils/dom";
import { confirmAndDeleteProject } from "./dashboard";
import { openGoogleAccountModal, openGoogleSignInModal } from "./google-signin-modal";

export function renderTopbar(container: HTMLElement): void {
  const state = getState();
  container.innerHTML = "";

  const left = el("div", { class: "topbar-left" }, [
    el("span", { class: "topbar-title" }, [APP_NAME]),
    el("span", { class: "version-pill" }, [APP_VERSION]),
  ]);

  if (state.activeProject) {
    const activeProj = state.activeProject;
    const modLabel = MODULE_TASKS[activeProj.module].label;
    const delProjBtn = el(
      "button",
      {
        class: "btn btn-sm btn-ghost",
        type: "button",
        title: `Delete Project (${activeProj.name})`,
        style: "padding:3px 6px",
      },
      [svgIcon("trash", 12)]
    );
    delProjBtn.onclick = () => confirmAndDeleteProject(activeProj.id, activeProj.name);
    left.append(
      el("span", { class: "topbar-sep" }, ["/"]),
      el("span", { class: "topbar-project" }, [activeProj.name]),
      el("span", { class: "badge badge-accent" }, [modLabel]),
      el("span", { class: "badge" }, [activeProj.task]),
      delProjBtn
    );
  }

  const right = el("div", { class: "topbar-right" });

  const conn = state.connectivity;
  const connText = state.connectivityChecking
    ? "Checking..."
    : conn.online
    ? conn.latencyMs !== null
      ? `Online · ${conn.latencyMs}ms`
      : "Online"
    : "Offline Mode";
  const netBtn = el(
    "button",
    {
      class: `btn btn-sm ${conn.online ? "btn-ghost" : ""}`,
      type: "button",
      title: `${conn.message} — Click to retry connection`,
      disabled: state.connectivityChecking,
    },
    [
      el(
        "span",
        {
          class: `badge ${conn.online ? "badge-success" : "badge-warn"}`,
          style: "border:none;padding:0 4px;background:transparent",
        },
        [connText]
      ),
      svgIcon("refresh", 12),
    ]
  );
  netBtn.onclick = async () => {
    const res = await refreshConnectivity();
    showToast(res.online ? `Online (${res.latencyMs ?? 0}ms)` : "Still in Offline Mode", res.online ? "success" : "info");
  };
  right.appendChild(netBtn);

  const auth = state.workspace?.auth;
  const authBtn = el(
    "button",
    {
      class: `btn btn-sm ${auth?.connected ? "" : "btn-ghost"}`,
      type: "button",
      title: auth?.connected
        ? "Manage or logout of Google & Colab account"
        : "Sign in to Google & Colab Compute",
    },
    [
      el(
        "span",
        {
          class: `badge ${auth?.connected && auth?.colabAuthorized ? "badge-success" : "badge-warn"}`,
          style: "border:none;padding:0 4px;background:transparent",
        },
        [
          auth?.connected
            ? auth?.colabAuthorized
              ? "Colab Linked"
              : "Colab Scope Needed"
            : "Link Google",
        ]
      ),
      auth?.user?.email ? el("span", { class: "muted" }, [auth.user.email]) : null,
    ]
  );
  authBtn.onclick = () => {
    if (auth?.connected) {
      openGoogleAccountModal();
    } else {
      openGoogleSignInModal();
    }
  };
  right.appendChild(authBtn);

  const evalBtn = el("button", { class: "btn btn-sm", type: "button" }, [
    svgIcon("flask", 13),
    "Test & Evaluate",
  ]);
  evalBtn.onclick = () => setState({ activeView: "evaluate" });
  right.appendChild(evalBtn);

  if (state.activeProject) {
    const trainBtn = el("button", { class: "btn btn-sm btn-primary", type: "button" }, [
      svgIcon("cpu", 13),
      "Train on Colab",
    ]);
    trainBtn.onclick = () => setState({ activeView: "train" });
    right.appendChild(trainBtn);
  }

  container.append(left, right);
}
