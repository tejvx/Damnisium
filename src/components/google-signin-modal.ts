import { api } from "../api";
import { getState, setState } from "../state";
import type { OAuthProgressPayload, PreparedOAuthSession } from "../types";
import { el, openModal, showToast, svgIcon } from "../utils/dom";

export function openGoogleAccountModal(): void {
  const ws = getState().workspace;
  const email = ws?.auth.user?.email || "Google Account";

  openModal("Google & Colab Account", (close) => {
    const colabReady = Boolean(ws?.auth.colabAuthorized);
    const wrap = el("div", { style: "display:flex;flex-direction:column;gap:12px" });
    const statusBox = el("div", { class: "oauth-status-box" }, [
      el("div", { style: "display:flex;align-items:center;justify-content:space-between;gap:8px" }, [
        el("span", { style: "font-weight:600;color:var(--text-1);font-size:13px" }, [email]),
        el("span", { class: `badge ${colabReady ? "badge-success" : "badge-warn"}` }, [
          colabReady ? "Colab Ready" : "Colab Scope Needed",
        ]),
      ]),
      el("div", { class: "dim", style: "font-size:12px;line-height:1.45" }, [
        colabReady
          ? "Logs out of your linked Google & Colab account on this device and clears saved tokens."
          : "Signed in without the Google Colaboratory scope. Click Switch / Re-sign in to grant headless Colab GPU access.",
      ]),
    ]);

    const logoutBtn = el(
      "button",
      { class: "btn btn-danger", type: "button", style: "flex:1" },
      [svgIcon("logout", 13), "Logout of Colab"]
    );
    const switchBtn = el(
      "button",
      { class: "btn", type: "button", style: "flex:1" },
      [svgIcon("external", 13), "Switch Account"]
    );

    logoutBtn.onclick = async () => {
      logoutBtn.setAttribute("disabled", "true");
      switchBtn.setAttribute("disabled", "true");
      try {
        await api.disconnectGoogleAccount();
        const updated = await api.getWorkspaceInfo();
        setState({ workspace: updated });
        close();
        showToast("Logged out of Google & Colab", "info");
      } catch (err) {
        logoutBtn.removeAttribute("disabled");
        switchBtn.removeAttribute("disabled");
        showToast(String(err), "error");
      }
    };

    switchBtn.onclick = () => {
      close();
      openGoogleSignInModal();
    };

    wrap.append(statusBox, el("div", { class: "oauth-btn-row" }, [logoutBtn, switchBtn]));
    return wrap;
  });
}

export function openGoogleSignInModal(onVerified?: () => void): void {
  let completed = false;
  let disposed = false;
  let unlisten: (() => void) | null = null;
  let activeSession: PreparedOAuthSession | null = null;
  let copyTimer: number | null = null;

  openModal(
    "Sign in with Google",
    (close) => {
      const wrap = el("div", { style: "display:flex;flex-direction:column;gap:12px" });
      const desc = el("div", { class: "dim", style: "font-size:12px;line-height:1.5" }, [
        "Connect your Google account for headless Colab GPU compute. Verification completes locally over loopback HTTP.",
      ]);

      const statusBox = el("div", { class: "oauth-status-box" });
      const actionsRow = el("div", { class: "oauth-btn-row" });

      const signInBtn = el(
        "button",
        { class: "btn btn-primary", type: "button", style: "flex:1", disabled: true },
        [svgIcon("external", 13), "Sign in"]
      );
      const copyBtn = el(
        "button",
        { class: "btn", type: "button", style: "flex:1", disabled: true },
        [svgIcon("copy", 13), "Copy sign in link"]
      );

      const renderStatus = (
        kind: "idle" | "waiting" | "verifying" | "error",
        text: string,
        detail?: string
      ) => {
        statusBox.innerHTML = "";
        statusBox.className = `oauth-status-box${kind === "error" ? " error" : ""}`;
        if (kind === "verifying") {
          statusBox.appendChild(
            el("div", { style: "display:flex;align-items:center;gap:8px;font-size:12px" }, [
              el("div", { class: "spinner" }),
              el("span", {}, [text]),
            ])
          );
        } else {
          statusBox.appendChild(el("div", { style: "font-size:12px" }, [text]));
        }
        if (detail) {
          statusBox.appendChild(
            el("div", { class: "mono dim", style: "font-size:11px;word-break:break-all" }, [detail])
          );
        }
      };

      const startPrepare = async () => {
        signInBtn.setAttribute("disabled", "true");
        copyBtn.setAttribute("disabled", "true");
        activeSession = null;
        renderStatus("waiting", "Starting localhost OAuth callback listener...");
        try {
          const sess = await api.prepareGoogleOAuth();
          if (disposed) return;
          activeSession = sess;
          signInBtn.removeAttribute("disabled");
          copyBtn.removeAttribute("disabled");
          renderStatus(
            "idle",
            "Ready. Click Sign in or copy the link to any signed-in browser.",
            sess.redirectUri
          );
        } catch (err) {
          if (disposed) return;
          renderStatus("error", "OAuth listener initialization failed.", String(err));
          const retryBtn = el("button", { class: "btn btn-sm", type: "button" }, [
            svgIcon("refresh", 12),
            "Try Again",
          ]);
          retryBtn.onclick = () => void startPrepare();
          statusBox.appendChild(retryBtn);
        }
      };

      signInBtn.onclick = () => {
        if (!activeSession) return;
        renderStatus(
          "waiting",
          "Waiting for Google sign-in confirmation in your browser...",
          activeSession.redirectUri
        );
        void api.openExternalUrl(activeSession.authUrl);
      };

      copyBtn.onclick = async () => {
        if (!activeSession) return;
        try {
          await navigator.clipboard.writeText(activeSession.authUrl);
          if (copyTimer !== null) window.clearTimeout(copyTimer);
          copyBtn.innerHTML = "";
          copyBtn.append(svgIcon("check", 13), "Copied");
          renderStatus(
            "waiting",
            "Sign-in URL copied. Complete authorization in that browser tab.",
            activeSession.redirectUri
          );
          copyTimer = window.setTimeout(() => {
            if (!disposed) {
              copyBtn.innerHTML = "";
              copyBtn.append(svgIcon("copy", 13), "Copy sign in link");
            }
          }, 2000);
        } catch (err) {
          showToast(`Clipboard copy failed: ${String(err)}`, "error");
        }
      };

      void api
        .onGoogleOAuthState((payload: OAuthProgressPayload) => {
          if (disposed) return;
          if (payload.phase === "verifying") {
            signInBtn.setAttribute("disabled", "true");
            copyBtn.setAttribute("disabled", "true");
            renderStatus("verifying", "Verifying account...");
          } else if (payload.phase === "verified") {
            completed = true;
            void api.getWorkspaceInfo().then((ws) => {
              setState({ workspace: ws });
              const email = ws.auth.user?.email ?? "Google Account";
              showToast(`Verified Google account: ${email}`, "success");
              close();
              onVerified?.();
            });
          } else if (payload.phase === "error") {
            signInBtn.setAttribute("disabled", "true");
            copyBtn.setAttribute("disabled", "true");
            renderStatus(
              "error",
              payload.message || "Authentication failed",
              payload.error ?? "Account verification did not succeed."
            );
            const retryBtn = el("button", { class: "btn btn-sm btn-primary", type: "button" }, [
              svgIcon("refresh", 12),
              "Retry Sign in",
            ]);
            retryBtn.onclick = () => void startPrepare();
            statusBox.appendChild(retryBtn);
          }
        })
        .then((fn) => {
          if (disposed) {
            fn();
          } else {
            unlisten = fn;
          }
        });

      actionsRow.append(signInBtn, copyBtn);
      wrap.append(desc, actionsRow, statusBox);
      void startPrepare();
      return wrap;
    },
    () => {
      disposed = true;
      if (copyTimer !== null) window.clearTimeout(copyTimer);
      if (unlisten) {
        unlisten();
        unlisten = null;
      }
      if (!completed) {
        void api.cancelGoogleOAuth();
      }
    }
  );
}
