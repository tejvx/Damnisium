import { api } from "../api";
import { APP_NAME, APP_VERSION } from "../constants";
import { el, openModal, svgIcon } from "../utils/dom";

const DISCORD_INVITE_URL = "https://discord.gg/2APSU9YhND";

export function openAboutModal(): void {
  openModal(
    `About ${APP_NAME}`,
    (close) => {
      const discordBtn = el(
        "button",
        { class: "btn btn-sm", type: "button" },
        [svgIcon("external", 12), "Join Discord Community"]
      );
      discordBtn.onclick = () => void api.openExternalUrl(DISCORD_INVITE_URL);

      const closeBtn = el("button", { class: "btn btn-primary", type: "button" }, ["Close"]);
      closeBtn.onclick = close;

      return el("div", { class: "about-body" }, [
        el("img", { src: "/assets/damnisium.png", alt: APP_NAME, class: "about-emblem" }),
        el("div", { style: "display:flex;align-items:center;gap:8px;margin-top:4px" }, [
          el("span", { style: "font-size:16px;font-weight:600" }, [APP_NAME]),
          el("span", { class: "version-pill" }, [APP_VERSION]),
        ]),
        el("div", { class: "dim mono", style: "font-size:11px" }, [
          "Image · Audio · Text / Q&A ML Workstation",
        ]),
        el("div", { class: "dim mono", style: "font-size:11px;margin-top:2px" }, [
          "Created by Tej",
        ]),
        el(
          "div",
          {
            style:
              "width:100%;margin-top:6px;padding:12px;border-radius:var(--radius-md);background:var(--bg-1);border:1px solid var(--border-1);display:flex;flex-direction:column;align-items:center;gap:8px",
          },
          [
            el("div", { style: "font-size:12px;font-weight:600;color:var(--text-1)" }, [
              "Community & Support",
            ]),
            el("div", { class: "dim", style: "font-size:11.5px;line-height:1.45" }, [
              "Join our Discord community for model tips, troubleshooting, support, and fun.",
            ]),
            el("div", { style: "display:flex;align-items:center;gap:8px;flex-wrap:wrap;justify-content:center" }, [
              discordBtn,
              el("span", { class: "mono dim", style: "font-size:11px" }, [DISCORD_INVITE_URL]),
            ]),
          ]
        ),
        el(
          "div",
          { class: "modal-actions", style: "width:100%;justify-content:center;margin-top:10px" },
          [closeBtn]
        ),
      ]);
    },
    undefined,
    { hideHeaderClose: true, centerTitle: true }
  );
}
