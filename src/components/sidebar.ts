import { APP_NAME } from "../constants";
import { getState, setState } from "../state";
import type { AppView } from "../types";
import { el, svgIcon } from "../utils/dom";
import { openAboutModal } from "./about-modal";

export function renderSidebar(container: HTMLElement): void {
  const state = getState();
  container.innerHTML = "";
  container.className = `app-rail${state.sidebarExpanded ? "" : " collapsed"}`;

  const aboutLogo = el("div", { class: "rail-about-circle" }, [svgIcon("helpCircle", 17)]);

  const brandWrap = el("div", { class: "rail-brand-wrap", title: `About Us · ${APP_NAME}` }, [
    aboutLogo,
    el("div", { class: "rail-brand-text" }, [
      el("span", { class: "rail-brand-name" }, ["About Us"]),
    ]),
  ]);
  brandWrap.onclick = () => openAboutModal();

  const toggleBtn = el("button", {
    class: "rail-collapse-btn",
    type: "button",
    title: state.sidebarExpanded ? "Collapse sidebar" : "Expand sidebar",
  }, [svgIcon(state.sidebarExpanded ? "chevronLeft" : "chevronRight", 14)]);
  toggleBtn.onclick = () => setState({ sidebarExpanded: !state.sidebarExpanded });

  const header = el("div", { class: "rail-header" }, [
    brandWrap,
    state.sidebarExpanded ? toggleBtn : null,
  ]);

  const mainItems: { view: AppView; icon: string; label: string; disabled?: boolean }[] = [
    { view: "dashboard", icon: "grid", label: "Projects" },
    {
      view: "workspace",
      icon: "layers",
      label: "Dataset Studio",
      disabled: !state.activeProject,
    },
    { view: "train", icon: "cpu", label: "Colab Training" },
    { view: "hub", icon: "globe", label: "Uncensored Hub" },
    { view: "evaluate", icon: "flask", label: "Test & Evaluate" },
  ];

  const nav = el("div", { class: "rail-nav" });
  const indicator = el("div", { class: "rail-indicator", style: "opacity:0" });
  nav.appendChild(indicator);

  let activeIdx = -1;
  mainItems.forEach((item, idx) => {
    const isActive = state.activeView === item.view;
    if (isActive) activeIdx = idx;
    const btn = el(
      "button",
      {
        class: `rail-btn${isActive ? " active" : ""}`,
        type: "button",
        title: item.label,
        disabled: item.disabled,
      },
      [svgIcon(item.icon, 16), el("span", { class: "rail-label" }, [item.label])]
    );
    btn.onclick = () => {
      if (!item.disabled) {
        setState({ activeView: item.view });
      }
    };
    nav.appendChild(btn);
  });

  if (activeIdx >= 0) {
    const y = activeIdx * 40 + 8;
    indicator.style.opacity = "1";
    indicator.style.transform = `translateY(${y}px)`;
  }

  const spacer = el("div", { class: "rail-spacer" });

  const settingsBtn = el(
    "button",
    {
      class: `rail-btn${state.activeView === "settings" ? " active" : ""}`,
      type: "button",
      title: "Settings",
    },
    [svgIcon("settings", 16), el("span", { class: "rail-label" }, ["Settings"])]
  );
  settingsBtn.onclick = () => setState({ activeView: "settings" });

  const footerGroup = el("div", { class: "rail-footer-group" }, [
    settingsBtn,
    !state.sidebarExpanded ? toggleBtn : null,
  ]);

  container.append(header, nav, spacer, footerGroup);
}
