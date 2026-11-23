import type { TrainingMetricPoint } from "../types";

export function renderMetricChartSvg(
  metrics: TrainingMetricPoint[],
  mode: "loss" | "score"
): SVGSVGElement {
  const ns = "http://www.w3.org/2000/svg";
  const w = 460;
  const h = 150;
  const padL = 38;
  const padR = 12;
  const padT = 12;
  const padB = 22;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", String(h));

  const seriesA: { x: number; y: number }[] = [];
  const seriesB: { x: number; y: number }[] = [];

  for (let i = 0; i < metrics.length; i++) {
    const m = metrics[i];
    const xVal = m.epoch || m.step || i + 1;
    if (mode === "loss") {
      if (m.trainLoss !== null && Number.isFinite(m.trainLoss)) {
        seriesA.push({ x: xVal, y: m.trainLoss });
      }
      if (m.valLoss !== null && Number.isFinite(m.valLoss)) {
        seriesB.push({ x: xVal, y: m.valLoss });
      }
    } else if (m.score !== null && Number.isFinite(m.score)) {
      seriesA.push({ x: xVal, y: m.score });
    }
  }

  for (let i = 0; i <= 3; i++) {
    const gy = padT + (plotH * i) / 3;
    const gridLine = document.createElementNS(ns, "line");
    gridLine.setAttribute("x1", String(padL));
    gridLine.setAttribute("y1", String(gy));
    gridLine.setAttribute("x2", String(w - padR));
    gridLine.setAttribute("y2", String(gy));
    gridLine.setAttribute("stroke", "rgba(255,255,255,0.06)");
    gridLine.setAttribute("stroke-width", "1");
    svg.appendChild(gridLine);
  }

  const allPts = [...seriesA, ...seriesB];
  if (allPts.length === 0) {
    const emptyText = document.createElementNS(ns, "text");
    emptyText.setAttribute("x", String(w / 2));
    emptyText.setAttribute("y", String(h / 2));
    emptyText.setAttribute("text-anchor", "middle");
    emptyText.setAttribute("fill", "#71717a");
    emptyText.setAttribute("font-size", "11");
    emptyText.textContent = "Waiting for epoch metrics";
    svg.appendChild(emptyText);
    return svg;
  }

  const minX = Math.min(...allPts.map((p) => p.x));
  const maxX = Math.max(minX + 1, ...allPts.map((p) => p.x));
  let minY = Math.min(...allPts.map((p) => p.y));
  let maxY = Math.max(...allPts.map((p) => p.y));
  if (Math.abs(maxY - minY) < 1e-6) {
    minY = Math.max(0, minY - 0.1);
    maxY = maxY + 0.1;
  }

  const toPx = (pt: { x: number; y: number }) => ({
    px: padL + ((pt.x - minX) / (maxX - minX)) * plotW,
    py: padT + (1 - (pt.y - minY) / (maxY - minY)) * plotH,
  });

  const drawPath = (pts: { x: number; y: number }[], stroke: string) => {
    if (pts.length === 0) return;
    const mapped = pts.map(toPx);
    if (mapped.length > 1) {
      const d = mapped
        .map((p, idx) => `${idx === 0 ? "M" : "L"}${p.px.toFixed(1)},${p.py.toFixed(1)}`)
        .join(" ");
      const path = document.createElementNS(ns, "path");
      path.setAttribute("d", d);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", stroke);
      path.setAttribute("stroke-width", "2");
      svg.appendChild(path);
    }

    const last = mapped[mapped.length - 1];
    const dot = document.createElementNS(ns, "circle");
    dot.setAttribute("cx", last.px.toFixed(1));
    dot.setAttribute("cy", last.py.toFixed(1));
    dot.setAttribute("r", mapped.length === 1 ? "4" : "3");
    dot.setAttribute("fill", stroke);
    svg.appendChild(dot);
  };

  const fmtY = (v: number): string => (Math.abs(v) >= 10 ? v.toFixed(2) : v.toFixed(3));
  const fmtEp = (v: number): string =>
    Math.abs(v - Math.round(v)) < 0.05 ? String(Math.round(v)) : v.toFixed(1);

  drawPath(seriesA, mode === "loss" ? "#ec4899" : "#10b981");
  drawPath(seriesB, "#f59e0b");

  const labelMax = document.createElementNS(ns, "text");
  labelMax.setAttribute("x", String(padL - 5));
  labelMax.setAttribute("y", String(padT + 8));
  labelMax.setAttribute("text-anchor", "end");
  labelMax.setAttribute("fill", "#71717a");
  labelMax.setAttribute("font-size", "10");
  labelMax.textContent = fmtY(maxY);
  svg.appendChild(labelMax);

  const labelMin = document.createElementNS(ns, "text");
  labelMin.setAttribute("x", String(padL - 5));
  labelMin.setAttribute("y", String(padT + plotH));
  labelMin.setAttribute("text-anchor", "end");
  labelMin.setAttribute("fill", "#71717a");
  labelMin.setAttribute("font-size", "10");
  labelMin.textContent = fmtY(minY);
  svg.appendChild(labelMin);

  const labelXStart = document.createElementNS(ns, "text");
  labelXStart.setAttribute("x", String(padL));
  labelXStart.setAttribute("y", String(h - 5));
  labelXStart.setAttribute("text-anchor", "start");
  labelXStart.setAttribute("fill", "#71717a");
  labelXStart.setAttribute("font-size", "10");
  labelXStart.textContent = `Ep ${fmtEp(minX)}`;
  svg.appendChild(labelXStart);

  const lastPtX = Math.max(...allPts.map((p) => p.x));
  if (allPts.length > 1 || lastPtX !== minX) {
    const labelXEnd = document.createElementNS(ns, "text");
    labelXEnd.setAttribute("x", String(w - padR));
    labelXEnd.setAttribute("y", String(h - 5));
    labelXEnd.setAttribute("text-anchor", "end");
    labelXEnd.setAttribute("fill", "#71717a");
    labelXEnd.setAttribute("font-size", "10");
    labelXEnd.textContent = `Ep ${fmtEp(lastPtX)}`;
    svg.appendChild(labelXEnd);
  }

  return svg;
}
