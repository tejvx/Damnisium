import type { BoundingBox, LabelClass, PolygonPoint, PolygonShape } from "../../types";
import { randomId } from "../../utils/format";

export type CanvasTool = "select" | "box" | "polygon";

export interface SelectedShape {
  kind: "box" | "polygon";
  id: string;
}

export interface CanvasViewState {
  zoom: number;
  offsetX: number;
  offsetY: number;
}

interface HandleHit {
  kind: "box-handle" | "box-body" | "poly-vertex" | "poly-body";
  id: string;
  handleIdx: number;
}

export class ImageCanvasEngine {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private img: HTMLImageElement | null = null;
  private imgW = 1;
  private imgH = 1;

  private zoom = 1;
  private offsetX = 0;
  private offsetY = 0;
  private hasFitted = false;

  private tool: CanvasTool = "box";
  private activeClassId = "";
  private classes = new Map<string, LabelClass>();
  private boxes: BoundingBox[] = [];
  private polygons: PolygonShape[] = [];
  private selected: SelectedShape | null = null;

  private pointerX = 0;
  private pointerY = 0;
  private pointerInside = false;
  private spaceDown = false;

  private isPanning = false;
  private panStartX = 0;
  private panStartY = 0;
  private panOriginX = 0;
  private panOriginY = 0;

  private drawStartNorm: { x: number; y: number } | null = null;
  private draftBox: BoundingBox | null = null;
  private draftPoly: PolygonPoint[] = [];

  private activeDrag:
    | {
      mode: "move-box" | "resize-box" | "move-vertex";
      id: string;
      handleIdx: number;
      startNormX: number;
      startNormY: number;
      moved: boolean;
      origBox?: BoundingBox;
      origPoint?: PolygonPoint;
    }
    | null = null;

  private dirty = false;
  private rafPending = false;
  private resizeObs: ResizeObserver;

  private loadedSrc: string | null = null;
  private onMutate: (boxes: BoundingBox[], polygons: PolygonShape[]) => void;
  private onSelectChange: (sel: SelectedShape | null) => void;

  constructor(
    canvas: HTMLCanvasElement,
    onMutate: (boxes: BoundingBox[], polygons: PolygonShape[]) => void,
    onSelectChange: (sel: SelectedShape | null) => void
  ) {
    this.canvas = canvas;
    const context = canvas.getContext("2d");
    if (!context) {
      throw new Error("2D canvas context unavailable");
    }
    this.ctx = context;
    this.onMutate = onMutate;
    this.onSelectChange = onSelectChange;

    this.resizeObs = new ResizeObserver(() => this.resizeCanvas());
    if (canvas.parentElement) {
      this.resizeObs.observe(canvas.parentElement);
    }
    this.bindEvents();
    this.resizeCanvas();
  }

  destroy(): void {
    this.resizeObs.disconnect();
    window.removeEventListener("keydown", this.handleKeyDown);
    window.removeEventListener("keyup", this.handleKeyUp);
    window.removeEventListener("mouseup", this.handleMouseUp);
  }

  setTool(tool: CanvasTool): void {
    this.tool = tool;
    this.draftPoly = [];
    this.draftBox = null;
    this.drawStartNorm = null;
    this.updateCursor();
    this.scheduleRender();
  }

  setActiveClass(classId: string): void {
    this.activeClassId = classId;
  }

  setData(
    classes: LabelClass[],
    boxes: BoundingBox[],
    polygons: PolygonShape[],
    selected: SelectedShape | null
  ): void {
    this.classes.clear();
    for (const c of classes) {
      this.classes.set(c.id, c);
    }
    this.boxes = boxes.map((b) => ({ ...b }));
    this.polygons = polygons.map((p) => ({
      ...p,
      points: p.points.map((pt) => ({ ...pt })),
    }));
    this.selected = selected;
    this.scheduleRender();
  }

  getViewState(): CanvasViewState | null {
    if (!this.img || !this.hasFitted || this.zoom <= 0) {
      return null;
    }
    return { zoom: this.zoom, offsetX: this.offsetX, offsetY: this.offsetY };
  }

  loadImage(src: string | null, preserveView?: CanvasViewState | null): void {
    this.draftPoly = [];
    this.draftBox = null;
    this.drawStartNorm = null;
    this.activeDrag = null;
    if (!src) {
      this.loadedSrc = null;
      this.img = null;
      this.hasFitted = false;
      this.scheduleRender();
      return;
    }
    const sameImage = this.loadedSrc === src && this.img !== null;
    this.loadedSrc = src;
    if (sameImage) {
      if (preserveView && preserveView.zoom > 0) {
        this.zoom = preserveView.zoom;
        this.offsetX = preserveView.offsetX;
        this.offsetY = preserveView.offsetY;
        this.hasFitted = true;
      } else if (!this.hasFitted) {
        this.fitView();
        return;
      }
      this.scheduleRender();
      return;
    }
    if (preserveView && preserveView.zoom > 0) {
      this.zoom = preserveView.zoom;
      this.offsetX = preserveView.offsetX;
      this.offsetY = preserveView.offsetY;
      this.hasFitted = true;
    } else {
      this.hasFitted = false;
    }
    const targetSrc = src;
    const image = new Image();
    image.onload = () => {
      if (this.loadedSrc !== targetSrc) return;
      this.img = image;
      this.imgW = image.naturalWidth || 1;
      this.imgH = image.naturalHeight || 1;
      if (preserveView && preserveView.zoom > 0) {
        this.zoom = preserveView.zoom;
        this.offsetX = preserveView.offsetX;
        this.offsetY = preserveView.offsetY;
        this.hasFitted = true;
        this.scheduleRender();
      } else {
        this.fitView();
      }
    };
    image.onerror = () => {
      if (this.loadedSrc !== targetSrc) return;
      this.img = null;
      this.hasFitted = false;
      this.scheduleRender();
    };
    image.src = src;
  }

  fitView(): void {
    if (!this.img) {
      this.hasFitted = false;
      this.scheduleRender();
      return;
    }
    const rect = this.canvas.getBoundingClientRect();
    const vw = this.canvas.parentElement?.clientWidth || rect.width;
    const vh = this.canvas.parentElement?.clientHeight || rect.height;
    if (vw < 32 || vh < 32) {
      this.hasFitted = false;
      return;
    }
    const pad = 28;
    const availW = Math.max(32, vw - pad * 2);
    const availH = Math.max(32, vh - pad * 2);
    this.zoom = Math.min(availW / this.imgW, availH / this.imgH);
    this.offsetX = (vw - this.imgW * this.zoom) / 2;
    this.offsetY = (vh - this.imgH * this.zoom) / 2;
    this.hasFitted = true;
    this.scheduleRender();
  }

  zoomBy(factor: number): void {
    const rect = this.canvas.getBoundingClientRect();
    this.zoomAt(rect.width / 2, rect.height / 2, factor);
  }

  hasReadyPolygon(): boolean {
    return this.tool === "polygon" && this.draftPoly.length >= 3;
  }

  finishPolygonIfReady(): void {
    if (this.tool === "polygon" && this.draftPoly.length >= 3) {
      const poly: PolygonShape = {
        id: randomId("poly"),
        classId: this.activeClassId,
        points: this.draftPoly.map((p) => ({ ...p })),
      };
      this.polygons = [...this.polygons, poly];
      this.draftPoly = [];
      this.selected = { kind: "polygon", id: poly.id };
      this.onSelectChange(this.selected);
      this.onMutate(this.boxes, this.polygons);
      this.scheduleRender();
    }
  }

  cancelDraft(): void {
    this.draftPoly = [];
    this.draftBox = null;
    this.drawStartNorm = null;
    this.scheduleRender();
  }

  private resizeCanvas(): void {
    const parent = this.canvas.parentElement;
    if (!parent) return;
    const w = parent.clientWidth;
    const h = parent.clientHeight;
    if (w < 2 || h < 2) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (this.img && !this.hasFitted) {
      this.fitView();
    } else {
      this.scheduleRender();
    }
  }

  private screenToNorm(sx: number, sy: number): { x: number; y: number } {
    const nx = (sx - this.offsetX) / Math.max(1e-6, this.imgW * this.zoom);
    const ny = (sy - this.offsetY) / Math.max(1e-6, this.imgH * this.zoom);
    return {
      x: Math.max(0, Math.min(1, nx)),
      y: Math.max(0, Math.min(1, ny)),
    };
  }

  private normToScreen(nx: number, ny: number): { x: number; y: number } {
    return {
      x: this.offsetX + nx * this.imgW * this.zoom,
      y: this.offsetY + ny * this.imgH * this.zoom,
    };
  }

  private zoomAt(cx: number, cy: number, factor: number): void {
    const prevZoom = this.zoom;
    const nextZoom = Math.max(0.05, Math.min(16, prevZoom * factor));
    const scale = nextZoom / prevZoom;
    this.offsetX = cx - (cx - this.offsetX) * scale;
    this.offsetY = cy - (cy - this.offsetY) * scale;
    this.zoom = nextZoom;
    this.scheduleRender();
  }

  private updateCursor(): void {
    if (this.isPanning || this.spaceDown) {
      this.canvas.style.cursor = "grab";
    } else if (this.tool === "box" || this.tool === "polygon") {
      this.canvas.style.cursor = "crosshair";
    } else {
      this.canvas.style.cursor = "default";
    }
  }

  private handleKeyDown = (e: KeyboardEvent): void => {
    const tag = (document.activeElement as HTMLElement | null)?.tagName;
    if (e.code === "Space" && tag !== "INPUT" && tag !== "TEXTAREA" && tag !== "SELECT") {
      e.preventDefault();
      this.spaceDown = true;
      this.updateCursor();
    }
  };

  private handleKeyUp = (e: KeyboardEvent): void => {
    if (e.code === "Space") {
      this.spaceDown = false;
      this.updateCursor();
    }
  };

  private handleMouseUp = (): void => {
    if (this.isPanning) {
      this.isPanning = false;
      this.updateCursor();
    }
    if (this.activeDrag) {
      const wasMoved = this.activeDrag.moved;
      this.activeDrag = null;
      if (wasMoved) {
        this.onMutate(this.boxes, this.polygons);
      }
    }
    if (this.draftBox && this.drawStartNorm) {
      const finished = this.draftBox;
      this.draftBox = null;
      this.drawStartNorm = null;
      if (finished.w > 0.006 && finished.h > 0.006) {
        this.boxes = [...this.boxes, finished];
        this.selected = { kind: "box", id: finished.id };
        this.onSelectChange(this.selected);
        this.onMutate(this.boxes, this.polygons);
      }
      this.scheduleRender();
    }
  };

  private bindEvents(): void {
    window.addEventListener("keydown", this.handleKeyDown);
    window.addEventListener("keyup", this.handleKeyUp);
    window.addEventListener("mouseup", this.handleMouseUp);

    this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    this.canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const rect = this.canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      this.zoomAt(sx, sy, e.deltaY < 0 ? 1.12 : 0.89);
    });

    this.canvas.addEventListener("mousedown", (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;

      if (e.button === 1 || e.button === 2 || this.spaceDown) {
        this.isPanning = true;
        this.panStartX = sx;
        this.panStartY = sy;
        this.panOriginX = this.offsetX;
        this.panOriginY = this.offsetY;
        this.updateCursor();
        return;
      }

      if (!this.img) return;
      const norm = this.screenToNorm(sx, sy);

      if (this.tool === "select") {
        const hit = this.hitTest(sx, sy);
        if (!hit) {
          this.selected = null;
          this.onSelectChange(null);
          this.scheduleRender();
          return;
        }
        if (hit.kind === "box-handle") {
          const box = this.boxes.find((b) => b.id === hit.id);
          if (box) {
            this.selected = { kind: "box", id: box.id };
            this.onSelectChange(this.selected);
            this.activeDrag = {
              mode: "resize-box",
              id: box.id,
              handleIdx: hit.handleIdx,
              startNormX: norm.x,
              startNormY: norm.y,
              moved: false,
              origBox: { ...box },
            };
          }
        } else if (hit.kind === "box-body") {
          const box = this.boxes.find((b) => b.id === hit.id);
          if (box) {
            this.selected = { kind: "box", id: box.id };
            this.onSelectChange(this.selected);
            this.activeDrag = {
              mode: "move-box",
              id: box.id,
              handleIdx: -1,
              startNormX: norm.x,
              startNormY: norm.y,
              moved: false,
              origBox: { ...box },
            };
          }
        } else if (hit.kind === "poly-vertex") {
          const poly = this.polygons.find((p) => p.id === hit.id);
          if (poly) {
            this.selected = { kind: "polygon", id: poly.id };
            this.onSelectChange(this.selected);
            this.activeDrag = {
              mode: "move-vertex",
              id: poly.id,
              handleIdx: hit.handleIdx,
              startNormX: norm.x,
              startNormY: norm.y,
              moved: false,
              origPoint: { ...poly.points[hit.handleIdx] },
            };
          }
        } else if (hit.kind === "poly-body") {
          this.selected = { kind: "polygon", id: hit.id };
          this.onSelectChange(this.selected);
        }
        this.scheduleRender();
        return;
      }

      if (!this.activeClassId) return;

      if (this.tool === "box") {
        this.drawStartNorm = norm;
        this.draftBox = {
          id: randomId("box"),
          classId: this.activeClassId,
          x: norm.x,
          y: norm.y,
          w: 0,
          h: 0,
        };
        this.scheduleRender();
      } else if (this.tool === "polygon") {
        if (this.draftPoly.length >= 3) {
          const first = this.normToScreen(this.draftPoly[0].x, this.draftPoly[0].y);
          const dist = Math.hypot(sx - first.x, sy - first.y);
          if (dist < 11) {
            this.finishPolygonIfReady();
            return;
          }
        }
        this.draftPoly.push(norm);
        this.scheduleRender();
      }
    });

    this.canvas.addEventListener("dblclick", () => {
      if (this.tool === "polygon" && this.draftPoly.length >= 3) {
        this.finishPolygonIfReady();
      }
    });

    this.canvas.addEventListener("mousemove", (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      this.pointerX = sx;
      this.pointerY = sy;
      this.pointerInside = true;

      if (this.isPanning) {
        this.offsetX = this.panOriginX + (sx - this.panStartX);
        this.offsetY = this.panOriginY + (sy - this.panStartY);
        this.scheduleRender();
        return;
      }

      const norm = this.screenToNorm(sx, sy);

      if (this.activeDrag) {
        const dx = norm.x - this.activeDrag.startNormX;
        const dy = norm.y - this.activeDrag.startNormY;
        if (Math.abs(dx) > 1e-4 || Math.abs(dy) > 1e-4) {
          this.activeDrag.moved = true;
        }
        if (this.activeDrag.mode === "move-box" && this.activeDrag.origBox) {
          const orig = this.activeDrag.origBox;
          const b = this.boxes.find((item) => item.id === this.activeDrag!.id);
          if (b) {
            b.x = Math.max(0, Math.min(1 - orig.w, orig.x + dx));
            b.y = Math.max(0, Math.min(1 - orig.h, orig.y + dy));
          }
        } else if (this.activeDrag.mode === "resize-box" && this.activeDrag.origBox) {
          const orig = this.activeDrag.origBox;
          const b = this.boxes.find((item) => item.id === this.activeDrag!.id);
          if (b) {
            let x1 = orig.x;
            let y1 = orig.y;
            let x2 = orig.x + orig.w;
            let y2 = orig.y + orig.h;
            const hi = this.activeDrag.handleIdx;
            if (hi === 0 || hi === 3) x1 = Math.min(x2 - 0.005, norm.x);
            if (hi === 1 || hi === 2) x2 = Math.max(x1 + 0.005, norm.x);
            if (hi === 0 || hi === 1) y1 = Math.min(y2 - 0.005, norm.y);
            if (hi === 2 || hi === 3) y2 = Math.max(y1 + 0.005, norm.y);
            b.x = Math.max(0, x1);
            b.y = Math.max(0, y1);
            b.w = Math.min(1 - b.x, x2 - x1);
            b.h = Math.min(1 - b.y, y2 - y1);
          }
        } else if (this.activeDrag.mode === "move-vertex") {
          const poly = this.polygons.find((p) => p.id === this.activeDrag!.id);
          if (poly && poly.points[this.activeDrag.handleIdx]) {
            poly.points[this.activeDrag.handleIdx] = { x: norm.x, y: norm.y };
          }
        }
        this.scheduleRender();
        return;
      }

      if (this.tool === "box" && this.drawStartNorm && this.draftBox) {
        const x1 = Math.min(this.drawStartNorm.x, norm.x);
        const y1 = Math.min(this.drawStartNorm.y, norm.y);
        const x2 = Math.max(this.drawStartNorm.x, norm.x);
        const y2 = Math.max(this.drawStartNorm.y, norm.y);
        this.draftBox.x = x1;
        this.draftBox.y = y1;
        this.draftBox.w = x2 - x1;
        this.draftBox.h = y2 - y1;
      }

      this.scheduleRender();
    });

    this.canvas.addEventListener("mouseleave", () => {
      this.pointerInside = false;
      this.scheduleRender();
    });
  }

  private hitTest(sx: number, sy: number): HandleHit | null {
    if (this.selected?.kind === "box") {
      const sb = this.boxes.find((b) => b.id === this.selected!.id);
      if (sb) {
        const corners = this.boxCornersScreen(sb);
        for (let i = 0; i < corners.length; i++) {
          if (Math.hypot(sx - corners[i].x, sy - corners[i].y) <= 8) {
            return { kind: "box-handle", id: sb.id, handleIdx: i };
          }
        }
      }
    }

    if (this.selected?.kind === "polygon") {
      const sp = this.polygons.find((p) => p.id === this.selected!.id);
      if (sp) {
        for (let i = 0; i < sp.points.length; i++) {
          const pt = this.normToScreen(sp.points[i].x, sp.points[i].y);
          if (Math.hypot(sx - pt.x, sy - pt.y) <= 8) {
            return { kind: "poly-vertex", id: sp.id, handleIdx: i };
          }
        }
      }
    }

    for (let i = this.boxes.length - 1; i >= 0; i--) {
      const b = this.boxes[i];
      const tl = this.normToScreen(b.x, b.y);
      const br = this.normToScreen(b.x + b.w, b.y + b.h);
      if (sx >= tl.x && sx <= br.x && sy >= tl.y && sy <= br.y) {
        return { kind: "box-body", id: b.id, handleIdx: -1 };
      }
    }

    for (let i = this.polygons.length - 1; i >= 0; i--) {
      const p = this.polygons[i];
      const pts = p.points.map((pt) => this.normToScreen(pt.x, pt.y));
      if (this.pointInPoly(sx, sy, pts)) {
        return { kind: "poly-body", id: p.id, handleIdx: -1 };
      }
    }
    return null;
  }

  private pointInPoly(x: number, y: number, pts: { x: number; y: number }[]): boolean {
    let inside = false;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const xi = pts[i].x;
      const yi = pts[i].y;
      const xj = pts[j].x;
      const yj = pts[j].y;
      const intersect =
        yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-9) + xi;
      if (intersect) inside = !inside;
    }
    return inside;
  }

  private boxCornersScreen(b: BoundingBox): { x: number; y: number }[] {
    const tl = this.normToScreen(b.x, b.y);
    const br = this.normToScreen(b.x + b.w, b.y + b.h);
    return [
      { x: tl.x, y: tl.y },
      { x: br.x, y: tl.y },
      { x: br.x, y: br.y },
      { x: tl.x, y: br.y },
    ];
  }

  private scheduleRender(): void {
    this.dirty = true;
    if (this.rafPending) return;
    this.rafPending = true;
    requestAnimationFrame(() => {
      this.rafPending = false;
      if (this.dirty) {
        this.dirty = false;
        this.render();
      }
    });
  }

  private render(): void {
    const rect = this.canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, w, h);

    if (!this.img) {
      ctx.fillStyle = "#667389";
      ctx.font = "13px Segoe UI, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Import or select an image to start labeling", w / 2, h / 2);
      return;
    }

    const drawW = this.imgW * this.zoom;
    const drawH = this.imgH * this.zoom;
    ctx.drawImage(this.img, this.offsetX, this.offsetY, drawW, drawH);

    const drawBox = (b: BoundingBox, isDraft: boolean) => {
      const cls = this.classes.get(b.classId);
      const color = cls?.color ?? "#38bdf8";
      const tl = this.normToScreen(b.x, b.y);
      const bw = b.w * drawW;
      const bh = b.h * drawH;
      const isSelected = !isDraft && this.selected?.kind === "box" && this.selected.id === b.id;

      ctx.fillStyle = `${color}26`;
      ctx.fillRect(tl.x, tl.y, bw, bh);
      ctx.strokeStyle = isSelected ? "#ffffff" : color;
      ctx.lineWidth = isSelected ? 2.2 : 1.7;
      ctx.strokeRect(tl.x, tl.y, bw, bh);

      if (cls) {
        const label =
          b.confidence !== undefined && b.confidence !== null
            ? `${cls.name} ${Math.round(b.confidence * 100)}%`
            : cls.name;
        ctx.font = "600 11px Segoe UI, sans-serif";
        const tw = ctx.measureText(label).width + 10;
        const ty = Math.max(0, tl.y - 18);
        ctx.fillStyle = color;
        ctx.fillRect(tl.x, ty, tw, 18);
        ctx.fillStyle = "#090b0f";
        ctx.textAlign = "left";
        ctx.fillText(label, tl.x + 5, ty + 13);
      }

      if (isSelected) {
        for (const c of this.boxCornersScreen(b)) {
          ctx.fillStyle = "#ffffff";
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          ctx.fillRect(c.x - 4, c.y - 4, 8, 8);
          ctx.strokeRect(c.x - 4, c.y - 4, 8, 8);
        }
      }
    };

    for (const b of this.boxes) {
      drawBox(b, false);
    }
    if (this.draftBox) {
      drawBox(this.draftBox, true);
    }

    for (const poly of this.polygons) {
      if (poly.points.length < 2) continue;
      const cls = this.classes.get(poly.classId);
      const color = cls?.color ?? "#a855f7";
      const isSelected = this.selected?.kind === "polygon" && this.selected.id === poly.id;
      const pts = poly.points.map((pt) => this.normToScreen(pt.x, pt.y));

      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) {
        ctx.lineTo(pts[i].x, pts[i].y);
      }
      ctx.closePath();
      ctx.fillStyle = `${color}2b`;
      ctx.fill();
      ctx.strokeStyle = isSelected ? "#ffffff" : color;
      ctx.lineWidth = isSelected ? 2.2 : 1.7;
      ctx.stroke();

      if (cls) {
        ctx.font = "600 11px Segoe UI, sans-serif";
        const tw = ctx.measureText(cls.name).width + 10;
        ctx.fillStyle = color;
        ctx.fillRect(pts[0].x, Math.max(0, pts[0].y - 18), tw, 18);
        ctx.fillStyle = "#090b0f";
        ctx.textAlign = "left";
        ctx.fillText(cls.name, pts[0].x + 5, Math.max(13, pts[0].y - 5));
      }

      if (isSelected) {
        for (const pt of pts) {
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, 4.5, 0, Math.PI * 2);
          ctx.fillStyle = "#ffffff";
          ctx.fill();
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      }
    }

    if (this.draftPoly.length > 0) {
      const color = this.classes.get(this.activeClassId)?.color ?? "#38bdf8";
      const pts = this.draftPoly.map((pt) => this.normToScreen(pt.x, pt.y));
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) {
        ctx.lineTo(pts[i].x, pts[i].y);
      }
      if (this.pointerInside) {
        ctx.lineTo(this.pointerX, this.pointerY);
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.8;
      ctx.stroke();
      for (let i = 0; i < pts.length; i++) {
        ctx.beginPath();
        ctx.arc(pts[i].x, pts[i].y, i === 0 ? 5.5 : 3.5, 0, Math.PI * 2);
        ctx.fillStyle = i === 0 ? "#ffffff" : color;
        ctx.fill();
      }
    }

    if (this.pointerInside && (this.tool === "box" || this.tool === "polygon")) {
      ctx.save();
      ctx.strokeStyle = "rgba(255,255,255,0.22)";
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(this.pointerX, 0);
      ctx.lineTo(this.pointerX, h);
      ctx.moveTo(0, this.pointerY);
      ctx.lineTo(w, this.pointerY);
      ctx.stroke();
      ctx.restore();
    }
  }
}
