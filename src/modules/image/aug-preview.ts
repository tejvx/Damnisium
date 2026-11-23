import type { ImageAugConfig } from "../../types";

let cachedPreviewSrc: string | null = null;
let cachedPreviewImg: HTMLImageElement | null = null;
let drawSeq = 0;

function drawPreviewFrame(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  img: HTMLImageElement,
  aug: ImageAugConfig
): void {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#090b0f";
  ctx.fillRect(0, 0, w, h);
  if (img.naturalWidth <= 0 || img.naturalHeight <= 0) return;

  ctx.save();
  ctx.translate(w / 2, h / 2);
  if (aug.rotateDeg > 0) {
    ctx.rotate((aug.rotateDeg * Math.PI) / 180);
  }
  ctx.scale(aug.flipHorizontal ? -1 : 1, aug.flipVertical ? -1 : 1);

  const brightness = 100 + aug.brightnessPct;
  const contrast = 100 + aug.contrastPct;
  ctx.filter = `brightness(${brightness}%) contrast(${contrast}%) blur(${aug.blurPx}px)`;

  const scale = Math.min((w - 12) / img.naturalWidth, (h - 12) / img.naturalHeight);
  const dw = img.naturalWidth * scale;
  const dh = img.naturalHeight * scale;
  ctx.drawImage(img, -dw / 2, -dh / 2, dw, dh);
  ctx.restore();
}

export function renderImageAugPreview(
  canvas: HTMLCanvasElement,
  imageSrc: string | null,
  aug: ImageAugConfig
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const seq = ++drawSeq;
  const w = canvas.width;
  const h = canvas.height;

  if (!imageSrc) {
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#090b0f";
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "#667389";
    ctx.font = "11px Segoe UI, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Select an image", w / 2, h / 2);
    return;
  }

  if (cachedPreviewSrc === imageSrc && cachedPreviewImg?.complete && cachedPreviewImg.naturalWidth > 0) {
    drawPreviewFrame(ctx, w, h, cachedPreviewImg, aug);
    return;
  }

  const img = new Image();
  img.onload = () => {
    cachedPreviewSrc = imageSrc;
    cachedPreviewImg = img;
    if (seq !== drawSeq || !canvas.isConnected) return;
    drawPreviewFrame(ctx, w, h, img, aug);
  };
  img.src = imageSrc;
}
