import { copyFileSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { join, resolve } from "node:path";

const rootDir = resolve(process.cwd());
const sourceDir = join(rootDir, "node_modules", "onnxruntime-web", "dist");
const targetDir = join(rootDir, "public", "ort");

if (!existsSync(sourceDir)) {
  process.exit(0);
}

rmSync(targetDir, { recursive: true, force: true });
mkdirSync(targetDir, { recursive: true });

for (const filename of ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"]) {
  const src = join(sourceDir, filename);
  if (existsSync(src)) {
    copyFileSync(src, join(targetDir, filename));
  }
}
