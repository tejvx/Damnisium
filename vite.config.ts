import { defineConfig } from "vite";

export default defineConfig({
  clearScreen: false,
  resolve: {
    conditions: ["onnxruntime-web-use-extern-wasm"]
  },
  server: {
    port: 1420,
    strictPort: true,
    watch: {
      ignored: ["**/src-tauri/**"]
    }
  },
  assetsInclude: ["**/*.wasm"],
  optimizeDeps: {
    exclude: ["onnxruntime-web", "@huggingface/transformers"]
  },
  build: {
    target: "es2022",
    minify: "esbuild",
    outDir: "dist",
    emptyOutDir: true,
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        manualChunks: {
          ort: ["onnxruntime-web/wasm"],
          transformers: ["@huggingface/transformers"]
        }
      }
    }
  }
});
