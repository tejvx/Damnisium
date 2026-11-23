use crate::models::{GgufGenerationResult, LlamaRuntimeStatus};
use futures_util::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use std::fs::{self, File};
use std::io::Read;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use zip::ZipArchive;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

const CREATE_NO_WINDOW: u32 = 0x08000000;

#[derive(Deserialize)]
struct GithubAsset {
    name: String,
    browser_download_url: String,
    size: Option<u64>,
}

#[derive(Deserialize)]
struct GithubRelease {
    tag_name: String,
    draft: bool,
    assets: Vec<GithubAsset>,
}

#[derive(Deserialize, serde::Serialize, Default)]
struct LlamaVersionMeta {
    release_tag: String,
    asset_name: String,
}

#[derive(Deserialize)]
struct ChatChoiceMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
    usage: Option<ChatUsage>,
}

#[derive(Deserialize)]
struct ChatUsage {
    completion_tokens: Option<u32>,
}

struct RunningLlama {
    child: Child,
    port: u16,
    model_path: String,
    mmproj_path: Option<String>,
}

pub struct GgufServerState {
    inner: Mutex<Option<RunningLlama>>,
}

impl Default for GgufServerState {
    fn default() -> Self {
        Self {
            inner: Mutex::new(None),
        }
    }
}

impl Drop for GgufServerState {
    fn drop(&mut self) {
        self.unload();
    }
}

impl GgufServerState {
    pub fn llama_dir(data_dir: &Path) -> PathBuf {
        data_dir.join("runtime").join("llama")
    }

    pub fn llama_exe(data_dir: &Path) -> PathBuf {
        Self::llama_dir(data_dir).join("llama-server.exe")
    }

    pub fn status(&self, data_dir: &Path) -> LlamaRuntimeStatus {
        let dir = Self::llama_dir(data_dir);
        let exe = Self::llama_exe(data_dir);
        let meta_path = dir.join("version.json");
        let release_tag = fs::read(meta_path)
            .ok()
            .and_then(|b| serde_json::from_slice::<LlamaVersionMeta>(&b).ok())
            .map(|m| m.release_tag)
            .filter(|s| !s.is_empty());

        let mut active_model_path = None;
        let mut running = false;
        if let Ok(mut guard) = self.inner.lock() {
            if let Some(proc_info) = guard.as_mut() {
                match proc_info.child.try_wait() {
                    Ok(None) => {
                        running = true;
                        active_model_path = Some(proc_info.model_path.clone());
                    }
                    _ => {
                        *guard = None;
                    }
                }
            }
        }

        LlamaRuntimeStatus {
            installed: exe.is_file(),
            running,
            binary_path: exe.to_string_lossy().to_string(),
            active_model_path,
            release_tag,
            download_size_mb: 24,
        }
    }

    pub fn unload(&self) {
        if let Ok(mut guard) = self.inner.lock() {
            if let Some(mut proc_info) = guard.take() {
                let _ = proc_info.child.kill();
                let _ = proc_info.child.wait();
            }
        }
    }

    fn alloc_free_port() -> Result<u16, String> {
        let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
        let port = listener.local_addr().map_err(|e| e.to_string())?.port();
        Ok(port)
    }

    pub async fn ensure_started(
        &self,
        data_dir: &Path,
        model_path: &Path,
        mmproj_path: Option<&Path>,
    ) -> Result<u16, String> {
        if !model_path.is_file() {
            return Err(format!("GGUF model file not found: {}", model_path.display()));
        }
        let exe = Self::llama_exe(data_dir);
        if !exe.is_file() {
            return Err("llama.cpp runtime is not installed yet.".into());
        }

        let target_model = model_path.to_string_lossy().to_string();
        let target_mmproj = mmproj_path.map(|p| p.to_string_lossy().to_string());

        {
            let mut guard = self
                .inner
                .lock()
                .map_err(|_| "GGUF server mutex poisoned.".to_string())?;
            if let Some(existing) = guard.as_mut() {
                if matches!(existing.child.try_wait(), Ok(None))
                    && existing.model_path == target_model
                    && existing.mmproj_path == target_mmproj
                {
                    return Ok(existing.port);
                }
            }
            if let Some(mut old) = guard.take() {
                let _ = old.child.kill();
                let _ = old.child.wait();
            }
        }

        let port = Self::alloc_free_port()?;
        let mut cmd = Command::new(&exe);
        cmd.arg("-m")
            .arg(model_path)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("-c")
            .arg("2048")
            .arg("-ngl")
            .arg("0")
            .arg("--no-webui")
            .current_dir(Self::llama_dir(data_dir))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());

        if let Some(mm) = mmproj_path {
            if mm.is_file() {
                cmd.arg("--mmproj").arg(mm);
            }
        }

        #[cfg(windows)]
        cmd.creation_flags(CREATE_NO_WINDOW);

        let child = cmd
            .spawn()
            .map_err(|e| format!("Failed launching llama-server.exe: {e}"))?;

        {
            let mut guard = self
                .inner
                .lock()
                .map_err(|_| "GGUF server mutex poisoned.".to_string())?;
            *guard = Some(RunningLlama {
                child,
                port,
                model_path: target_model,
                mmproj_path: target_mmproj,
            });
        }

        let client = Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .map_err(|e| e.to_string())?;
        let health_url = format!("http://127.0.0.1:{port}/health");

        for _ in 0..180 {
            {
                let mut guard = self
                    .inner
                    .lock()
                    .map_err(|_| "GGUF server mutex poisoned.".to_string())?;
                if let Some(proc_info) = guard.as_mut() {
                    if let Ok(Some(status)) = proc_info.child.try_wait() {
                        let mut err_text = String::new();
                        if let Some(mut stderr) = proc_info.child.stderr.take() {
                            let _ = stderr.read_to_string(&mut err_text);
                        }
                        *guard = None;
                        let snippet: String = err_text.chars().rev().take(400).collect();
                        let rev: String = snippet.chars().rev().collect();
                        return Err(format!(
                            "llama-server exited early ({status}): {}",
                            rev.trim()
                        ));
                    }
                }
            }
            if let Ok(resp) = client.get(&health_url).send().await {
                if resp.status().is_success() {
                    return Ok(port);
                }
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }

        self.unload();
        Err("Timed out waiting for llama-server.exe health endpoint.".into())
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn chat(
        &self,
        data_dir: &Path,
        model_path: &Path,
        mmproj_path: Option<&Path>,
        system: &str,
        prompt: &str,
        max_tokens: u32,
        temperature: f64,
    ) -> Result<GgufGenerationResult, String> {
        let port = self.ensure_started(data_dir, model_path, mmproj_path).await?;
        let started = Instant::now();
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(|e| e.to_string())?;

        let mut messages = Vec::new();
        if !system.trim().is_empty() {
            messages.push(serde_json::json!({
                "role": "system",
                "content": system.trim()
            }));
        }
        messages.push(serde_json::json!({
            "role": "user",
            "content": prompt.trim()
        }));

        let payload = serde_json::json!({
            "model": "local",
            "messages": messages,
            "max_tokens": max_tokens.clamp(16, 2048),
            "temperature": temperature.clamp(0.0, 2.0),
            "stream": false
        });

        let resp = client
            .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("GGUF request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("llama-server returned {status}: {text}"));
        }

        let parsed: ChatCompletionResponse = resp.json().await.map_err(|e| e.to_string())?;
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let output = parsed
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or("")
            .trim()
            .to_string();
        let tokens = parsed
            .usage
            .and_then(|u| u.completion_tokens)
            .unwrap_or_else(|| (output.split_whitespace().count() as u32).max(1));
        let secs = (elapsed_ms as f64 / 1000.0).max(0.05);

        Ok(GgufGenerationResult {
            output,
            tokens_per_sec: tokens as f64 / secs,
            duration_ms: elapsed_ms,
        })
    }
}

pub async fn install_llama_runtime(data_dir: &Path) -> Result<LlamaRuntimeStatus, String> {
    let dir = GgufServerState::llama_dir(data_dir);
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let client = Client::builder()
        .user_agent("Damnisium-App")
        .timeout(Duration::from_secs(300))
        .build()
        .map_err(|e| e.to_string())?;

    let releases: Vec<GithubRelease> = client
        .get("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10")
        .send()
        .await
        .map_err(|e| format!("Failed querying llama.cpp releases: {e}"))?
        .error_for_status()
        .map_err(|e| format!("GitHub releases API returned error: {e}"))?
        .json()
        .await
        .map_err(|e| format!("Failed parsing llama.cpp release metadata: {e}"))?;

    let (tag, asset) = releases
        .into_iter()
        .filter(|r| !r.draft)
        .find_map(|r| {
            let tag = r.tag_name;
            let matched = r.assets.into_iter().find(|a| {
                let low = a.name.to_lowercase();
                low.starts_with("llama-")
                    && low.contains("win-cpu-x64")
                    && low.ends_with(".zip")
            });
            matched.map(|a| (tag, a))
        })
        .ok_or_else(|| "No Windows x64 CPU release archive found in latest llama.cpp releases.".to_string())?;

    let zip_path = dir.join(&asset.name);
    let resp = client
        .get(&asset.browser_download_url)
        .send()
        .await
        .map_err(|e| format!("Failed downloading llama.cpp runtime: {e}"))?
        .error_for_status()
        .map_err(|e| format!("llama.cpp download error: {e}"))?;

    let total_expected = resp.content_length();
    let mut downloaded_bytes: u64 = 0;
    let mut out = tokio::fs::File::create(&zip_path)
        .await
        .map_err(|e| e.to_string())?;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let bytes = match chunk {
            Ok(b) => b,
            Err(e) => {
                drop(out);
                let _ = tokio::fs::remove_file(&zip_path).await;
                return Err(format!("Download stream interrupted: {e}"));
            }
        };
        if let Err(e) = tokio::io::AsyncWriteExt::write_all(&mut out, &bytes).await {
            drop(out);
            let _ = tokio::fs::remove_file(&zip_path).await;
            return Err(e.to_string());
        }
        downloaded_bytes += bytes.len() as u64;
    }
    if let Err(e) = tokio::io::AsyncWriteExt::flush(&mut out).await {
        drop(out);
        let _ = tokio::fs::remove_file(&zip_path).await;
        return Err(e.to_string());
    }
    drop(out);

    if downloaded_bytes == 0 {
        let _ = fs::remove_file(&zip_path);
        return Err("Downloaded llama.cpp archive is empty.".into());
    }
    if let Some(exp) = total_expected {
        if downloaded_bytes != exp {
            let _ = fs::remove_file(&zip_path);
            return Err(format!(
                "Incomplete llama.cpp download: got {downloaded_bytes} bytes, expected {exp}."
            ));
        }
    }

    let archive_file = File::open(&zip_path).map_err(|e| e.to_string())?;
    let mut archive = ZipArchive::new(archive_file).map_err(|e| {
        let _ = fs::remove_file(&zip_path);
        format!("Invalid llama.cpp zip archive: {e}")
    })?;
    for i in 0..archive.len() {
        let mut item = archive.by_index(i).map_err(|e| e.to_string())?;
        if item.is_dir() {
            continue;
        }
        let Some(fname) = Path::new(item.name()).file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let low = fname.to_lowercase();
        if low == "llama-server.exe" || low.ends_with(".dll") {
            let dest = dir.join(fname);
            let mut dest_file = File::create(&dest).map_err(|e| e.to_string())?;
            std::io::copy(&mut item, &mut dest_file).map_err(|e| e.to_string())?;
        }
    }
    drop(archive);
    let _ = fs::remove_file(&zip_path);

    let exe = GgufServerState::llama_exe(data_dir);
    let exe_size = fs::metadata(&exe).map(|m| m.len()).unwrap_or(0);
    if !exe.is_file() || exe_size < 1024 {
        return Err("Extracted archive did not contain a valid llama-server.exe.".into());
    }
    let mut mz = [0u8; 2];
    {
        let mut ef = File::open(&exe).map_err(|e| e.to_string())?;
        ef.read_exact(&mut mz).map_err(|e| e.to_string())?;
    }
    if &mz != b"MZ" {
        let _ = fs::remove_file(&exe);
        return Err("Extracted llama-server.exe is not a valid Windows executable.".into());
    }
    let mut probe = Command::new(&exe);
    probe
        .arg("--version")
        .current_dir(&dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(windows)]
    probe.creation_flags(CREATE_NO_WINDOW);
    let probe_ok = probe.status().map(|s| s.success()).unwrap_or(false);
    if !probe_ok {
        return Err("llama-server.exe failed execution verification check (--version).".into());
    }

    let meta = LlamaVersionMeta {
        release_tag: tag.clone(),
        asset_name: asset.name,
    };
    let meta_bytes = serde_json::to_vec_pretty(&meta).map_err(|e| e.to_string())?;
    fs::write(dir.join("version.json"), meta_bytes).map_err(|e| e.to_string())?;

    let approx_mb = asset
        .size
        .map(|s| ((s as f64) / 1_048_576.0).round() as u32)
        .unwrap_or(24);

    Ok(LlamaRuntimeStatus {
        installed: true,
        running: false,
        binary_path: exe.to_string_lossy().to_string(),
        active_model_path: None,
        release_tag: Some(tag),
        download_size_mb: approx_mb,
    })
}
