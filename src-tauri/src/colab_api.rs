use crate::models::ColabRuntimeSpec;
use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use chrono::DateTime;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use uuid::Uuid;

pub const COLAB_API_BASE: &str = "https://colaboratory.googleapis.com";
pub const COLAB_MCP_URL: &str = "https://colaboratory.googleapis.com/mcp";
const JUPYTER_CHUNK_BYTES: usize = 8 * 1024 * 1024;
const JUPYTER_DIRECT_LIMIT: u64 = 8 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ColabConnectionInfo {
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub token: String,
    #[serde(default)]
    pub expire_time: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ColabRuntime {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub runtime_spec: Option<ColabRuntimeSpec>,
    #[serde(default)]
    pub state: Option<String>,
    #[serde(default)]
    pub connection_info: Option<ColabConnectionInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ColabOperation {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub done: Option<bool>,
    #[serde(default)]
    pub error: Option<Value>,
    #[serde(default)]
    pub response: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct ListRuntimesResponse {
    #[serde(default)]
    runtimes: Vec<ColabRuntime>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JupyterContentItem {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub path: String,
    #[serde(rename = "type", default)]
    pub item_type: String,
    #[serde(default)]
    pub content: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct McpExecResult {
    #[serde(default)]
    pub stdout: String,
    #[serde(default)]
    pub stderr: String,
    #[serde(default)]
    pub result: Option<Value>,
    #[serde(default)]
    pub execution_error: Option<Value>,
    #[serde(default)]
    pub output_truncated: bool,
}

#[derive(Debug, Clone)]
pub enum CreateRuntimeStart {
    Ready(ColabRuntime),
    Operation(ColabOperation),
}

pub struct ColabClient {
    http: Client,
    mcp_http: Client,
    access_token: String,
    gcp_project_id: String,
    api_base: String,
    mcp_url: String,
    mcp_session_id: Option<String>,
    mcp_initialized: bool,
    rpc_seq: u64,
}

impl ColabClient {
    pub fn new(access_token: String, gcp_project_id: String) -> Result<Self, String> {
        Self::with_endpoints(
            access_token,
            gcp_project_id,
            COLAB_API_BASE.to_string(),
            COLAB_MCP_URL.to_string(),
        )
    }

    pub fn with_endpoints(
        access_token: String,
        gcp_project_id: String,
        api_base: String,
        mcp_url: String,
    ) -> Result<Self, String> {
        let http = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(|e| format!("Failed creating Colab HTTP client: {e}"))?;
        let mcp_http = Client::builder()
            .timeout(Duration::from_secs(620))
            .build()
            .map_err(|e| format!("Failed creating Colab MCP client: {e}"))?;
        Ok(Self {
            http,
            mcp_http,
            access_token,
            gcp_project_id,
            api_base: api_base.trim_end_matches('/').to_string(),
            mcp_url,
            mcp_session_id: None,
            mcp_initialized: false,
            rpc_seq: 1,
        })
    }

    pub fn set_access_token(&mut self, access_token: String) {
        self.access_token = access_token;
    }

    fn apply_goog_headers(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let r = req.bearer_auth(&self.access_token);
        if self.gcp_project_id.trim().is_empty() {
            r
        } else {
            r.header("x-goog-user-project", self.gcp_project_id.trim())
        }
    }

    pub async fn list_runtime_specs(&self) -> Result<Vec<ColabRuntimeSpec>, String> {
        let url = format!("{}/v1beta/runtimespecs", self.api_base);
        let resp = self
            .apply_goog_headers(self.http.get(&url))
            .send()
            .await
            .map_err(|e| format!("Colab runtimespecs request failed: {e}"))?;
        let val: Value = decode_colab_json(resp, "Colab runtimespecs error").await?;
        Ok(parse_runtime_specs(&val))
    }

    pub async fn list_runtimes(&self) -> Result<Vec<ColabRuntime>, String> {
        let url = format!("{}/v1beta/runtimes", self.api_base);
        let resp = self
            .apply_goog_headers(self.http.get(&url))
            .send()
            .await
            .map_err(|e| format!("Colab runtimes lookup failed: {e}"))?;
        let list: ListRuntimesResponse = decode_colab_json(resp, "Colab runtimes error").await?;
        Ok(list.runtimes)
    }

    pub async fn get_runtime(&self, runtime_name: &str) -> Result<ColabRuntime, String> {
        let norm = normalize_runtime_name(runtime_name);
        let url = format!("{}/v1beta/{norm}", self.api_base);
        let resp = self
            .apply_goog_headers(self.http.get(&url))
            .send()
            .await
            .map_err(|e| format!("Colab get_runtime failed: {e}"))?;
        decode_colab_json(resp, "Colab get_runtime error").await
    }

    pub async fn start_create_runtime(
        &self,
        spec: &ColabRuntimeSpec,
    ) -> Result<CreateRuntimeStart, String> {
        let url = format!("{}/v1beta/runtimes", self.api_base);
        let payload = json!({
            "runtimeSpec": {
                "variant": spec.variant,
                "accelerator": spec.accelerator,
                "shape": if spec.shape.trim().is_empty() { "SHAPE_STANDARD" } else { &spec.shape }
            }
        });
        let resp = self
            .apply_goog_headers(self.http.post(&url))
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("Colab runtime allocation failed: {e}"))?;
        let val: Value = decode_colab_json(resp, "Colab create runtime error").await?;

        if val.get("connectionInfo").is_some() && val.get("name").is_some() {
            let rt: ColabRuntime =
                serde_json::from_value(val).map_err(|e| format!("Invalid ColabRuntime JSON: {e}"))?;
            return Ok(CreateRuntimeStart::Ready(rt));
        }

        let op: ColabOperation = serde_json::from_value(val)
            .map_err(|e| format!("Invalid Colab operation response: {e}"))?;
        Ok(CreateRuntimeStart::Operation(op))
    }

    pub async fn poll_operation(&self, op_name: &str) -> Result<ColabOperation, String> {
        let op_path = op_name.trim_start_matches('/');
        let op_url = format!("{}/v1/{op_path}", self.api_base);
        let poll_resp = self
            .apply_goog_headers(self.http.get(&op_url))
            .send()
            .await
            .map_err(|e| format!("Colab operation poll failed: {e}"))?;
        decode_colab_json(poll_resp, "Colab operation poll error").await
    }

    pub async fn finalize_runtime_from_op(&self, op: ColabOperation) -> Result<ColabRuntime, String> {
        if let Some(err_val) = op.error {
            let formatted = format_google_error_value("Colab allocation operation failed", 500, &err_val);
            return Err(formatted);
        }

        let rt_val = op
            .response
            .ok_or_else(|| "Colab operation finished without runtime payload.".to_string())?;
        let mut runtime: ColabRuntime = serde_json::from_value(rt_val)
            .map_err(|e| format!("Invalid runtime inside Colab operation: {e}"))?;
        if runtime
            .connection_info
            .as_ref()
            .map(|c| c.url.is_empty() || c.token.is_empty())
            .unwrap_or(true)
            && !runtime.name.is_empty()
        {
            runtime = self.get_runtime(&runtime.name).await?;
        }
        Ok(runtime)
    }

    pub async fn find_reusable_runtime(&self, target_spec: &ColabRuntimeSpec) -> Option<ColabRuntime> {
        let existing = self.list_runtimes().await.ok()?;
        for candidate in existing {
            if candidate.name.trim().is_empty() {
                continue;
            }
            let st_up = candidate
                .state
                .as_deref()
                .unwrap_or("")
                .trim()
                .to_ascii_uppercase();
            if matches!(st_up.as_str(), "DELETING" | "TERMINATED" | "FAILED") {
                continue;
            }
            let matches = candidate
                .runtime_spec
                .as_ref()
                .map(|rs| runtime_spec_matches(rs, target_spec))
                .unwrap_or(true);
            if !matches {
                continue;
            }
            if let Ok(refreshed) = self.get_runtime(&candidate.name).await {
                let ref_matches = refreshed
                    .runtime_spec
                    .as_ref()
                    .map(|rs| runtime_spec_matches(rs, target_spec))
                    .unwrap_or(true);
                if !ref_matches {
                    continue;
                }
                let ref_st = refreshed
                    .state
                    .as_deref()
                    .unwrap_or("")
                    .trim()
                    .to_ascii_uppercase();
                if matches!(ref_st.as_str(), "DELETING" | "TERMINATED" | "FAILED") {
                    continue;
                }
                if refreshed
                    .connection_info
                    .as_ref()
                    .map(|ci| !ci.url.is_empty() && !ci.token.is_empty())
                    .unwrap_or(false)
                {
                    return Some(refreshed);
                }
            }
        }
        None
    }

    #[allow(dead_code)]
    pub async fn create_runtime_and_wait(
        &self,
        spec: &ColabRuntimeSpec,
        cancelled: &AtomicBool,
    ) -> Result<ColabRuntime, String> {
        if cancelled.load(Ordering::Relaxed) {
            return Err("Training run was cancelled.".into());
        }
        let mut op = match self.start_create_runtime(spec).await? {
            CreateRuntimeStart::Ready(rt) => return Ok(rt),
            CreateRuntimeStart::Operation(op) => op,
        };

        let started = std::time::Instant::now();
        let mut delay_ms = 1500u64;
        while !op.done.unwrap_or(false) {
            if cancelled.load(Ordering::Relaxed) {
                return Err("Training run was cancelled during runtime allocation.".into());
            }
            if started.elapsed() > Duration::from_secs(300) {
                return Err("Timed out waiting for Colab GPU runtime allocation.".into());
            }
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            delay_ms = (delay_ms + 500).min(4000);
            op = self.poll_operation(&op.name).await?;
        }

        self.finalize_runtime_from_op(op).await
    }

    #[allow(dead_code)]
    pub async fn pick_or_allocate_runtime(
        &self,
        preferred_accelerator: Option<&str>,
        preferred_variant: Option<&str>,
        cancelled: &AtomicBool,
    ) -> Result<(ColabRuntime, bool), String> {
        let specs = self.list_runtime_specs().await?;
        let target_spec = select_best_runtime_spec(&specs, preferred_accelerator, preferred_variant)
            .unwrap_or_else(|| ColabRuntimeSpec {
                variant: preferred_variant
                    .filter(|s| !s.trim().is_empty())
                    .unwrap_or("VARIANT_GPU")
                    .to_string(),
                accelerator: preferred_accelerator
                    .filter(|s| !s.trim().is_empty())
                    .unwrap_or("T4")
                    .to_string(),
                shape: "SHAPE_STANDARD".to_string(),
                eligible: true,
            });

        if let Some(rt) = self.find_reusable_runtime(&target_spec).await {
            return Ok((rt, false));
        }

        let created = self.create_runtime_and_wait(&target_spec, cancelled).await?;
        Ok((created, true))
    }

    pub async fn ensure_runtime_token_fresh(&self, runtime: &mut ColabRuntime) -> Result<bool, String> {
        let needs_refresh = match runtime.connection_info.as_ref() {
            None => true,
            Some(ci) if ci.token.is_empty() || ci.url.is_empty() => true,
            Some(ci) => {
                if let Ok(exp) = DateTime::parse_from_rfc3339(ci.expire_time.trim()) {
                    let now = chrono::Utc::now();
                    (exp.with_timezone(&chrono::Utc) - now).num_seconds() <= 600
                } else {
                    false
                }
            }
        };
        if needs_refresh && !runtime.name.trim().is_empty() {
            *runtime = self.get_runtime(&runtime.name).await?;
            return Ok(true);
        }
        Ok(false)
    }

    pub async fn delete_runtime(&self, runtime_name: &str) -> Result<(), String> {
        if runtime_name.trim().is_empty() {
            return Ok(());
        }
        let norm = normalize_runtime_name(runtime_name);
        let url = format!("{}/v1beta/{norm}", self.api_base);
        let resp = self
            .apply_goog_headers(self.http.delete(&url))
            .send()
            .await
            .map_err(|e| format!("Colab delete_runtime request failed: {e}"))?;
        let status = resp.status();
        if status.is_success() || status.as_u16() == 404 {
            return Ok(());
        }
        let body = resp.text().await.unwrap_or_default();
        Err(format_colab_error("Colab delete_runtime error", status.as_u16(), &body))
    }

    pub async fn resolve_remote_work_base(&self, runtime: &ColabRuntime) -> String {
        if let Ok(ci) = runtime_conn(runtime) {
            let url = format!("{}/api/contents/content", ci.url.trim_end_matches('/'));
            if let Ok(resp) = self
                .http
                .get(&url)
                .header("X-Colab-Runtime-Proxy-Token", &ci.token)
                .query(&[("content", "0")])
                .send()
                .await
            {
                if resp.status().is_success() {
                    return "content/damnisium_work".to_string();
                }
            }
        }
        "damnisium_work".to_string()
    }

    pub async fn ensure_remote_dir(&self, runtime: &ColabRuntime, remote_dir: &str) -> Result<(), String> {
        let ci = runtime_conn(runtime)?;
        let clean = remote_dir.trim_matches('/');
        if clean.is_empty() {
            return Ok(());
        }
        let mut built = String::new();
        for seg in clean.split('/').filter(|s| !s.is_empty()) {
            if !built.is_empty() {
                built.push('/');
            }
            built.push_str(seg);
            let encoded = encode_path_segments(&built);
            let url = format!("{}/api/contents/{encoded}", ci.url.trim_end_matches('/'));
            let get_resp = self
                .http
                .get(&url)
                .header("X-Colab-Runtime-Proxy-Token", &ci.token)
                .query(&[("content", "0")])
                .send()
                .await
                .map_err(|e| format!("Jupyter directory probe failed: {e}"))?;
            if get_resp.status().is_success() {
                continue;
            }
            let put_resp = self
                .http
                .put(&url)
                .header("X-Colab-Runtime-Proxy-Token", &ci.token)
                .json(&json!({ "type": "directory" }))
                .send()
                .await
                .map_err(|e| format!("Jupyter directory creation failed: {e}"))?;
            let status = put_resp.status();
            if !status.is_success() && status.as_u16() != 409 {
                let body = put_resp.text().await.unwrap_or_default();
                return Err(format!("Jupyter mkdir {built} HTTP {status}: {body}"));
            }
        }
        Ok(())
    }

    pub async fn upload_bytes(
        &self,
        runtime: &ColabRuntime,
        bytes: &[u8],
        remote_path: &str,
    ) -> Result<(), String> {
        let ci = runtime_conn(runtime)?;
        let clean = remote_path.trim_start_matches('/');
        if let Some((parent, _)) = clean.rsplit_once('/') {
            self.ensure_remote_dir(runtime, parent).await?;
        }
        let filename = clean.rsplit('/').next().unwrap_or(clean);
        let url = format!(
            "{}/api/contents/{}",
            ci.url.trim_end_matches('/'),
            encode_path_segments(clean)
        );
        let payload = json!({
            "name": filename,
            "path": clean,
            "type": "file",
            "format": "base64",
            "content": B64.encode(bytes)
        });
        let resp = self
            .http
            .put(&url)
            .header("X-Colab-Runtime-Proxy-Token", &ci.token)
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("Jupyter upload failed for {clean}: {e}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Jupyter upload {clean} HTTP {status}: {body}"));
        }
        Ok(())
    }

    pub async fn upload_file_streaming<F>(
        &self,
        runtime: &ColabRuntime,
        local_path: &Path,
        remote_path: &str,
        mut on_progress: F,
    ) -> Result<(), String>
    where
        F: FnMut(u64, u64),
    {
        let ci = runtime_conn(runtime)?;
        let clean = remote_path.trim_start_matches('/');
        if let Some((parent, _)) = clean.rsplit_once('/') {
            self.ensure_remote_dir(runtime, parent).await?;
        }
        let filename = clean.rsplit('/').next().unwrap_or(clean);
        let mut file = File::open(local_path)
            .map_err(|e| format!("Cannot open {}: {e}", local_path.display()))?;
        let total_size = file.metadata().map(|m| m.len()).unwrap_or(0);
        let url = format!(
            "{}/api/contents/{}",
            ci.url.trim_end_matches('/'),
            encode_path_segments(clean)
        );

        if total_size <= JUPYTER_DIRECT_LIMIT {
            let mut buf = Vec::with_capacity(total_size as usize);
            file.read_to_end(&mut buf).map_err(|e| e.to_string())?;
            let payload = json!({
                "name": filename,
                "path": clean,
                "type": "file",
                "format": "base64",
                "content": B64.encode(&buf)
            });
            let resp = self
                .http
                .put(&url)
                .header("X-Colab-Runtime-Proxy-Token", &ci.token)
                .json(&payload)
                .send()
                .await
                .map_err(|e| format!("Jupyter upload failed for {clean}: {e}"))?;
            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(format!("Jupyter upload {clean} HTTP {status}: {body}"));
            }
            on_progress(total_size, total_size);
            return Ok(());
        }

        let mut buf = vec![0u8; JUPYTER_CHUNK_BYTES];
        let mut uploaded = 0u64;
        let mut chunk_idx = 1i64;
        loop {
            let mut filled = 0usize;
            while filled < buf.len() {
                let n = file.read(&mut buf[filled..]).map_err(|e| e.to_string())?;
                if n == 0 {
                    break;
                }
                filled += n;
            }
            if filled == 0 {
                break;
            }
            uploaded += filled as u64;
            let is_last = uploaded >= total_size || filled < buf.len();
            let chunk_marker = if is_last { -1 } else { chunk_idx };
            let payload = json!({
                "name": filename,
                "path": clean,
                "type": "file",
                "format": "base64",
                "chunk": chunk_marker,
                "content": B64.encode(&buf[..filled])
            });
            let resp = self
                .http
                .put(&url)
                .header("X-Colab-Runtime-Proxy-Token", &ci.token)
                .json(&payload)
                .send()
                .await
                .map_err(|e| format!("Jupyter chunked upload failed for {clean}: {e}"))?;
            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(format!(
                    "Jupyter chunked upload {clean} (chunk {chunk_marker}) HTTP {status}: {body}"
                ));
            }
            on_progress(uploaded, total_size);
            if is_last {
                break;
            }
            chunk_idx += 1;
        }
        Ok(())
    }

    pub async fn list_remote_directory(
        &self,
        runtime: &ColabRuntime,
        remote_dir: &str,
    ) -> Result<Vec<JupyterContentItem>, String> {
        let ci = runtime_conn(runtime)?;
        let clean = remote_dir.trim_matches('/');
        let url = format!(
            "{}/api/contents/{}",
            ci.url.trim_end_matches('/'),
            encode_path_segments(clean)
        );
        let resp = self
            .http
            .get(&url)
            .header("X-Colab-Runtime-Proxy-Token", &ci.token)
            .query(&[("content", "1")])
            .send()
            .await
            .map_err(|e| format!("Jupyter list {clean} failed: {e}"))?;
        if resp.status().as_u16() == 404 {
            return Ok(Vec::new());
        }
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Jupyter list {clean} HTTP {status}: {body}"));
        }
        let item: JupyterContentItem = resp
            .json()
            .await
            .map_err(|e| format!("Invalid Jupyter directory JSON: {e}"))?;
        let Some(content_val) = item.content else {
            return Ok(Vec::new());
        };
        serde_json::from_value::<Vec<JupyterContentItem>>(content_val)
            .map_err(|e| format!("Invalid Jupyter directory content array: {e}"))
    }

    pub async fn download_file_streaming(
        &self,
        runtime: &ColabRuntime,
        remote_path: &str,
        dest_path: &Path,
    ) -> Result<u64, String> {
        let ci = runtime_conn(runtime)?;
        let clean = remote_path.trim_start_matches('/');
        let encoded = encode_path_segments(clean);
        let url = format!("{}/files/{}", ci.url.trim_end_matches('/'), encoded);
        if let Some(parent) = dest_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let mut part_os = dest_path.as_os_str().to_os_string();
        part_os.push(".part");
        let part_path = std::path::PathBuf::from(part_os);
        let resp = self
            .http
            .get(&url)
            .header("X-Colab-Runtime-Proxy-Token", &ci.token)
            .send()
            .await
            .map_err(|e| format!("Jupyter file download failed for {clean}: {e}"))?;
        let status = resp.status();
        if status.as_u16() == 403 || status.as_u16() == 404 {
            let api_url = format!("{}/api/contents/{}", ci.url.trim_end_matches('/'), encoded);
            let api_resp = self
                .http
                .get(&api_url)
                .header("X-Colab-Runtime-Proxy-Token", &ci.token)
                .query(&[("content", "1"), ("format", "base64")])
                .send()
                .await
                .map_err(|e| format!("Jupyter API download failed for {clean}: {e}"))?;
            if !api_resp.status().is_success() {
                let st = api_resp.status();
                let b = api_resp.text().await.unwrap_or_default();
                return Err(format!("Jupyter download {clean} HTTP {st}: {b}"));
            }
            let val: Value = api_resp
                .json()
                .await
                .map_err(|e| format!("Invalid Jupyter file content JSON for {clean}: {e}"))?;
            let b64_raw = val
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("Jupyter API response missing content for {clean}"))?;
            let cleaned: String = b64_raw.chars().filter(|c| !c.is_ascii_whitespace()).collect();
            let decoded = B64
                .decode(cleaned.as_bytes())
                .map_err(|e| format!("Invalid base64 payload for {clean}: {e}"))?;
            if decoded.is_empty() {
                return Err(format!("Downloaded remote file {clean} is empty."));
            }
            std::fs::write(&part_path, &decoded).map_err(|e| e.to_string())?;
            let total = decoded.len() as u64;
            let disk_len = std::fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0);
            if disk_len != total {
                let _ = std::fs::remove_file(&part_path);
                return Err(format!(
                    "Disk verification mismatch for {clean}: wrote {total} bytes, found {disk_len}."
                ));
            }
            if dest_path.exists() {
                let _ = std::fs::remove_file(dest_path);
            }
            std::fs::rename(&part_path, dest_path).map_err(|e| e.to_string())?;
            return Ok(total);
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Jupyter download {clean} HTTP {status}: {body}"));
        }
        let mut out = File::create(&part_path).map_err(|e| e.to_string())?;
        let mut stream = resp.bytes_stream();
        let mut total = 0u64;
        while let Some(chunk_res) = stream.next().await {
            let chunk = match chunk_res {
                Ok(c) => c,
                Err(e) => {
                    drop(out);
                    let _ = std::fs::remove_file(&part_path);
                    return Err(format!("Jupyter download stream error for {clean}: {e}"));
                }
            };
            total += chunk.len() as u64;
            if let Err(e) = out.write_all(&chunk) {
                drop(out);
                let _ = std::fs::remove_file(&part_path);
                return Err(e.to_string());
            }
        }
        if let Err(e) = out.flush() {
            drop(out);
            let _ = std::fs::remove_file(&part_path);
            return Err(e.to_string());
        }
        drop(out);
        if total == 0 {
            let _ = std::fs::remove_file(&part_path);
            return Err(format!("Downloaded remote file {clean} is empty."));
        }
        let disk_len = std::fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0);
        if disk_len != total {
            let _ = std::fs::remove_file(&part_path);
            return Err(format!(
                "Disk verification mismatch for {clean}: streamed {total} bytes, found {disk_len}."
            ));
        }
        if dest_path.exists() {
            let _ = std::fs::remove_file(dest_path);
        }
        std::fs::rename(&part_path, dest_path).map_err(|e| e.to_string())?;
        Ok(total)
    }

    async fn ensure_mcp_initialized(&mut self) -> Result<(), String> {
        if self.mcp_initialized {
            return Ok(());
        }
        let req_id = self.next_rpc_id();
        let init_body = json!({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "Damnisium",
                    "version": "1.0.0"
                }
            }
        });
        let resp = self
            .apply_goog_headers(self.mcp_http.post(&self.mcp_url))
            .header("Accept", "application/json, text/event-stream")
            .json(&init_body)
            .send()
            .await
            .map_err(|e| format!("Colab MCP initialize failed: {e}"))?;

        if let Some(sid) = resp
            .headers()
            .get("mcp-session-id")
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            self.mcp_session_id = Some(sid.to_string());
        }
        let status = resp.status();
        if status.is_success() {
            let ctype = resp
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_string();
            let body = resp.text().await.unwrap_or_default();
            let _ = parse_mcp_jsonrpc_result(status.as_u16(), &ctype, &body, req_id)?;
            self.mcp_initialized = true;
            let notif_body = json!({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            });
            let mut notif_req = self
                .apply_goog_headers(self.mcp_http.post(&self.mcp_url))
                .header("Accept", "application/json, text/event-stream");
            if let Some(ref sid) = self.mcp_session_id {
                notif_req = notif_req.header("Mcp-Session-Id", sid);
            }
            let _ = notif_req.json(&notif_body).send().await;
            return Ok(());
        }
        if status.as_u16() == 400 || status.as_u16() == 405 {
            self.mcp_initialized = true;
            return Ok(());
        }
        let body = resp.text().await.unwrap_or_default();
        Err(format_colab_error(
            "Colab MCP initialize error",
            status.as_u16(),
            &body,
        ))
    }

    fn next_rpc_id(&mut self) -> u64 {
        let id = self.rpc_seq;
        self.rpc_seq += 1;
        id
    }

    async fn call_mcp_tool(&mut self, tool_name: &str, arguments: Value) -> Result<Value, String> {
        self.call_mcp_tool_streaming(tool_name, arguments, |_| {}).await
    }

    async fn call_mcp_tool_streaming<F>(
        &mut self,
        tool_name: &str,
        arguments: Value,
        mut on_progress: F,
    ) -> Result<Value, String>
    where
        F: FnMut(&str),
    {
        self.ensure_mcp_initialized().await?;
        let req_id = self.next_rpc_id();
        let rpc_body = json!({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        });
        let mut req = self
            .apply_goog_headers(self.mcp_http.post(&self.mcp_url))
            .header("Accept", "application/json, text/event-stream");
        if let Some(ref sid) = self.mcp_session_id {
            req = req.header("Mcp-Session-Id", sid);
        }
        let resp = req
            .json(&rpc_body)
            .send()
            .await
            .map_err(|e| format!("Colab MCP {tool_name} network error: {e}"))?;
        if let Some(sid) = resp
            .headers()
            .get("mcp-session-id")
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            self.mcp_session_id = Some(sid.to_string());
        }
        let status = resp.status();
        let ctype = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            if status.as_u16() == 404 {
                self.mcp_initialized = false;
                self.mcp_session_id = None;
            }
            return Err(format_colab_error(
                &format!("Colab MCP {tool_name} error"),
                status.as_u16(),
                &body,
            ));
        }

        if ctype.to_ascii_lowercase().contains("text/event-stream") {
            let mut rpc_messages: Vec<Value> = Vec::new();
            let mut line_buf: Vec<u8> = Vec::new();
            let mut stream = resp.bytes_stream();
            while let Some(chunk_res) = stream.next().await {
                let chunk = chunk_res
                    .map_err(|e| format!("Colab MCP {tool_name} stream error: {e}"))?;
                line_buf.extend_from_slice(&chunk);
                while let Some(pos) = line_buf.iter().position(|&b| b == b'\n') {
                    let raw_line: Vec<u8> = line_buf.drain(..=pos).collect();
                    let line_str = String::from_utf8_lossy(&raw_line);
                    process_mcp_sse_line(&line_str, &mut rpc_messages, &mut on_progress);
                }
            }
            if !line_buf.is_empty() {
                let tail_str = String::from_utf8_lossy(&line_buf);
                process_mcp_sse_line(&tail_str, &mut rpc_messages, &mut on_progress);
            }
            return select_mcp_jsonrpc_result(&rpc_messages, req_id);
        }

        let body = resp.text().await.unwrap_or_default();
        parse_mcp_jsonrpc_result(status.as_u16(), &ctype, &body, req_id)
    }

    pub async fn list_remote_files_recursive(
        &self,
        runtime: &ColabRuntime,
        remote_dir: &str,
    ) -> Result<Vec<JupyterContentItem>, String> {
        let mut out = Vec::new();
        let mut stack = vec![remote_dir.trim_matches('/').to_string()];
        while let Some(dir) = stack.pop() {
            let entries = self.list_remote_directory(runtime, &dir).await?;
            for item in entries {
                if item.name.starts_with('.') || item.name == ".ipynb_checkpoints" {
                    continue;
                }
                if item.item_type == "directory" {
                    let sub = if item.path.trim().is_empty() {
                        format!("{dir}/{}", item.name)
                    } else {
                        item.path.clone()
                    };
                    stack.push(sub);
                } else {
                    out.push(item);
                }
            }
        }
        Ok(out)
    }

    pub async fn create_session(
        &mut self,
        runtime_name: &str,
        session_id: &str,
    ) -> Result<String, String> {
        let parent = normalize_runtime_name(runtime_name);
        let clean_sid = sanitize_mcp_session_id(session_id);
        let result = self
            .call_mcp_tool(
                "create_session",
                json!({
                    "parent": parent,
                    "sessionId": clean_sid
                }),
            )
            .await?;
        let text_val = extract_mcp_tool_payload(&result)?;
        let session_name = text_val
            .get("name")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToString::to_string)
            .unwrap_or_else(|| format!("{parent}/sessions/{clean_sid}"));
        Ok(session_name)
    }

    #[allow(dead_code)]
    pub async fn execute_code_with_id(
        &mut self,
        session_name: &str,
        code: &str,
        execution_id: &str,
    ) -> Result<McpExecResult, String> {
        self.execute_code_with_id_streaming(session_name, code, execution_id, |_| {})
            .await
    }

    pub async fn execute_code_with_id_streaming<F>(
        &mut self,
        session_name: &str,
        code: &str,
        execution_id: &str,
        on_progress: F,
    ) -> Result<McpExecResult, String>
    where
        F: FnMut(&str),
    {
        let result = self
            .call_mcp_tool_streaming(
                "execute_code",
                json!({
                    "session": session_name,
                    "code": code,
                    "executionId": execution_id
                }),
                on_progress,
            )
            .await?;
        let payload = extract_mcp_tool_payload(&result)?;
        serde_json::from_value::<McpExecResult>(payload)
            .map_err(|e| format!("Invalid MCP execute_code payload: {e}"))
    }

    #[allow(dead_code)]
    pub async fn execute_code_resumable(
        &mut self,
        session_name: &str,
        code: &str,
        execution_id: Option<&str>,
        cancelled: &AtomicBool,
    ) -> Result<(String, McpExecResult), String> {
        self.execute_code_resumable_streaming(session_name, code, execution_id, cancelled, |_| {})
            .await
    }

    pub async fn execute_code_resumable_streaming<F>(
        &mut self,
        session_name: &str,
        code: &str,
        execution_id: Option<&str>,
        cancelled: &AtomicBool,
        mut on_progress: F,
    ) -> Result<(String, McpExecResult), String>
    where
        F: FnMut(&str),
    {
        let exec_id = execution_id
            .filter(|s| !s.trim().is_empty())
            .map(ToString::to_string)
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let mut backoff_secs = 2u64;
        for attempt in 0..8 {
            if cancelled.load(Ordering::Relaxed) {
                return Err("Training run cancelled.".into());
            }
            match self
                .execute_code_with_id_streaming(session_name, code, &exec_id, &mut on_progress)
                .await
            {
                Ok(res) => return Ok((exec_id, res)),
                Err(err) => {
                    if attempt < 7 && is_transient_mcp_error(&err) {
                        tokio::time::sleep(Duration::from_secs(backoff_secs)).await;
                        backoff_secs = (backoff_secs * 2).min(20);
                        continue;
                    }
                    return Err(err);
                }
            }
        }
        Err("Colab MCP execute_code exhausted transient retries.".into())
    }

    pub async fn delete_session(&mut self, session_name: &str) -> Result<(), String> {
        if session_name.trim().is_empty() {
            return Ok(());
        }
        let _ = self
            .call_mcp_tool(
                "delete_session",
                json!({
                    "name": session_name
                }),
            )
            .await?;
        Ok(())
    }
}

pub fn sanitize_mcp_session_id(raw: &str) -> String {
    let mut cleaned = String::with_capacity(raw.len());
    let mut prev_dash = false;
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() {
            cleaned.push(ch.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash {
            cleaned.push('-');
            prev_dash = true;
        }
    }
    let trimmed = cleaned.trim_matches('-').to_string();
    let mut prefixed = if trimmed.is_empty() {
        "damnisium-session".to_string()
    } else if !trimmed.as_bytes()[0].is_ascii_lowercase() {
        format!("s-{trimmed}")
    } else {
        trimmed
    };
    if prefixed.len() > 63 {
        prefixed.truncate(63);
        prefixed = prefixed.trim_end_matches('-').to_string();
    }
    if prefixed.is_empty() {
        "damnisium-session".to_string()
    } else {
        prefixed
    }
}

pub fn strip_ansi_codes(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            if matches!(chars.peek(), Some('[')) {
                chars.next();
                for c in chars.by_ref() {
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
            continue;
        }
        out.push(ch);
    }
    out
}

pub fn format_mcp_execution_error(err_val: &Value) -> String {
    let ename = strip_ansi_codes(
        err_val
            .get("name")
            .or_else(|| err_val.get("ename"))
            .and_then(|v| v.as_str())
            .unwrap_or("PythonError"),
    );
    let evalue_raw = strip_ansi_codes(
        err_val
            .get("value")
            .or_else(|| err_val.get("evalue"))
            .and_then(|v| v.as_str())
            .unwrap_or(""),
    );
    let first_line = evalue_raw
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("")
        .to_string();

    let header = if first_line.is_empty() {
        ename.trim().to_string()
    } else if first_line.starts_with(ename.trim())
        || first_line.starts_with("KeyError:")
        || first_line.starts_with("ValueError:")
        || first_line.starts_with("RuntimeError:")
        || first_line.starts_with("FileNotFoundError:")
        || first_line.starts_with("OSError:")
    {
        first_line
    } else {
        format!("{}: {first_line}", ename.trim())
    };

    let tb_lines: Vec<String> = err_val
        .get("traceback")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str())
                .map(strip_ansi_codes)
                .flat_map(|s| {
                    s.lines()
                        .map(|l| l.trim_end().to_string())
                        .collect::<Vec<_>>()
                })
                .filter(|l| {
                    let t = l.trim();
                    !t.is_empty()
                        && !t.chars().all(|c| c == '-')
                        && t != header
                        && !t.contains("Traceback (most recent call last)")
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let tail_len = tb_lines.len().min(4);
    if tail_len == 0 || evalue_raw.len() > 80 {
        header
    } else {
        let tail = tb_lines[tb_lines.len() - tail_len..].join("\n");
        format!("{header}\n{tail}")
    }
}

pub fn is_auth_expired_error(err: &str) -> bool {
    let low = err.to_ascii_lowercase();
    low.contains("http 401")
        || low.contains("unauthenticated")
        || low.contains("invalid authentication credentials")
}

pub fn is_transient_allocation_error(err: &str) -> bool {
    let low = err.to_ascii_lowercase();
    low.contains("deadline_exceeded")
        || low.contains("context deadline exceeded")
        || low.contains("unavailable")
        || low.contains("internal")
        || low.contains("timed out")
        || low.contains("timeout")
        || low.contains("http 429")
        || low.contains("http 500")
        || low.contains("http 502")
        || low.contains("http 503")
        || low.contains("http 504")
}

fn is_transient_mcp_error(err: &str) -> bool {
    let low = err.to_ascii_lowercase();
    low.contains("deadline_exceeded")
        || low.contains("unavailable")
        || low.contains("timed out")
        || low.contains("timeout")
        || low.contains("connection reset")
        || low.contains("unexpected eof")
        || low.contains("empty mcp response")
        || low.contains("broken pipe")
        || low.contains("aborted")
        || low.contains("internal")
        || low.contains("http 429")
        || low.contains("http 500")
        || low.contains("http 502")
        || low.contains("http 503")
        || low.contains("http 504")
        || low.contains(" 500 ")
        || low.contains(" 503 ")
        || low.contains(" 504 ")
        || low.contains(" 429 ")
}

fn normalize_runtime_name(raw: &str) -> String {
    let trimmed = raw.trim().trim_start_matches('/');
    if trimmed.starts_with("runtimes/") {
        trimmed.to_string()
    } else {
        format!("runtimes/{trimmed}")
    }
}

fn runtime_conn(runtime: &ColabRuntime) -> Result<&ColabConnectionInfo, String> {
    let ci = runtime
        .connection_info
        .as_ref()
        .ok_or_else(|| "Colab runtime is missing connectionInfo.".to_string())?;
    if ci.url.trim().is_empty() || ci.token.trim().is_empty() {
        return Err("Colab runtime connectionInfo URL or token is empty.".into());
    }
    Ok(ci)
}

fn encode_path_segments(path: &str) -> String {
    path.trim_matches('/')
        .split('/')
        .filter(|s| !s.is_empty())
        .map(|seg| url::form_urlencoded::byte_serialize(seg.as_bytes()).collect::<String>())
        .collect::<Vec<_>>()
        .join("/")
}

fn parse_single_runtime_spec(item: &Value) -> Option<ColabRuntimeSpec> {
    if let Some(key_obj) = item.get("key") {
        let mut spec = serde_json::from_value::<ColabRuntimeSpec>(key_obj.clone()).ok()?;
        spec.eligible = item
            .get("eligible")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if !spec.variant.is_empty() || !spec.accelerator.is_empty() {
            return Some(spec);
        }
        return None;
    }
    let target = item.get("runtimeSpec").unwrap_or(item);
    let mut spec = serde_json::from_value::<ColabRuntimeSpec>(target.clone()).ok()?;
    spec.eligible = item
        .get("eligible")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    if !spec.variant.is_empty() || !spec.accelerator.is_empty() {
        Some(spec)
    } else {
        None
    }
}

fn parse_runtime_specs(val: &Value) -> Vec<ColabRuntimeSpec> {
    let mut specs = Vec::new();
    let candidate = val
        .get("runtimeSpecs")
        .or_else(|| val.get("specs"))
        .unwrap_or(val);
    match candidate {
        Value::Array(items) => {
            for item in items {
                if let Some(spec) = parse_single_runtime_spec(item) {
                    specs.push(spec);
                }
            }
        }
        Value::Object(map) => {
            for (_k, item) in map {
                if let Some(spec) = parse_single_runtime_spec(item) {
                    specs.push(spec);
                }
            }
        }
        _ => {}
    }
    specs
}

fn accelerator_rank(spec: &ColabRuntimeSpec) -> i32 {
    let v = spec.variant.to_ascii_uppercase();
    let a = spec.accelerator.to_ascii_uppercase();
    let base = if v.contains("GPU") {
        100
    } else if v.contains("TPU") {
        50
    } else {
        10
    };
    let accel_bonus = match a.as_str() {
        "T4" => 40,
        "L4" => 35,
        "A100" | "A100_40GB" | "A100_80GB" => 30,
        "H100" => 25,
        "V100" => 20,
        "NONE" | "" => 0,
        _ => 15,
    };
    let shape_bonus = if spec.shape.to_ascii_uppercase().contains("STANDARD") {
        2
    } else {
        1
    };
    base + accel_bonus + shape_bonus
}

pub(crate) fn select_best_runtime_spec(
    specs: &[ColabRuntimeSpec],
    preferred_accelerator: Option<&str>,
    preferred_variant: Option<&str>,
) -> Option<ColabRuntimeSpec> {
    if specs.is_empty() {
        return None;
    }
    let eligible_pool: Vec<&ColabRuntimeSpec> = specs.iter().filter(|s| s.eligible).collect();
    let active_pool: Vec<&ColabRuntimeSpec> = if eligible_pool.is_empty() {
        specs.iter().collect()
    } else {
        eligible_pool
    };
    let pref_acc = preferred_accelerator
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_ascii_uppercase);
    let pref_var = preferred_variant
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_ascii_uppercase);

    if pref_acc.is_some() || pref_var.is_some() {
        let mut exact: Vec<&ColabRuntimeSpec> = active_pool
            .iter()
            .copied()
            .filter(|s| {
                let ok_acc = pref_acc
                    .as_ref()
                    .map(|p| s.accelerator.eq_ignore_ascii_case(p))
                    .unwrap_or(true);
                let ok_var = pref_var
                    .as_ref()
                    .map(|p| s.variant.eq_ignore_ascii_case(p))
                    .unwrap_or(true);
                ok_acc && ok_var
            })
            .collect();
        exact.sort_by_key(|s| -accelerator_rank(s));
        if let Some(best) = exact.first() {
            return Some((*best).clone());
        }
    }

    let mut ranked = active_pool;
    ranked.sort_by_key(|s| -accelerator_rank(s));
    ranked.first().map(|s| (*s).clone())
}

pub(crate) fn runtime_spec_matches(candidate: &ColabRuntimeSpec, target: &ColabRuntimeSpec) -> bool {
    let var_ok = target.variant.is_empty()
        || candidate.variant.is_empty()
        || candidate.variant.eq_ignore_ascii_case(&target.variant);
    let acc_ok = target.accelerator.is_empty()
        || candidate.accelerator.is_empty()
        || candidate.accelerator.eq_ignore_ascii_case(&target.accelerator)
        || candidate
            .accelerator
            .to_ascii_uppercase()
            .contains(&target.accelerator.to_ascii_uppercase());
    var_ok && acc_ok
}

async fn decode_colab_json<T: serde::de::DeserializeOwned>(
    resp: reqwest::Response,
    context: &str,
) -> Result<T, String> {
    let status = resp.status();
    let body = resp
        .text()
        .await
        .map_err(|e| format!("{context}: failed reading response ({e})"))?;
    if !status.is_success() {
        return Err(format_colab_error(context, status.as_u16(), &body));
    }
    serde_json::from_str::<T>(&body).map_err(|e| {
        let snippet: String = body.chars().take(280).collect();
        format!("{context}: invalid JSON ({e}): {snippet}")
    })
}

pub fn format_colab_error(context: &str, status: u16, body: &str) -> String {
    if let Ok(val) = serde_json::from_str::<Value>(body) {
        return format_google_error_value(context, status, &val);
    }
    let trimmed = body.trim();
    if trimmed.is_empty() {
        format!("{context}: HTTP {status}")
    } else {
        format!("{context} (HTTP {status}): {trimmed}")
    }
}

fn format_google_error_value(context: &str, status: u16, val: &Value) -> String {
    let err = val.get("error").unwrap_or(val);
    let code = err
        .get("code")
        .and_then(|c| c.as_u64())
        .unwrap_or(status as u64);
    let msg = err
        .get("message")
        .and_then(|m| m.as_str())
        .unwrap_or("Colab API request rejected");
    let mut reason = String::new();
    let mut activation_url = String::new();

    if let Some(details) = err.get("details").and_then(|d| d.as_array()) {
        for item in details {
            if reason.is_empty() {
                if let Some(r) = item.get("reason").and_then(|v| v.as_str()) {
                    reason = r.to_string();
                }
            }
            if activation_url.is_empty() {
                if let Some(u) = item
                    .get("metadata")
                    .and_then(|m| m.get("activationUrl"))
                    .and_then(|v| v.as_str())
                {
                    activation_url = u.to_string();
                }
            }
            if activation_url.is_empty() {
                if let Some(links) = item.get("links").and_then(|l| l.as_array()) {
                    for l in links {
                        if let Some(u) = l.get("url").and_then(|v| v.as_str()) {
                            activation_url = u.to_string();
                            break;
                        }
                    }
                }
            }
        }
    }
    if reason.is_empty() {
        reason = err
            .get("status")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();
    }

    let code_prefix = if code < 100 { "gRPC" } else { "HTTP" };
    let mut out = if reason.is_empty() {
        format!("{context} ({code_prefix} {code}): {msg}")
    } else {
        format!("{context} ({code_prefix} {code} {reason}): {msg}")
    };
    if !activation_url.is_empty() {
        out.push_str(&format!(" | Enable API: {activation_url}"));
    }
    out
}

fn process_mcp_sse_line<F>(raw_line: &str, rpc_messages: &mut Vec<Value>, on_progress: &mut F)
where
    F: FnMut(&str),
{
    let line_t = raw_line.trim();
    if let Some(data_part) = line_t.strip_prefix("data:") {
        let json_str = data_part.trim();
        if json_str.is_empty() || json_str == "[DONE]" {
            return;
        }
        if let Ok(v) = serde_json::from_str::<Value>(json_str) {
            if v.get("method").and_then(|m| m.as_str()) == Some("notifications/progress") {
                if let Some(msg) = v
                    .get("params")
                    .and_then(|p| p.get("message"))
                    .and_then(|m| m.as_str())
                {
                    on_progress(msg);
                }
            }
            rpc_messages.push(v);
        }
    }
}

fn select_mcp_jsonrpc_result(rpc_messages: &[Value], target_id: u64) -> Result<Value, String> {
    let chosen = rpc_messages
        .iter()
        .rfind(|v| v.get("id").and_then(|i| i.as_u64()) == Some(target_id))
        .or_else(|| rpc_messages.iter().rfind(|v| v.get("result").is_some() || v.get("error").is_some()))
        .or_else(|| rpc_messages.last())
        .ok_or_else(|| "MCP stream contained no JSON-RPC payload.".to_string())?;

    if let Some(rpc_err) = chosen.get("error") {
        let code = rpc_err.get("code").and_then(|c| c.as_i64()).unwrap_or(-1);
        let msg = rpc_err
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("Unknown MCP JSON-RPC error");
        return Err(format!("Colab MCP JSON-RPC error ({code}): {msg}"));
    }

    let result = chosen
        .get("result")
        .cloned()
        .ok_or_else(|| "MCP JSON-RPC response lacked result field.".to_string())?;
    if result
        .get("isError")
        .and_then(|b| b.as_bool())
        .unwrap_or(false)
    {
        let err_text = extract_mcp_text_segments(&result);
        return Err(if err_text.is_empty() {
            "Colab MCP tool returned isError=true.".into()
        } else {
            format!("Colab MCP tool error: {err_text}")
        });
    }
    Ok(result)
}

fn parse_mcp_jsonrpc_result(
    status: u16,
    content_type: &str,
    body: &str,
    target_id: u64,
) -> Result<Value, String> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return Err(format!("Empty MCP response (HTTP {status})."));
    }

    let mut rpc_messages: Vec<Value> = Vec::new();
    if content_type.to_ascii_lowercase().contains("text/event-stream")
        || trimmed.starts_with("data:")
        || trimmed.starts_with("event:")
    {
        for line in trimmed.lines() {
            process_mcp_sse_line(line, &mut rpc_messages, &mut |_| {});
        }
    } else if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        if let Some(arr) = v.as_array() {
            rpc_messages.extend(arr.iter().cloned());
        } else {
            rpc_messages.push(v);
        }
    } else {
        return Err(format!("Unparseable MCP response (HTTP {status}): {trimmed}"));
    }

    select_mcp_jsonrpc_result(&rpc_messages, target_id)
}

fn extract_mcp_text_segments(result: &Value) -> String {
    result
        .get("content")
        .and_then(|c| c.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| item.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

fn extract_mcp_tool_payload(result: &Value) -> Result<Value, String> {
    if let Some(structured) = result.get("structuredContent") {
        if !structured.is_null() {
            return Ok(structured.clone());
        }
    }
    let text = extract_mcp_text_segments(result);
    if text.trim().is_empty() {
        return Ok(json!({}));
    }
    match serde_json::from_str::<Value>(text.trim()) {
        Ok(v) => Ok(v),
        Err(_) => Ok(json!({ "stdout": text })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[derive(Clone, Debug)]
    struct RecordedReq {
        method: String,
        path: String,
        headers: String,
        body: String,
    }

    async fn read_http_request(sock: &mut tokio::net::TcpStream) -> Option<RecordedReq> {
        let mut buf = Vec::new();
        let mut temp = [0u8; 4096];
        let mut header_end = None;
        let mut content_len = 0usize;

        loop {
            let n = sock.read(&mut temp).await.ok()?;
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&temp[..n]);
            if header_end.is_none() {
                if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    header_end = Some(pos + 4);
                    let hdr_str = String::from_utf8_lossy(&buf[..pos]);
                    for line in hdr_str.lines() {
                        if let Some((k, v)) = line.split_once(':') {
                            if k.trim().eq_ignore_ascii_case("content-length") {
                                content_len = v.trim().parse::<usize>().unwrap_or(0);
                            }
                        }
                    }
                }
            }
            if let Some(he) = header_end {
                if buf.len() >= he + content_len {
                    break;
                }
            }
        }
        let he = header_end?;
        let head = String::from_utf8_lossy(&buf[..he]).to_string();
        let first = head.lines().next().unwrap_or("");
        let mut parts = first.split_whitespace();
        let method = parts.next().unwrap_or("").to_string();
        let path = parts.next().unwrap_or("").to_string();
        let body = String::from_utf8_lossy(&buf[he..buf.len().min(he + content_len)]).to_string();
        Some(RecordedReq {
            method,
            path,
            headers: head,
            body,
        })
    }

    async fn spawn_route_server<F>(handler: F) -> (String, Arc<Mutex<Vec<RecordedReq>>>)
    where
        F: Fn(&RecordedReq) -> (String, Vec<(&'static str, &'static str)>, Vec<u8>) + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let recorded = Arc::new(Mutex::new(Vec::new()));
        let rec_clone = recorded.clone();
        let h = Arc::new(handler);
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                let rec_inner = rec_clone.clone();
                let h_inner = h.clone();
                tokio::spawn(async move {
                    if let Some(req) = read_http_request(&mut sock).await {
                        let (status, extra_hdrs, body) = h_inner(&req);
                        if let Ok(mut g) = rec_inner.lock() {
                            g.push(req);
                        }
                        let mut head = format!(
                            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n",
                            body.len()
                        );
                        for (k, v) in extra_hdrs {
                            head.push_str(&format!("{k}: {v}\r\n"));
                        }
                        head.push_str("\r\n");
                        let _ = sock.write_all(head.as_bytes()).await;
                        let _ = sock.write_all(&body).await;
                        let _ = sock.shutdown().await;
                    }
                });
            }
        });
        (format!("http://127.0.0.1:{port}"), recorded)
    }

    #[tokio::test]
    async fn test_runtime_lifecycle_and_hardware_selection() {
        let rt_poll_count = Arc::new(Mutex::new(0usize));
        let poll_c = rt_poll_count.clone();
        let (base, recorded) = spawn_route_server(move |req| match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/v1beta/runtimespecs") => (
                "200 OK".into(),
                vec![("Content-Type", "application/json")],
                br#"{"runtimeSpecs":[{"key":{"variant":"VARIANT_GPU","accelerator":"A100","shape":"SHAPE_HIGHMEM"}},{"eligible":true,"key":{"variant":"VARIANT_CPU","accelerator":"NONE","shape":"SHAPE_STANDARD"}},{"eligible":true,"key":{"variant":"VARIANT_GPU","accelerator":"T4","shape":"SHAPE_STANDARD"}}]}"#.to_vec(),
            ),
            ("GET", "/v1beta/runtimes") => (
                "200 OK".into(),
                vec![("Content-Type", "application/json")],
                br#"{"runtimes":[]}"#.to_vec(),
            ),
            ("POST", "/v1beta/runtimes") => (
                "200 OK".into(),
                vec![("Content-Type", "application/json")],
                br#"{"name":"operations/op_123","done":false}"#.to_vec(),
            ),
            ("GET", "/v1/operations/op_123") => {
                let mut cnt = poll_c.lock().unwrap();
                *cnt += 1;
                if *cnt == 1 {
                    (
                        "200 OK".into(),
                        vec![("Content-Type", "application/json")],
                        br#"{"name":"operations/op_123","done":false}"#.to_vec(),
                    )
                } else {
                    (
                        "200 OK".into(),
                        vec![("Content-Type", "application/json")],
                        br#"{"name":"operations/op_123","done":true,"response":{"name":"runtimes/rt_abc","runtimeSpec":{"variant":"VARIANT_GPU","accelerator":"T4","shape":"SHAPE_STANDARD"},"connectionInfo":{"url":"http://127.0.0.1:9999","token":"proxy_tok_1","expireTime":"2099-01-01T00:00:00Z"}}}"#.to_vec(),
                    )
                }
            }
            ("DELETE", "/v1beta/runtimes/rt_abc") => (
                "200 OK".into(),
                vec![("Content-Type", "application/json")],
                b"{}".to_vec(),
            ),
            _ => ("404 Not Found".into(), vec![], b"{}".to_vec()),
        })
        .await;

        let client = ColabClient::with_endpoints(
            "tok_oauth_1".into(),
            "100200300400".into(),
            base.clone(),
            format!("{base}/mcp"),
        )
        .unwrap();

        let specs = client.list_runtime_specs().await.unwrap();
        assert_eq!(specs.len(), 3);
        assert!(!specs[0].eligible);
        assert!(specs[2].eligible);
        let auto_best = select_best_runtime_spec(&specs, None, None).unwrap();
        assert_eq!(auto_best.accelerator, "T4");
        let cancel = AtomicBool::new(false);
        let (rt, created) = client
            .pick_or_allocate_runtime(Some("T4"), Some("VARIANT_GPU"), &cancel)
            .await
            .unwrap();
        assert!(created);
        assert_eq!(rt.name, "runtimes/rt_abc");
        assert_eq!(rt.connection_info.as_ref().unwrap().token, "proxy_tok_1");
        client.delete_runtime(&rt.name).await.unwrap();

        let reqs = recorded.lock().unwrap();
        assert!(reqs
            .iter()
            .all(|r| r.headers.to_ascii_lowercase().contains("x-goog-user-project: 100200300400")));
    }

    #[tokio::test]
    async fn test_error_classifiers_grpc_formatting_and_reusable_hydration() {
        assert!(is_auth_expired_error(
            "Colab operation poll error (HTTP 401 UNAUTHENTICATED): bad token"
        ));
        assert!(is_auth_expired_error("invalid authentication credentials"));
        assert!(!is_auth_expired_error("HTTP 403 SERVICE_DISABLED"));

        assert!(is_transient_allocation_error(
            "Colab allocation operation failed (gRPC 4 DEADLINE_EXCEEDED): context deadline exceeded"
        ));
        assert!(is_transient_allocation_error("HTTP 503 UNAVAILABLE"));
        assert!(!is_transient_allocation_error("HTTP 403 SERVICE_DISABLED"));

        let grpc_err = format_google_error_value(
            "Colab allocation operation failed",
            500,
            &json!({
                "code": 4,
                "message": "context deadline exceeded",
                "status": "DEADLINE_EXCEEDED"
            }),
        );
        assert!(grpc_err.contains("(gRPC 4 DEADLINE_EXCEEDED)"));
        assert!(!grpc_err.contains("HTTP 4"));

        let (base, _) = spawn_route_server(move |req| match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/v1beta/runtimes") => (
                "200 OK".into(),
                vec![("Content-Type", "application/json")],
                br#"{"runtimes":[{"name":"runtimes/rt_warm","state":"RUNNING","runtimeSpec":{"variant":"VARIANT_GPU","accelerator":"T4","shape":"SHAPE_STANDARD"}}]}"#.to_vec(),
            ),
            ("GET", "/v1beta/runtimes/rt_warm") => (
                "200 OK".into(),
                vec![("Content-Type", "application/json")],
                br#"{"name":"runtimes/rt_warm","state":"RUNNING","runtimeSpec":{"variant":"VARIANT_GPU","accelerator":"T4","shape":"SHAPE_STANDARD"},"connectionInfo":{"url":"http://127.0.0.1:8888","token":"warm_tok","expireTime":"2099-01-01T00:00:00Z"}}"#.to_vec(),
            ),
            _ => ("404 Not Found".into(), vec![], b"{}".to_vec()),
        })
        .await;

        let client = ColabClient::with_endpoints(
            "tok".into(),
            "100200300400".into(),
            base.clone(),
            format!("{base}/mcp"),
        )
        .unwrap();
        let target = ColabRuntimeSpec {
            variant: "VARIANT_GPU".into(),
            accelerator: "T4".into(),
            shape: "SHAPE_STANDARD".into(),
            eligible: true,
        };
        let reused = client.find_reusable_runtime(&target).await.unwrap();
        assert_eq!(reused.name, "runtimes/rt_warm");
        assert_eq!(reused.connection_info.unwrap().token, "warm_tok");
    }

    #[tokio::test]
    async fn test_jupyter_streaming_upload_and_download() {
        let chunks = Arc::new(Mutex::new(Vec::<(Option<i64>, Vec<u8>)>::new()));
        let chunks_c = chunks.clone();
        let (base, recorded) = spawn_route_server(move |req| {
            let p = req.path.split('?').next().unwrap_or("");
            if req.method == "GET" && p == "/api/contents/content" {
                return (
                    "200 OK".into(),
                    vec![("Content-Type", "application/json")],
                    br#"{"name":"content","path":"content","type":"directory"}"#.to_vec(),
                );
            }
            if req.method == "GET" && p.starts_with("/api/contents/damnisium_work") {
                if p.ends_with("/output") {
                    return (
                        "200 OK".into(),
                        vec![("Content-Type", "application/json")],
                        br#"{"name":"output","path":"damnisium_work/output","type":"directory","content":[{"name":".ipynb_checkpoints","path":"damnisium_work/output/.ipynb_checkpoints","type":"directory"},{"name":"best.onnx","path":"damnisium_work/output/best.onnx","type":"file","size":12},{"name":"fallback.bin","path":"damnisium_work/output/fallback.bin","type":"file","size":8}]}"#.to_vec(),
                    );
                }
                if p == "/api/contents/damnisium_work/output/fallback.bin" {
                    let b64_pay = B64.encode(b"FALLBACK");
                    let body = format!(
                        r#"{{"name":"fallback.bin","path":"damnisium_work/output/fallback.bin","type":"file","format":"base64","content":"{b64_pay}"}}"#
                    );
                    return (
                        "200 OK".into(),
                        vec![("Content-Type", "application/json")],
                        body.into_bytes(),
                    );
                }
                return (
                    "200 OK".into(),
                    vec![("Content-Type", "application/json")],
                    br#"{"name":"damnisium_work","path":"damnisium_work","type":"directory"}"#.to_vec(),
                );
            }
            if req.method == "PUT" && p == "/api/contents/damnisium_work/big.bin" {
                let v: Value = serde_json::from_str(&req.body).unwrap();
                let ch = v.get("chunk").and_then(|c| c.as_i64());
                let raw = B64.decode(v["content"].as_str().unwrap()).unwrap();
                chunks_c.lock().unwrap().push((ch, raw));
                return ("201 Created".into(), vec![("Content-Type", "application/json")], b"{}".to_vec());
            }
            if req.method == "GET" && p == "/files/damnisium_work/output/best.onnx" {
                return (
                    "200 OK".into(),
                    vec![("Content-Type", "application/octet-stream")],
                    b"ONNX_BIN_OK!".to_vec(),
                );
            }
            ("404 Not Found".into(), vec![], b"{}".to_vec())
        })
        .await;

        let rt = ColabRuntime {
            name: "runtimes/r1".into(),
            runtime_spec: None,
            state: None,
            connection_info: Some(ColabConnectionInfo {
                url: base.clone(),
                token: "jup_proxy_tok".into(),
                expire_time: "2099-01-01T00:00:00Z".into(),
            }),
        };
        let client = ColabClient::new("tok".into(), "100200300400".into()).unwrap();

        let tmp_dir = std::env::temp_dir().join(format!("damnisium_jup_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let big_file = tmp_dir.join("big.bin");
        let total_len = JUPYTER_CHUNK_BYTES + 128;
        let payload = vec![0x5Au8; total_len];
        std::fs::write(&big_file, &payload).unwrap();

        client
            .upload_file_streaming(&rt, &big_file, "damnisium_work/big.bin", |_, _| {})
            .await
            .unwrap();

        {
            let ch_list = chunks.lock().unwrap();
            assert_eq!(ch_list.len(), 2);
            assert_eq!(ch_list[0].0, Some(1));
            assert_eq!(ch_list[0].1.len(), JUPYTER_CHUNK_BYTES);
            assert_eq!(ch_list[1].0, Some(-1));
            assert_eq!(ch_list[1].1.len(), 128);
        }

        let resolved_base = client.resolve_remote_work_base(&rt).await;
        assert_eq!(resolved_base, "content/damnisium_work");

        let files = client.list_remote_files_recursive(&rt, "damnisium_work/output").await.unwrap();
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].name, "best.onnx");
        assert_eq!(files[1].name, "fallback.bin");

        let dl_dest = tmp_dir.join("best.onnx");
        let got = client
            .download_file_streaming(&rt, "damnisium_work/output/best.onnx", &dl_dest)
            .await
            .unwrap();
        assert_eq!(got, 12);
        assert_eq!(std::fs::read(&dl_dest).unwrap(), b"ONNX_BIN_OK!");

        let fb_dest = tmp_dir.join("fallback.bin");
        let got_fb = client
            .download_file_streaming(&rt, "damnisium_work/output/fallback.bin", &fb_dest)
            .await
            .unwrap();
        assert_eq!(got_fb, 8);
        assert_eq!(std::fs::read(&fb_dest).unwrap(), b"FALLBACK");

        let reqs = recorded.lock().unwrap();
        assert!(reqs
            .iter()
            .all(|r| r.headers.to_ascii_lowercase().contains("x-colab-runtime-proxy-token: jup_proxy_tok")));
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[tokio::test]
    async fn test_mcp_session_and_resumable_execution_retry() {
        let exec_attempts = Arc::new(Mutex::new(Vec::<String>::new()));
        let exec_c = exec_attempts.clone();
        let (base, _) = spawn_route_server(move |req| {
            let v: Value = serde_json::from_str(&req.body).unwrap();
            let id = v.get("id").and_then(|x| x.as_u64()).unwrap_or(0);
            let method = v.get("method").and_then(|m| m.as_str()).unwrap_or("");
            if method == "initialize" {
                return (
                    "200 OK".into(),
                    vec![("Content-Type", "application/json"), ("Mcp-Session-Id", "mcp_sid_1")],
                    format!(r#"{{"jsonrpc":"2.0","id":{id},"result":{{"protocolVersion":"2024-11-05"}}}}"#).into_bytes(),
                );
            }
            if method == "notifications/initialized" {
                return ("202 Accepted".into(), vec![], b"{}".to_vec());
            }
            let tool = v["params"]["name"].as_str().unwrap_or("");
            if tool == "create_session" {
                let req_sid = v["params"]["arguments"]["sessionId"].as_str().unwrap_or("");
                assert!(!req_sid.contains('_'));
                let inner = format!(r#"{{"name":"runtimes/rt1/sessions/{req_sid}"}}"#);
                let resp_json = json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": { "content": [{ "type": "text", "text": inner }] }
                });
                return (
                    "200 OK".into(),
                    vec![("Content-Type", "application/json")],
                    resp_json.to_string().into_bytes(),
                );
            }
            if tool == "execute_code" {
                assert!(v["params"]["arguments"].get("target").is_none());
                assert!(v["params"]["arguments"]["session"].as_str().unwrap_or("").starts_with("runtimes/rt1/sessions/"));
                let eid = v["params"]["arguments"]["executionId"].as_str().unwrap().to_string();
                let mut g = exec_c.lock().unwrap();
                g.push(eid);
                if g.len() == 1 {
                    let err_resp = json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": { "code": 504, "message": "DEADLINE_EXCEEDED waiting for cell" }
                    });
                    return (
                        "200 OK".into(),
                        vec![("Content-Type", "application/json")],
                        err_resp.to_string().into_bytes(),
                    );
                }
                let ok_inner = r#"{"stdout":"epoch 1 ok\n","stderr":"","outputTruncated":false}"#;
                let prog_line = json!({
                    "jsonrpc": "2.0",
                    "method": "notifications/progress",
                    "params": { "message": "live chunk 1\n", "progress": 1 }
                });
                let sse_line = json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": { "content": [{ "type": "text", "text": ok_inner }] }
                });
                let sse = format!("event: message\ndata: {}\n\nevent: message\ndata: {}\n\n", prog_line, sse_line);
                return (
                    "200 OK".into(),
                    vec![("Content-Type", "text/event-stream")],
                    sse.into_bytes(),
                );
            }
            if tool == "delete_session" {
                let ok_resp = json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": { "content": [{ "type": "text", "text": "{}" }] }
                });
                return (
                    "200 OK".into(),
                    vec![("Content-Type", "application/json")],
                    ok_resp.to_string().into_bytes(),
                );
            }
            ("404 Not Found".into(), vec![], b"{}".to_vec())
        })
        .await;

        let mut client = ColabClient::with_endpoints(
            "tok".into(),
            "100200300400".into(),
            base.clone(),
            format!("{base}/mcp"),
        )
        .unwrap();
        let sess = client
            .create_session("runtimes/rt1", "damnisium-run_0e0f331efe6b")
            .await
            .unwrap();
        assert_eq!(sess, "runtimes/rt1/sessions/damnisium-run-0e0f331efe6b");
        assert_eq!(sanitize_mcp_session_id("99_Bad__ID!"), "s-99-bad-id");
        let err_fmt = format_mcp_execution_error(&json!({
            "name": "ValueError",
            "value": "\u{1b}[31mbad shape\u{1b}[0m",
            "traceback": ["\u{1b}[0;31mline 1\u{1b}[0m", "---"]
        }));
        assert!(err_fmt.contains("ValueError: bad shape"));
        assert!(!err_fmt.contains('\x1b'));
        assert_eq!(strip_ansi_codes("\u{1b}[1;32mOK\u{1b}[0m"), "OK");
        let cancel = AtomicBool::new(false);
        let mut streamed_msgs = Vec::new();
        let (used_eid, res) = client
            .execute_code_resumable_streaming(
                &sess,
                "print('epoch 1 ok')",
                Some("exec-uuid-777"),
                &cancel,
                |msg| streamed_msgs.push(msg.to_string()),
            )
            .await
            .unwrap();
        assert_eq!(used_eid, "exec-uuid-777");
        assert_eq!(res.stdout, "epoch 1 ok\n");
        assert_eq!(streamed_msgs, vec!["live chunk 1\n"]);
        {
            let recorded_eids = exec_attempts.lock().unwrap();
            assert_eq!(*recorded_eids, vec!["exec-uuid-777", "exec-uuid-777"]);
        }
        client.delete_session(&sess).await.unwrap();
    }
}
