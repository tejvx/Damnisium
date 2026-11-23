use crate::models::{
    HfDownloadProgressPayload, HfHubModel, HfHubResponse, HfModelFile,
};
use crate::storage::StorageManager;
use futures_util::stream::{self, StreamExt};
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Duration;
use tauri::{AppHandle, Emitter};
use url::form_urlencoded;

#[derive(Debug, Deserialize)]
struct HfApiSibling {
    rfilename: String,
    size: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct HfApiModelEntry {
    id: Option<String>,
    #[serde(rename = "modelId")]
    model_id: Option<String>,
    author: Option<String>,
    downloads: Option<u64>,
    likes: Option<u64>,
    #[serde(rename = "trendingScore")]
    trending_score: Option<f64>,
    #[serde(rename = "lastModified")]
    last_modified: Option<String>,
    pipeline_tag: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    siblings: Vec<HfApiSibling>,
    gguf: Option<Value>,
    safetensors: Option<Value>,
}

pub fn normalize_repo_id(raw: &str) -> Result<String, String> {
    let mut s = raw.trim();
    if s.is_empty() {
        return Err("Hugging Face repo ID cannot be empty.".into());
    }
    if let Some(pos) = s.find('#') {
        s = &s[..pos];
    }
    if let Some(pos) = s.find('?') {
        s = &s[..pos];
    }
    s = s.trim_end_matches('/');

    if let Some(rest) = s
        .strip_prefix("https://huggingface.co/")
        .or_else(|| s.strip_prefix("http://huggingface.co/"))
        .or_else(|| s.strip_prefix("https://www.huggingface.co/"))
        .or_else(|| s.strip_prefix("http://www.huggingface.co/"))
    {
        s = rest;
    }

    let parts: Vec<&str> = s.split('/').filter(|p| !p.is_empty()).collect();
    if parts.len() < 2 {
        return Err("Expected Hugging Face format 'owner/repo' or full repo URL.".into());
    }
    let owner = parts[0];
    let repo = parts[1];
    if owner == ".." || repo == ".." || owner.contains('\\') || repo.contains('\\') {
        return Err("Invalid Hugging Face repository identifier.".into());
    }
    Ok(format!("{owner}/{repo}"))
}

pub fn parse_param_count_b(text: &str, safetensors_val: Option<&Value>) -> Option<f64> {
    if let Some(st) = safetensors_val {
        if let Some(total) = st.get("total").and_then(|v| v.as_u64()) {
            if total > 0 {
                return Some((total as f64) / 1_000_000_000.0);
            }
        }
        if let Some(params_obj) = st.get("parameters").and_then(|v| v.as_object()) {
            let sum: u64 = params_obj.values().filter_map(|v| v.as_u64()).sum();
            if sum > 0 {
                return Some((sum as f64) / 1_000_000_000.0);
            }
        }
    }

    let lower = text.to_lowercase();
    let bytes = lower.as_bytes();
    let len = bytes.len();
    let mut best: Option<f64> = None;

    for i in 0..len {
        let is_sep_before = i == 0
            || matches!(
                bytes[i - 1],
                b'-' | b'_' | b'/' | b' ' | b'(' | b'['
            )
            || (bytes[i - 1] == b'e'
                && (i == 1 || matches!(bytes[i - 2], b'-' | b'_' | b'/' | b' ')));
        if !is_sep_before {
            continue;
        }
        if !bytes[i].is_ascii_digit() {
            continue;
        }
        let mut j = i;
        let mut seen_dot = false;
        while j < len {
            let b = bytes[j];
            if b.is_ascii_digit() {
                j += 1;
            } else if b == b'.' && !seen_dot {
                seen_dot = true;
                j += 1;
            } else {
                break;
            }
        }
        if j >= len || (bytes[j] != b'b' && bytes[j] != b'm') {
            continue;
        }
        let unit = bytes[j];
        let after_ok = j + 1 == len
            || matches!(
                bytes[j + 1],
                b'-' | b'_' | b'/' | b' ' | b'.' | b')' | b']' | b','
            );
        if !after_ok {
            continue;
        }
        if let Ok(val) = lower[i..j].parse::<f64>() {
            let val_b = if unit == b'b' {
                if val > 0.05 && val <= 1000.0 {
                    Some(val)
                } else {
                    None
                }
            } else if (1.0..=999.0).contains(&val) {
                Some(val / 1000.0)
            } else {
                None
            };
            if let Some(candidate) = val_b {
                best = Some(best.map(|prev| prev.max(candidate)).unwrap_or(candidate));
            }
        }
    }
    best
}

pub fn format_param_label(param_b: f64) -> String {
    if param_b < 1.0 {
        format!("{}M", (param_b * 1000.0).round() as u64)
    } else if (param_b - param_b.round()).abs() < 0.05 && param_b >= 2.0 {
        format!("{}B", param_b.round() as u64)
    } else {
        format!("{param_b:.1}B")
    }
}

pub fn format_size_label(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", (bytes as f64) / 1_073_741_824.0)
    } else {
        format!("{} MB", ((bytes as f64) / 1_048_576.0).round().max(1.0) as u64)
    }
}

fn infer_module_hint(pipeline_tag: Option<&str>, tags: &[String], repo_id: &str) -> String {
    let pt = pipeline_tag.unwrap_or("").to_lowercase();
    let low_repo = repo_id.to_lowercase();
    let all_tags = tags.join(" ").to_lowercase();

    if pt.contains("audio")
        || pt.contains("speech")
        || pt.contains("voice")
        || pt == "automatic-speech-recognition"
        || pt == "text-to-speech"
        || all_tags.contains("whisper")
        || all_tags.contains("audioset")
        || all_tags.contains("yamnet")
    {
        return "audio".to_string();
    }

    if pt.contains("image")
        || pt.contains("vision")
        || pt.contains("object-detection")
        || pt.contains("segmentation")
        || pt == "text-to-image"
        || pt == "image-classification"
        || low_repo.contains("-image-")
        || low_repo.contains("mobilenet")
        || low_repo.contains("yolo")
    {
        return "image".to_string();
    }

    "text".to_string()
}

fn classify_file_format(filename: &str) -> &'static str {
    let low = filename.to_lowercase();
    if low.ends_with(".gguf") {
        "GGUF"
    } else if low.ends_with(".onnx") {
        "ONNX"
    } else if low.ends_with(".safetensors") {
        "Safetensors"
    } else if low.ends_with(".bin") || low.ends_with(".pt") {
        "Weights"
    } else {
        "Other"
    }
}

fn build_file_entries(repo_id: &str, siblings: &[HfApiSibling]) -> Vec<HfModelFile> {
    let mut files = Vec::new();
    for s in siblings {
        let fmt = classify_file_format(&s.rfilename);
        if fmt == "Other" {
            continue;
        }
        let low = s.rfilename.to_lowercase();
        let is_mmproj = low.contains("mmproj");
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo_id,
            s.rfilename.trim_start_matches('/')
        );
        files.push(HfModelFile {
            rfilename: s.rfilename.clone(),
            size_bytes: s.size,
            format: fmt.to_string(),
            is_mmproj,
            download_url: url,
        });
    }
    files.sort_by(|a, b| {
        let ra = a.size_bytes.unwrap_or(u64::MAX);
        let rb = b.size_bytes.unwrap_or(u64::MAX);
        ra.cmp(&rb)
    });
    files
}

fn build_hub_model(entry: HfApiModelEntry, is_pinned: bool, summary_override: Option<&str>) -> Option<HfHubModel> {
    let repo_id = entry.id.or(entry.model_id)?;
    let parts: Vec<&str> = repo_id.split('/').collect();
    let author = entry
        .author
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| parts.first().copied().unwrap_or("").to_string());
    let name = parts.get(1).copied().unwrap_or(&repo_id).to_string();

    let mut format_set: HashSet<String> = HashSet::new();
    let files = build_file_entries(&repo_id, &entry.siblings);
    for f in &files {
        if f.format == "GGUF" || f.format == "ONNX" || f.format == "Safetensors" {
            format_set.insert(f.format.clone());
        }
    }
    for t in &entry.tags {
        let low = t.to_lowercase();
        if low == "gguf" {
            format_set.insert("GGUF".into());
        } else if low == "onnx" {
            format_set.insert("ONNX".into());
        } else if low == "safetensors" {
            format_set.insert("Safetensors".into());
        }
    }
    if entry.gguf.is_some() {
        format_set.insert("GGUF".into());
    }
    if entry.safetensors.is_some() {
        format_set.insert("Safetensors".into());
    }
    let mut format_tags: Vec<String> = format_set.into_iter().collect();
    format_tags.sort();

    let combined_text = format!("{} {}", repo_id, entry.tags.join(" "));
    let param_b = parse_param_count_b(&combined_text, entry.safetensors.as_ref());
    let param_label = param_b.map(format_param_label);

    let safetensor_bytes: u64 = files
        .iter()
        .filter(|f| !f.is_mmproj && f.format == "Safetensors")
        .filter_map(|f| f.size_bytes)
        .sum();
    let weight_bin_bytes: u64 = files
        .iter()
        .filter(|f| !f.is_mmproj && f.format == "Weights")
        .filter_map(|f| f.size_bytes)
        .filter(|&sz| sz > 1_000_000)
        .sum();
    let onnx_bytes: u64 = entry
        .siblings
        .iter()
        .filter(|s| {
            let l = s.rfilename.to_lowercase();
            !l.contains("mmproj") && (l.ends_with(".onnx") || l.ends_with(".onnx_data"))
        })
        .filter_map(|s| s.size)
        .sum();
    let primary_gguf_bytes = files
        .iter()
        .filter(|f| !f.is_mmproj && f.format == "GGUF")
        .filter_map(|f| f.size_bytes)
        .filter(|&sz| sz > 5_000_000)
        .min()
        .or_else(|| {
            files
                .iter()
                .filter(|f| !f.is_mmproj && f.format == "GGUF")
                .filter_map(|f| f.size_bytes)
                .min()
        })
        .or_else(|| {
            entry
                .gguf
                .as_ref()
                .and_then(|g| g.get("total"))
                .and_then(|v| v.as_u64())
                .filter(|&b| b > 0)
        });

    let total_weight_bytes = if safetensor_bytes > 0 {
        Some(safetensor_bytes)
    } else if weight_bin_bytes > 0 {
        Some(weight_bin_bytes)
    } else if let Some(gb) = primary_gguf_bytes {
        Some(gb)
    } else if onnx_bytes > 0 {
        Some(onnx_bytes)
    } else {
        files
            .iter()
            .filter(|f| !f.is_mmproj && f.format != "Other")
            .filter_map(|f| f.size_bytes)
            .filter(|&sz| sz > 0)
            .min()
    };

    let estimated_gb = total_weight_bytes
        .map(|b| (b as f64) / 1_073_741_824.0)
        .or_else(|| {
            param_b.map(|b| {
                if format_tags.iter().any(|f| f == "GGUF") {
                    b * 0.72
                } else {
                    b * 2.0
                }
            })
        });
    let size_label = total_weight_bytes
        .map(format_size_label)
        .or_else(|| estimated_gb.map(|g| format_size_label((g * 1_073_741_824.0) as u64)));

    let size_over_4gb = estimated_gb.map(|g| g > 4.0).unwrap_or(false);
    let param_over_4b = param_b.map(|b| b > 4.05).unwrap_or(false);
    let over_4gb = size_over_4gb || param_over_4b;

    let module_hint = infer_module_hint(entry.pipeline_tag.as_deref(), &entry.tags, &repo_id);
    let summary = summary_override
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            let pt = entry.pipeline_tag.as_deref().unwrap_or("open-weights");
            format!("{pt} · {}", format_tags.join(" / "))
        });

    Some(HfHubModel {
        repo_id,
        name,
        author,
        module_hint,
        format_tags,
        downloads: entry.downloads.unwrap_or(0),
        likes: entry.likes.unwrap_or(0),
        trending_score: entry.trending_score.unwrap_or(0.0),
        last_modified: entry.last_modified.unwrap_or_default(),
        param_label,
        estimated_gb,
        total_weight_bytes,
        size_label,
        over_4gb,
        is_pinned,
        summary,
        files,
    })
}

pub fn curated_pinned_presets() -> Vec<HfHubModel> {
    vec![
        HfHubModel {
            repo_id: "bartowski/Dolphin3.0-Llama3.2-1B-GGUF".into(),
            name: "Dolphin3.0-Llama3.2-1B-GGUF".into(),
            author: "bartowski".into(),
            module_hint: "text".into(),
            format_tags: vec!["GGUF".into()],
            downloads: 0,
            likes: 0,
            trending_score: 0.0,
            last_modified: String::new(),
            param_label: Some("1.0B".into()),
            estimated_gb: Some(0.81),
            total_weight_bytes: None,
            size_label: Some("830 MB".into()),
            over_4gb: false,
            is_pinned: true,
            summary: "Compact uncensored 1B GGUF model optimized for fast local Q&A and CPU inference.".into(),
            files: vec![],
        },
        HfHubModel {
            repo_id: "mradermacher/Qwen2.5-1.5B-Instruct-abliterated-GGUF".into(),
            name: "Qwen2.5-1.5B-Instruct-abliterated-GGUF".into(),
            author: "mradermacher".into(),
            module_hint: "text".into(),
            format_tags: vec!["GGUF".into()],
            downloads: 0,
            likes: 0,
            trending_score: 0.0,
            last_modified: String::new(),
            param_label: Some("1.5B".into()),
            estimated_gb: Some(0.98),
            total_weight_bytes: None,
            size_label: Some("1004 MB".into()),
            over_4gb: false,
            is_pinned: true,
            summary: "Abliterated Qwen2.5 1.5B GGUF checkpoints under 1.5 GB for fast CPU evaluation.".into(),
            files: vec![],
        },
        HfHubModel {
            repo_id: "dphn/Dolphin3.0-Llama3.2-1B".into(),
            name: "Dolphin3.0-Llama3.2-1B".into(),
            author: "dphn".into(),
            module_hint: "text".into(),
            format_tags: vec!["Safetensors".into()],
            downloads: 0,
            likes: 0,
            trending_score: 0.0,
            last_modified: String::new(),
            param_label: Some("1.0B".into()),
            estimated_gb: Some(2.36),
            total_weight_bytes: None,
            size_label: Some("2.4 GB".into()),
            over_4gb: false,
            is_pinned: true,
            summary: "LoRA/QLoRA-ready full Safetensors base for uncensored Colab Q&A fine-tuning.".into(),
            files: vec![],
        },
        HfHubModel {
            repo_id: "AtomicChat/Inkling-2-Mini-Uncensored-GGUF".into(),
            name: "Inkling-2-Mini-Uncensored-GGUF".into(),
            author: "AtomicChat".into(),
            module_hint: "text".into(),
            format_tags: vec!["GGUF".into()],
            downloads: 0,
            likes: 0,
            trending_score: 0.0,
            last_modified: String::new(),
            param_label: Some("Large".into()),
            estimated_gb: Some(8.66),
            total_weight_bytes: None,
            size_label: Some("8.7 GB".into()),
            over_4gb: true,
            is_pinned: true,
            summary: "Pinned Inkling-2 Mini uncensored GGUF release (>4GB; toggle filter to view or download).".into(),
            files: vec![],
        },
        HfHubModel {
            repo_id: "chatrlabs/Qwen-Image-2.1-Uncensored-Omni-Quantized".into(),
            name: "Qwen-Image-2.1-Uncensored-Omni-Quantized".into(),
            author: "chatrlabs".into(),
            module_hint: "image".into(),
            format_tags: vec!["GGUF".into(), "Safetensors".into()],
            downloads: 0,
            likes: 0,
            trending_score: 0.0,
            last_modified: String::new(),
            param_label: Some("NF4".into()),
            estimated_gb: Some(3.67),
            total_weight_bytes: None,
            size_label: Some("3.7 GB".into()),
            over_4gb: false,
            is_pinned: true,
            summary: "Quantized uncensored Qwen-Image 2.1 omni weights (NF4 3.67 GB and GGUF).".into(),
            files: vec![],
        },
        HfHubModel {
            repo_id: "timm/mobilenetv3_large_100.ra_in1k".into(),
            name: "mobilenetv3_large_100.ra_in1k".into(),
            author: "timm".into(),
            module_hint: "image".into(),
            format_tags: vec!["Safetensors".into()],
            downloads: 0,
            likes: 0,
            trending_score: 0.0,
            last_modified: String::new(),
            param_label: Some("5M".into()),
            estimated_gb: Some(0.02),
            total_weight_bytes: None,
            size_label: Some("22 MB".into()),
            over_4gb: false,
            is_pinned: true,
            summary: "Ultra-fast compact MobileNetV3 vision backbone for low-end image classification.".into(),
            files: vec![],
        },
        HfHubModel {
            repo_id: "openai/whisper-tiny".into(),
            name: "whisper-tiny".into(),
            author: "openai".into(),
            module_hint: "audio".into(),
            format_tags: vec!["Safetensors".into()],
            downloads: 0,
            likes: 0,
            trending_score: 0.0,
            last_modified: String::new(),
            param_label: Some("39M".into()),
            estimated_gb: Some(0.15),
            total_weight_bytes: None,
            size_label: Some("151 MB".into()),
            over_4gb: false,
            is_pinned: true,
            summary: "Open-weight Whisper-tiny multilingual ASR backbone ready for audio fine-tuning.".into(),
            files: vec![],
        },
        HfHubModel {
            repo_id: "MIT/ast-finetuned-audioset-10-10-0.4593".into(),
            name: "ast-finetuned-audioset-10-10-0.4593".into(),
            author: "MIT".into(),
            module_hint: "audio".into(),
            format_tags: vec!["Safetensors".into()],
            downloads: 0,
            likes: 0,
            trending_score: 0.0,
            last_modified: String::new(),
            param_label: Some("87M".into()),
            estimated_gb: Some(0.35),
            total_weight_bytes: None,
            size_label: Some("346 MB".into()),
            over_4gb: false,
            is_pinned: true,
            summary: "Audio Spectrogram Transformer backbone for open-weight audio event classification.".into(),
            files: vec![],
        },
    ]
}

fn build_client(hf_token: Option<&str>) -> Result<(Client, Option<String>), String> {
    let client = Client::builder()
        .user_agent("Damnisium-App/v1-beta")
        .timeout(Duration::from_secs(20))
        .build()
        .map_err(|e| e.to_string())?;
    let token = hf_token
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    Ok((client, token))
}

async fn fetch_single_entry(
    client: &Client,
    token: Option<&str>,
    repo_id: &str,
) -> Result<HfApiModelEntry, String> {
    let url = format!("https://huggingface.co/api/models/{repo_id}?blobs=true");
    let mut req = client.get(&url);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| format!("Failed inspecting {repo_id}: {e}"))?
        .error_for_status()
        .map_err(|e| format!("Hugging Face returned {e} for {repo_id}"))?;
    resp.json::<HfApiModelEntry>()
        .await
        .map_err(|e| format!("Failed parsing {repo_id} metadata: {e}"))
}

pub async fn inspect_hf_repo(
    repo_input: &str,
    hf_token: Option<&str>,
    is_pinned: bool,
) -> Result<HfHubModel, String> {
    let repo_id = normalize_repo_id(repo_input)?;
    let (client, token) = build_client(hf_token)?;
    let entry = fetch_single_entry(&client, token.as_deref(), &repo_id).await?;
    build_hub_model(entry, is_pinned, None)
        .ok_or_else(|| format!("Invalid model payload returned for {repo_id}"))
}

pub async fn fetch_uncensored_models(
    module_filter: &str,
    format_filter: &str,
    sort_by: &str,
    search_query: &str,
    hide_over_4gb: bool,
    hf_token: Option<&str>,
) -> HfHubResponse {
    let pinned_list = curated_pinned_presets();
    let pinned_map: HashMap<String, String> = pinned_list
        .iter()
        .map(|m| (m.repo_id.to_lowercase(), m.summary.clone()))
        .collect();

    let Ok((client, token)) = build_client(hf_token) else {
        return HfHubResponse {
            online: false,
            models: filter_models(
                pinned_list,
                module_filter,
                format_filter,
                search_query,
                hide_over_4gb,
            ),
            warning: Some("HTTP client initialization error; showing local presets.".into()),
        };
    };

    let mut params = vec![
        ("filter", "uncensored".to_string()),
        ("limit", "50".to_string()),
        ("expand[]", "downloads".to_string()),
        ("expand[]", "lastModified".to_string()),
        ("expand[]", "trendingScore".to_string()),
        ("expand[]", "tags".to_string()),
        ("expand[]", "pipeline_tag".to_string()),
        ("expand[]", "gguf".to_string()),
        ("expand[]", "safetensors".to_string()),
        ("expand[]", "siblings".to_string()),
    ];
    match sort_by {
        "downloads" => {
            params.push(("sort", "downloads".into()));
            params.push(("direction", "-1".into()));
        }
        "recent" => {
            params.push(("sort", "lastModified".into()));
            params.push(("direction", "-1".into()));
        }
        _ => {}
    }
    let trimmed_q = search_query.trim();
    if !trimmed_q.is_empty() {
        params.push(("search", trimmed_q.to_string()));
    }

    let qs: String = form_urlencoded::Serializer::new(String::new())
        .extend_pairs(params.iter().map(|(k, v)| (*k, v.as_str())))
        .finish();
    let url = format!("https://huggingface.co/api/models?{qs}");

    let mut req = client.get(&url);
    if let Some(t) = token.as_deref() {
        req = req.bearer_auth(t);
    }

    let entries = match req.send().await {
        Ok(resp) => match resp.error_for_status() {
            Ok(ok_resp) => match ok_resp.json::<Vec<HfApiModelEntry>>().await {
                Ok(list) => list,
                Err(e) => {
                    return HfHubResponse {
                        online: true,
                        models: filter_models(
                            pinned_list,
                            module_filter,
                            format_filter,
                            search_query,
                            hide_over_4gb,
                        ),
                        warning: Some(format!("Failed parsing HF API JSON: {e}")),
                    };
                }
            },
            Err(e) => {
                return HfHubResponse {
                    online: true,
                    models: filter_models(
                        pinned_list,
                        module_filter,
                        format_filter,
                        search_query,
                        hide_over_4gb,
                    ),
                    warning: Some(format!("HF API status error: {e}")),
                };
            }
        },
        Err(e) => {
            return HfHubResponse {
                online: false,
                models: filter_models(
                    pinned_list,
                    module_filter,
                    format_filter,
                    search_query,
                    hide_over_4gb,
                ),
                warning: Some(format!("Offline or unreachable Hugging Face Hub: {e}")),
            };
        }
    };

    let mut candidate_ids: Vec<String> = Vec::new();
    let mut initial_map: HashMap<String, HfHubModel> = HashMap::new();

    for entry in entries {
        let Some(repo_id) = entry.id.as_deref().or(entry.model_id.as_deref()) else {
            continue;
        };
        let key = repo_id.to_lowercase();
        let is_pinned = pinned_map.contains_key(&key);
        let summary_override = pinned_map.get(&key).map(|s| s.as_str());
        if let Some(model) = build_hub_model(entry, is_pinned, summary_override) {
            if !hide_over_4gb || !model.over_4gb {
                candidate_ids.push(model.repo_id.clone());
            }
            initial_map.insert(model.repo_id.to_lowercase(), model);
        }
    }

    for p in pinned_list {
        let key = p.repo_id.to_lowercase();
        if let std::collections::hash_map::Entry::Vacant(e) = initial_map.entry(key) {
            if !hide_over_4gb || !p.over_4gb {
                candidate_ids.push(p.repo_id.clone());
            }
            e.insert(p);
        }
    }

    let to_inspect: Vec<String> = candidate_ids.into_iter().take(18).collect();
    let detailed: Vec<Result<HfApiModelEntry, String>> = stream::iter(to_inspect)
        .map(|rid| {
            let c = client.clone();
            let t = token.clone();
            async move { fetch_single_entry(&c, t.as_deref(), &rid).await }
        })
        .buffer_unordered(10)
        .collect()
        .await;

    for entry in detailed.into_iter().flatten() {
        let key = entry
            .id
            .as_deref()
            .or(entry.model_id.as_deref())
            .unwrap_or("")
            .to_lowercase();
        let is_pinned = pinned_map.contains_key(&key);
        let summary_override = pinned_map.get(&key).map(|s| s.as_str());
        if let Some(refined) = build_hub_model(entry, is_pinned, summary_override) {
            initial_map.insert(key, refined);
        }
    }

    let mut combined: Vec<HfHubModel> = initial_map.into_values().collect();
    match sort_by {
        "downloads" => combined.sort_by(|a, b| {
            b.is_pinned
                .cmp(&a.is_pinned)
                .then(b.downloads.cmp(&a.downloads))
        }),
        "recent" => combined.sort_by(|a, b| {
            b.is_pinned
                .cmp(&a.is_pinned)
                .then(b.last_modified.cmp(&a.last_modified))
        }),
        _ => combined.sort_by(|a, b| {
            b.is_pinned
                .cmp(&a.is_pinned)
                .then(
                    b.trending_score
                        .partial_cmp(&a.trending_score)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        }),
    }

    HfHubResponse {
        online: true,
        models: filter_models(
            combined,
            module_filter,
            format_filter,
            search_query,
            hide_over_4gb,
        ),
        warning: None,
    }
}

fn filter_models(
    models: Vec<HfHubModel>,
    module_filter: &str,
    format_filter: &str,
    search_query: &str,
    hide_over_4gb: bool,
) -> Vec<HfHubModel> {
    let mod_q = module_filter.trim().to_lowercase();
    let fmt_q = format_filter.trim().to_lowercase();
    let kw = search_query.trim().to_lowercase();

    models
        .into_iter()
        .filter(|m| {
            if hide_over_4gb && m.over_4gb {
                return false;
            }
            if !mod_q.is_empty() && mod_q != "all" && m.module_hint != mod_q {
                return false;
            }
            if !fmt_q.is_empty()
                && fmt_q != "all"
                && !m
                    .format_tags
                    .iter()
                    .any(|f| f.eq_ignore_ascii_case(&fmt_q))
            {
                return false;
            }
            if !kw.is_empty()
                && !m.repo_id.to_lowercase().contains(&kw)
                && !m.summary.to_lowercase().contains(&kw)
                && !m.module_hint.to_lowercase().contains(&kw)
            {
                return false;
            }
            true
        })
        .collect()
}

pub async fn download_hf_file(
    app: AppHandle,
    storage: &StorageManager,
    repo_input: &str,
    rfilename: &str,
    hf_token: Option<&str>,
) -> Result<PathBuf, String> {
    let repo_id = normalize_repo_id(repo_input)?;
    let clean_rel = rfilename.trim().trim_start_matches('/');
    if clean_rel.is_empty() || clean_rel.contains("..") || clean_rel.contains('\\') {
        return Err("Invalid model filename.".into());
    }

    let safe_repo = repo_id.replace('/', "--");
    let leaf_name = PathBuf::from(clean_rel)
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| "Invalid filename.".to_string())?
        .to_string();

    let is_gguf = leaf_name.to_lowercase().ends_with(".gguf");
    let dest_dir = if is_gguf {
        storage.data_dir().join("models").join("gguf").join(&safe_repo)
    } else {
        storage
            .data_dir()
            .join("models")
            .join("custom")
            .join(&safe_repo)
    };
    tokio::fs::create_dir_all(&dest_dir)
        .await
        .map_err(|e| e.to_string())?;
    let dest_path = dest_dir.join(&leaf_name);
    let tmp_path = dest_dir.join(format!("{leaf_name}.part"));

    let url = format!("https://huggingface.co/{repo_id}/resolve/main/{clean_rel}");
    let client = Client::builder()
        .user_agent("Damnisium-App/v1-beta")
        .timeout(Duration::from_secs(3600))
        .build()
        .map_err(|e| e.to_string())?;

    let mut req = client.get(&url);
    if let Some(tok) = hf_token.map(|s| s.trim()).filter(|s| !s.is_empty()) {
        req = req.bearer_auth(tok);
    }

    let resp = req
        .send()
        .await
        .map_err(|e| format!("Failed downloading {leaf_name}: {e}"))?
        .error_for_status()
        .map_err(|e| format!("Hugging Face download error: {e}"))?;

    let total_bytes = resp.content_length();
    let mut downloaded_bytes: u64 = 0;
    let mut file = tokio::fs::File::create(&tmp_path)
        .await
        .map_err(|e| e.to_string())?;
    let mut stream = resp.bytes_stream();
    let mut last_emit = 0u64;

    while let Some(chunk) = stream.next().await {
        let bytes = match chunk {
            Ok(b) => b,
            Err(e) => {
                drop(file);
                let _ = tokio::fs::remove_file(&tmp_path).await;
                return Err(format!("Stream read error: {e}"));
            }
        };
        if let Err(e) = tokio::io::AsyncWriteExt::write_all(&mut file, &bytes).await {
            drop(file);
            let _ = tokio::fs::remove_file(&tmp_path).await;
            return Err(e.to_string());
        }
        downloaded_bytes += bytes.len() as u64;
        if downloaded_bytes - last_emit >= 256 * 1024 {
            last_emit = downloaded_bytes;
            let _ = app.emit(
                "hf-download-progress",
                HfDownloadProgressPayload {
                    repo_id: repo_id.clone(),
                    filename: leaf_name.clone(),
                    downloaded_bytes,
                    total_bytes,
                    done: false,
                },
            );
        }
    }
    if let Err(e) = tokio::io::AsyncWriteExt::flush(&mut file).await {
        drop(file);
        let _ = tokio::fs::remove_file(&tmp_path).await;
        return Err(e.to_string());
    }
    drop(file);

    if downloaded_bytes == 0 {
        let _ = tokio::fs::remove_file(&tmp_path).await;
        return Err(format!("Downloaded {leaf_name} is empty (0 bytes)."));
    }
    if let Some(exp) = total_bytes {
        if downloaded_bytes != exp {
            let _ = tokio::fs::remove_file(&tmp_path).await;
            return Err(format!(
                "Incomplete download for {leaf_name}: received {downloaded_bytes} of {exp} bytes."
            ));
        }
    }
    let disk_len = tokio::fs::metadata(&tmp_path)
        .await
        .map(|m| m.len())
        .unwrap_or(0);
    if disk_len != downloaded_bytes {
        let _ = tokio::fs::remove_file(&tmp_path).await;
        return Err(format!(
            "Disk verification failed for {leaf_name}: wrote {downloaded_bytes}, found {disk_len}."
        ));
    }
    if is_gguf {
        let mut head = [0u8; 4];
        let mut chk_file = std::fs::File::open(&tmp_path).map_err(|e| e.to_string())?;
        std::io::Read::read_exact(&mut chk_file, &mut head).map_err(|e| e.to_string())?;
        if &head != b"GGUF" {
            drop(chk_file);
            let _ = tokio::fs::remove_file(&tmp_path).await;
            return Err(format!(
                "Downloaded file {leaf_name} does not have a valid GGUF header."
            ));
        }
    }

    if dest_path.exists() {
        let _ = tokio::fs::remove_file(&dest_path).await;
    }
    tokio::fs::rename(&tmp_path, &dest_path)
        .await
        .map_err(|e| e.to_string())?;

    let _ = app.emit(
        "hf-download-progress",
        HfDownloadProgressPayload {
            repo_id,
            filename: leaf_name,
            downloaded_bytes,
            total_bytes: total_bytes.or(Some(downloaded_bytes)),
            done: true,
        },
    );

    Ok(dest_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_repo_id_formats() {
        assert_eq!(
            normalize_repo_id("bartowski/Dolphin3.0-Llama3.2-1B-GGUF").unwrap(),
            "bartowski/Dolphin3.0-Llama3.2-1B-GGUF"
        );
        assert_eq!(
            normalize_repo_id("https://huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF/tree/main")
                .unwrap(),
            "AtomicChat/Inkling-2-Mini-Uncensored-GGUF"
        );
        assert!(normalize_repo_id("invalid-single-segment").is_err());
    }

    #[test]
    fn test_parse_param_count_b_handles_small_and_large() {
        assert_eq!(
            parse_param_count_b("HuggingFaceTB/SmolLM2-135M-Instruct", None),
            Some(0.135)
        );
        assert_eq!(
            parse_param_count_b("HuggingFaceTB/SmolLM2-360M-Instruct", None),
            Some(0.36)
        );
        assert_eq!(
            parse_param_count_b("Qwen/Qwen2.5-1.7B-Instruct", None),
            Some(1.7)
        );
        assert_eq!(
            parse_param_count_b("bartowski/Dolphin3.0-Llama3.2-1B-GGUF", None),
            Some(1.0)
        );
        assert_eq!(
            parse_param_count_b("mradermacher/Qwen2.5-1.5B-Instruct-abliterated-GGUF", None),
            Some(1.5)
        );
        assert_eq!(
            parse_param_count_b("unsloth/Llama-3.1-8B-Instruct-GGUF", None),
            Some(8.0)
        );
        assert!(parse_param_count_b("chatrlabs/Qwen-Image-2.1-step90000", None).is_none());

        assert_eq!(format_param_label(0.135), "135M");
        assert_eq!(format_param_label(0.36), "360M");
        assert_eq!(format_param_label(1.7), "1.7B");
        assert_eq!(format_param_label(3.0), "3B");
        assert_eq!(format_size_label(285 * 1_048_576), "285 MB");
        assert_eq!(format_size_label((1.4 * 1_073_741_824.0) as u64), "1.4 GB");
    }

    #[test]
    fn test_build_hub_model_sums_sharded_safetensors_bytes() {
        let entry = HfApiModelEntry {
            id: Some("Qwen/Qwen2.5-3B-Instruct".into()),
            model_id: None,
            author: Some("Qwen".into()),
            downloads: Some(100),
            likes: Some(10),
            trending_score: Some(5.0),
            last_modified: Some("2026-01-01".into()),
            pipeline_tag: Some("text-generation".into()),
            tags: vec!["safetensors".into()],
            siblings: vec![
                HfApiSibling {
                    rfilename: "model-00001-of-00002.safetensors".into(),
                    size: Some(3_000_000_000),
                },
                HfApiSibling {
                    rfilename: "model-00002-of-00002.safetensors".into(),
                    size: Some(3_200_000_000),
                },
                HfApiSibling {
                    rfilename: "config.json".into(),
                    size: Some(1024),
                },
            ],
            gguf: None,
            safetensors: Some(serde_json::json!({"total": 3_090_000_000u64})),
        };
        let model = build_hub_model(entry, false, None).unwrap();
        assert_eq!(model.param_label.as_deref(), Some("3.1B"));
        assert_eq!(model.total_weight_bytes, Some(6_200_000_000));
        assert_eq!(model.size_label.as_deref(), Some("5.8 GB"));
    }
}
