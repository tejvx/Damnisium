use crate::constants::{autolabel_catalog, pretrained_catalog, APP_NAME, APP_VERSION};
use crate::gguf_runtime::{self, GgufServerState};
use crate::google_oauth::{self, OAuthManager};
use crate::hf_hub;
use crate::media::download_file_stream;
use crate::models::{
    AppConfig, AudioAugConfig, AudioSlice, AutoLabelModelInfo, BoundingBox, ConnectivityStatus,
    CustomModel, DatasetExportResult, EvalBundleInfo, EvalLabelsMeta, EvalMetricsMeta,
    ExportBundleResult, GgufGenerationResult, GoogleAuthStatus, HfHubModel, HfHubResponse,
    ImageAugConfig, LlamaRuntimeStatus, LocalGgufModel, PolygonShape, PreflightReport,
    PreparedOAuthSession, Project, ProjectModule, ProjectSummary, ProjectTask, SplitRatio,
    SplitType, TextAugConfig, TextEvalRow, TextItem, TrainingConfig, TrainingRun, WorkspaceInfo,
};
use crate::storage::StorageManager;
use crate::training::{self, PollerRegistry};
use chrono::Utc;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tauri::{ipc::Response, AppHandle, Emitter, Manager, State};
use tauri_plugin_opener::OpenerExt;
use uuid::Uuid;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

pub struct AppState {
    pub storage: RwLock<StorageManager>,
    pub pollers: Arc<PollerRegistry>,
    pub gguf_server: Arc<GgufServerState>,
    pub oauth: Arc<OAuthManager>,
}

fn read_storage(state: &State<'_, AppState>) -> Result<StorageManager, String> {
    state
        .storage
        .read()
        .map(|s| s.clone())
        .map_err(|_| "Storage lock poisoned.".to_string())
}

#[tauri::command]
pub fn get_workspace_info(state: State<'_, AppState>) -> Result<WorkspaceInfo, String> {
    let storage = read_storage(&state)?;
    let config = storage.load_config();
    let stored_auth = storage.load_auth().unwrap_or_default();
    let verified_email = stored_auth
        .user
        .as_ref()
        .map(|u| u.email.contains('@'))
        .unwrap_or(false);
    let connected = (!stored_auth.access_token.is_empty() || !stored_auth.refresh_token.is_empty())
        && verified_email;
    let colab_authorized = connected && google_oauth::has_colab_scope(&stored_auth.scopes);
    let auth = GoogleAuthStatus {
        connected,
        user: if connected { stored_auth.user } else { None },
        has_credentials: google_oauth::has_oauth_credentials(&config),
        drive_folder_id: stored_auth.drive_root_folder_id,
        scopes: if connected { stored_auth.scopes } else { String::new() },
        colab_authorized,
    };
    Ok(WorkspaceInfo {
        app_name: APP_NAME.to_string(),
        app_version: APP_VERSION.to_string(),
        data_dir: storage.data_dir().to_string_lossy().to_string(),
        config,
        auth,
        pretrained_models: pretrained_catalog(),
        custom_models: storage.load_custom_models(),
    })
}

#[tauri::command]
pub fn save_app_config(
    state: State<'_, AppState>,
    oauth_client_id: String,
    oauth_client_secret: String,
    hf_token: Option<String>,
    gcp_project_id: Option<String>,
    enable_drive_backup: Option<bool>,
) -> Result<AppConfig, String> {
    let storage = read_storage(&state)?;
    let mut cfg = storage.load_config();
    cfg.oauth_client_id = oauth_client_id.trim().to_string();
    cfg.oauth_client_secret = oauth_client_secret.trim().to_string();
    if let Some(tok) = hf_token {
        cfg.hf_token = tok.trim().to_string();
    }
    if let Some(proj) = gcp_project_id {
        cfg.gcp_project_id = proj.trim().to_string();
    }
    if let Some(drive_bk) = enable_drive_backup {
        cfg.enable_drive_backup = drive_bk;
    }
    storage.save_config(cfg)
}

#[tauri::command]
pub fn change_data_directory(
    app: AppHandle,
    state: State<'_, AppState>,
    new_dir: String,
) -> Result<WorkspaceInfo, String> {
    let trimmed = new_dir.trim();
    if trimmed.is_empty() {
        return Err("Target directory path cannot be empty.".into());
    }
    let path = PathBuf::from(trimmed);
    {
        let mut guard = state
            .storage
            .write()
            .map_err(|_| "Storage lock poisoned.".to_string())?;
        guard.set_data_dir(path.clone())?;
    }
    let _ = app.asset_protocol_scope().allow_directory(&path, true);
    get_workspace_info(state)
}

#[tauri::command]
pub async fn prepare_google_oauth(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<PreparedOAuthSession, String> {
    let storage = read_storage(&state)?;
    let config = storage.load_config();
    google_oauth::prepare_oauth_session(app, storage, &config, &state.oauth).await
}

#[tauri::command]
pub fn cancel_google_oauth(state: State<'_, AppState>) -> Result<(), String> {
    state.oauth.cancel();
    Ok(())
}

#[tauri::command]
pub fn disconnect_google_account(state: State<'_, AppState>) -> Result<GoogleAuthStatus, String> {
    state.oauth.cancel();
    let storage = read_storage(&state)?;
    storage.clear_auth()?;
    let config = storage.load_config();
    Ok(GoogleAuthStatus {
        connected: false,
        user: None,
        has_credentials: google_oauth::has_oauth_credentials(&config),
        drive_folder_id: None,
        scopes: String::new(),
        colab_authorized: false,
    })
}

#[tauri::command]
pub fn list_projects(state: State<'_, AppState>) -> Result<Vec<ProjectSummary>, String> {
    read_storage(&state)?.list_projects()
}

#[tauri::command]
pub fn create_project(
    state: State<'_, AppState>,
    name: String,
    module: ProjectModule,
    task: ProjectTask,
) -> Result<Project, String> {
    read_storage(&state)?.create_project(name, module, task)
}

#[tauri::command]
pub fn load_project(state: State<'_, AppState>, project_id: String) -> Result<Project, String> {
    read_storage(&state)?.load_project(&project_id)
}

#[tauri::command]
pub async fn delete_project(
    state: State<'_, AppState>,
    project_id: String,
) -> Result<(), String> {
    let storage = read_storage(&state)?;
    let pollers = state.pollers.clone();
    if let Ok(runs) = storage.list_runs() {
        for run in runs {
            if run.project_id == project_id {
                let _ = training::delete_training_run(&storage, &pollers, &run.id).await;
            }
        }
    }
    storage.delete_project(&project_id)
}

#[tauri::command]
pub fn add_label_class(
    state: State<'_, AppState>,
    project_id: String,
    name: String,
    color: Option<String>,
) -> Result<Project, String> {
    read_storage(&state)?.add_class(&project_id, name, color)
}

#[tauri::command]
pub fn update_label_class(
    state: State<'_, AppState>,
    project_id: String,
    class_id: String,
    name: String,
    color: String,
) -> Result<Project, String> {
    let storage = read_storage(&state)?;
    let mut project = storage.load_project(&project_id)?;
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("Class name cannot be empty.".into());
    }
    if let Some(cls) = project.classes.iter_mut().find(|c| c.id == class_id) {
        cls.name = trimmed.to_string();
        cls.color = color;
    } else {
        return Err("Class not found.".into());
    }
    storage.save_project(&project)?;
    storage.load_project(&project_id)
}

#[tauri::command]
pub fn delete_label_class(
    state: State<'_, AppState>,
    project_id: String,
    class_id: String,
) -> Result<Project, String> {
    let storage = read_storage(&state)?;
    let mut project = storage.load_project(&project_id)?;
    project.classes.retain(|c| c.id != class_id);
    for img in &mut project.image_items {
        img.boxes.retain(|b| b.class_id != class_id);
        img.polygons.retain(|p| p.class_id != class_id);
        img.image_labels.retain(|id| id != &class_id);
    }
    for aud in &mut project.audio_items {
        for slice in &mut aud.slices {
            if slice.class_id.as_deref() == Some(class_id.as_str()) {
                slice.class_id = None;
            }
        }
    }
    for txt in &mut project.text_items {
        if txt.intent_class_id.as_deref() == Some(class_id.as_str()) {
            txt.intent_class_id = None;
        }
    }
    storage.save_project(&project)?;
    storage.load_project(&project_id)
}

#[tauri::command]
pub fn import_images(
    state: State<'_, AppState>,
    project_id: String,
    file_paths: Vec<String>,
) -> Result<Project, String> {
    read_storage(&state)?.import_images(&project_id, file_paths)
}

#[tauri::command]
pub fn import_image_folder(
    state: State<'_, AppState>,
    project_id: String,
    folder_path: String,
) -> Result<Project, String> {
    let mut paths = Vec::new();
    for entry in walkdir::WalkDir::new(folder_path)
        .max_depth(3)
        .into_iter()
        .flatten()
    {
        if entry.file_type().is_file() {
            paths.push(entry.path().to_string_lossy().to_string());
        }
    }
    read_storage(&state)?.import_images(&project_id, paths)
}

#[tauri::command]
pub fn save_image_annotations(
    state: State<'_, AppState>,
    project_id: String,
    item_id: String,
    split: SplitType,
    boxes: Vec<BoundingBox>,
    polygons: Vec<PolygonShape>,
    image_labels: Vec<String>,
) -> Result<Project, String> {
    let storage = read_storage(&state)?;
    let mut project = storage.load_project(&project_id)?;
    let Some(item) = project.image_items.iter_mut().find(|i| i.id == item_id) else {
        return Err("Image item not found.".into());
    };
    item.split = split;
    item.boxes = boxes;
    item.polygons = polygons;
    item.image_labels = image_labels;
    storage.save_project(&project)?;
    storage.load_project(&project_id)
}

#[tauri::command]
pub fn delete_image_item(
    state: State<'_, AppState>,
    project_id: String,
    item_id: String,
) -> Result<Project, String> {
    let storage = read_storage(&state)?;
    let mut project = storage.load_project(&project_id)?;
    if let Some(pos) = project.image_items.iter().position(|i| i.id == item_id) {
        let removed = project.image_items.remove(pos);
        let p_media = storage.data_dir().join(&removed.rel_path);
        let p_thumb = storage.data_dir().join(&removed.thumb_rel_path);
        let _ = fs::remove_file(p_media);
        let _ = fs::remove_file(p_thumb);
    }
    storage.save_project(&project)?;
    storage.load_project(&project_id)
}

#[tauri::command]
pub fn import_audio_files(
    state: State<'_, AppState>,
    project_id: String,
    file_paths: Vec<String>,
) -> Result<Project, String> {
    read_storage(&state)?.import_audio_files(&project_id, file_paths)
}

#[tauri::command]
pub fn update_audio_metadata(
    state: State<'_, AppState>,
    project_id: String,
    item_id: String,
    duration_sec: f64,
    sample_rate: u32,
    channels: u16,
) -> Result<(), String> {
    let storage = read_storage(&state)?;
    let mut project = storage.load_project(&project_id)?;
    if let Some(item) = project.audio_items.iter_mut().find(|a| a.id == item_id) {
        item.duration_sec = duration_sec;
        item.sample_rate = sample_rate;
        item.channels = channels;
        storage.save_project(&project)?;
    }
    Ok(())
}

#[tauri::command]
pub fn save_audio_slices(
    state: State<'_, AppState>,
    project_id: String,
    item_id: String,
    split: SplitType,
    slices: Vec<AudioSlice>,
) -> Result<Project, String> {
    let storage = read_storage(&state)?;
    let mut project = storage.load_project(&project_id)?;
    let Some(item) = project.audio_items.iter_mut().find(|a| a.id == item_id) else {
        return Err("Audio item not found.".into());
    };
    item.split = split;
    item.slices = slices;
    storage.save_project(&project)?;
    storage.load_project(&project_id)
}

#[tauri::command]
pub fn delete_audio_item(
    state: State<'_, AppState>,
    project_id: String,
    item_id: String,
) -> Result<Project, String> {
    let storage = read_storage(&state)?;
    let mut project = storage.load_project(&project_id)?;
    if let Some(pos) = project.audio_items.iter().position(|a| a.id == item_id) {
        let removed = project.audio_items.remove(pos);
        let _ = fs::remove_file(storage.data_dir().join(&removed.rel_path));
    }
    storage.save_project(&project)?;
    storage.load_project(&project_id)
}

#[tauri::command]
pub fn upsert_text_item(
    state: State<'_, AppState>,
    project_id: String,
    item: TextItem,
) -> Result<Project, String> {
    let storage = read_storage(&state)?;
    let mut project = storage.load_project(&project_id)?;
    if let Some(existing) = project.text_items.iter_mut().find(|t| t.id == item.id) {
        *existing = item;
    } else {
        let mut created = item;
        if created.id.is_empty() {
            created.id = format!("txt_{}", &Uuid::new_v4().to_string().replace('-', "")[..12]);
        }
        if created.created_at.is_empty() {
            created.created_at = Utc::now().to_rfc3339();
        }
        project.text_items.push(created);
    }
    storage.save_project(&project)?;
    storage.load_project(&project_id)
}

#[tauri::command]
pub fn delete_text_item(
    state: State<'_, AppState>,
    project_id: String,
    item_id: String,
) -> Result<Project, String> {
    let storage = read_storage(&state)?;
    let mut project = storage.load_project(&project_id)?;
    project.text_items.retain(|t| t.id != item_id);
    storage.save_project(&project)?;
    storage.load_project(&project_id)
}

#[tauri::command]
pub fn preview_csv_headers(file_path: String) -> Result<Vec<String>, String> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(file_path)
        .map_err(|e| format!("Failed opening CSV: {e}"))?;
    let headers = reader.headers().map_err(|e| e.to_string())?;
    Ok(headers.iter().map(|s| s.to_string()).collect())
}

#[tauri::command]
pub fn import_text_jsonl(
    state: State<'_, AppState>,
    project_id: String,
    file_path: String,
) -> Result<Project, String> {
    read_storage(&state)?.import_text_jsonl(&project_id, &file_path)
}

#[tauri::command]
pub fn import_text_csv(
    state: State<'_, AppState>,
    project_id: String,
    file_path: String,
    prompt_col: String,
    completion_col: Option<String>,
    system_col: Option<String>,
    label_col: Option<String>,
) -> Result<Project, String> {
    read_storage(&state)?.import_text_csv(
        &project_id,
        &file_path,
        &prompt_col,
        completion_col.as_deref(),
        system_col.as_deref(),
        label_col.as_deref(),
    )
}

#[tauri::command]
pub fn export_text_dataset(
    state: State<'_, AppState>,
    project_id: String,
    format: String,
    dest_path: String,
) -> Result<usize, String> {
    read_storage(&state)?.export_text_dataset(&project_id, &format, &dest_path)
}

#[tauri::command]
pub fn update_project_settings(
    state: State<'_, AppState>,
    project_id: String,
    image_aug: ImageAugConfig,
    audio_aug: AudioAugConfig,
    text_aug: TextAugConfig,
    split_ratio: SplitRatio,
    apply_splits_now: bool,
) -> Result<Project, String> {
    let storage = read_storage(&state)?;
    let mut project = storage.load_project(&project_id)?;
    project.image_aug = image_aug;
    project.audio_aug = audio_aug;
    project.text_aug = text_aug;
    project.split_ratio = split_ratio;
    if apply_splits_now {
        storage.apply_split_ratios(&mut project);
    }
    storage.save_project(&project)?;
    storage.load_project(&project_id)
}

#[tauri::command]
pub fn export_dataset_zip(
    state: State<'_, AppState>,
    project_id: String,
    dest_path: String,
) -> Result<DatasetExportResult, String> {
    let storage = read_storage(&state)?;
    let project = storage.load_project(&project_id)?;
    storage.package_dataset_zip(&project, PathBuf::from(dest_path).as_path())
}

fn autolabel_sidecar_labels_path(onnx_path: &Path) -> PathBuf {
    let parent = onnx_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = onnx_path
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("custom");
    parent.join(format!("{stem}.labels.json"))
}

fn sanitize_autolabel_prefix(raw: &str) -> String {
    let cleaned: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    cleaned.trim_matches('_').to_string()
}

fn resolve_custom_autolabel_dest(autolabel_dir: &Path, src: &Path, fname: &str) -> PathBuf {
    let low = fname.to_ascii_lowercase();
    let mut candidate_name = fname.to_string();
    if low == "best.onnx" || low == "model.onnx" {
        let prefix = src
            .parent()
            .and_then(|p| {
                let p_name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if p_name.eq_ignore_ascii_case("weights") {
                    p.parent()
                        .and_then(|gp| gp.file_name())
                        .and_then(|n| n.to_str())
                } else if !p_name.is_empty() {
                    Some(p_name)
                } else {
                    None
                }
            })
            .map(sanitize_autolabel_prefix)
            .filter(|s| !s.is_empty());
        if let Some(pfx) = prefix {
            candidate_name = format!("{pfx}_{fname}");
        }
    }

    let candidate = autolabel_dir.join(&candidate_name);
    if candidate.exists() && candidate != src {
        let src_len = fs::metadata(src).map(|m| m.len()).unwrap_or(0);
        let dst_len = fs::metadata(&candidate).map(|m| m.len()).unwrap_or(0);
        if src_len != dst_len {
            let stem = Path::new(&candidate_name)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("custom");
            let uid = &Uuid::new_v4().to_string().replace('-', "")[..6];
            return autolabel_dir.join(format!("{stem}_{uid}.onnx"));
        }
    }
    candidate
}

fn import_custom_autolabel_onnx_with_storage(
    storage: &StorageManager,
    src: &Path,
) -> Result<AutoLabelModelInfo, String> {
    if !src.is_file() {
        return Err("ONNX file does not exist.".into());
    }
    let dir = storage.data_dir().join("models").join("autolabel");
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let fname = src
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("custom.onnx")
        .to_string();
    let dest = resolve_custom_autolabel_dest(&dir, src, &fname);
    if dest != src {
        fs::copy(src, &dest).map_err(|e| e.to_string())?;
    }
    let dest_fname = dest
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(&fname)
        .to_string();

    let mut classes = Vec::new();
    let mut input_size = 640u32;
    if let Some(src_labels) = src.parent().map(|p| p.join("labels.json")) {
        if src_labels.is_file() {
            if let Some(meta) = read_labels_meta(&src_labels) {
                classes = meta.classes;
                if let Some(sz) = meta.img_size.filter(|&s| s > 0) {
                    input_size = sz;
                }
                let sidecar = autolabel_sidecar_labels_path(&dest);
                let _ = fs::copy(&src_labels, &sidecar);
            }
        }
    }
    if classes.is_empty() {
        let sidecar = autolabel_sidecar_labels_path(&dest);
        if let Some(meta) = read_labels_meta(&sidecar) {
            classes = meta.classes;
            if let Some(sz) = meta.img_size.filter(|&s| s > 0) {
                input_size = sz;
            }
        }
    }

    let size_bytes = fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
    Ok(AutoLabelModelInfo {
        id: format!("custom_{dest_fname}"),
        name: dest_fname.clone(),
        filename: dest_fname,
        local_path: dest.to_string_lossy().to_string(),
        downloaded: true,
        size_bytes,
        input_size,
        approx_mb: (size_bytes / 1_048_576) as u32,
        is_custom: true,
        classes,
    })
}

#[tauri::command]
pub fn list_autolabel_models(state: State<'_, AppState>) -> Result<Vec<AutoLabelModelInfo>, String> {
    let storage = read_storage(&state)?;
    let dir = storage.data_dir().join("models").join("autolabel");
    let mut result = Vec::new();

    for spec in autolabel_catalog() {
        let local = dir.join(&spec.filename);
        let meta = fs::metadata(&local).ok();
        result.push(AutoLabelModelInfo {
            id: spec.id,
            name: spec.name,
            filename: spec.filename,
            local_path: local.to_string_lossy().to_string(),
            downloaded: meta.is_some(),
            size_bytes: meta.map(|m| m.len()).unwrap_or(0),
            input_size: spec.input_size,
            approx_mb: spec.approx_mb,
            is_custom: false,
            classes: vec![],
        });
    }

    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("onnx") {
                let fname = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default()
                    .to_string();
                if result.iter().any(|r| r.filename == fname) {
                    continue;
                }
                let sidecar = autolabel_sidecar_labels_path(&path);
                let labels_meta = read_labels_meta(&sidecar);
                let classes = labels_meta
                    .as_ref()
                    .map(|m| m.classes.clone())
                    .unwrap_or_default();
                let input_size = labels_meta
                    .as_ref()
                    .and_then(|m| m.img_size)
                    .filter(|&s| s > 0)
                    .unwrap_or(640);
                let size_bytes = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                result.push(AutoLabelModelInfo {
                    id: format!("custom_{fname}"),
                    name: fname.clone(),
                    filename: fname,
                    local_path: path.to_string_lossy().to_string(),
                    downloaded: true,
                    size_bytes,
                    input_size,
                    approx_mb: (size_bytes / 1_048_576) as u32,
                    is_custom: true,
                    classes,
                });
            }
        }
    }
    Ok(result)
}

#[tauri::command]
pub async fn download_autolabel_model(
    app: AppHandle,
    state: State<'_, AppState>,
    model_id: String,
) -> Result<AutoLabelModelInfo, String> {
    let storage = read_storage(&state)?;
    let spec = autolabel_catalog()
        .into_iter()
        .find(|s| s.id == model_id)
        .ok_or_else(|| format!("Unknown auto-label model '{model_id}'."))?;
    let dest = storage
        .data_dir()
        .join("models")
        .join("autolabel")
        .join(&spec.filename);
    let bytes = download_file_stream(&app, &spec.download_url, &dest, &spec.id).await?;
    Ok(AutoLabelModelInfo {
        id: spec.id,
        name: spec.name,
        filename: spec.filename,
        local_path: dest.to_string_lossy().to_string(),
        downloaded: true,
        size_bytes: bytes,
        input_size: spec.input_size,
        approx_mb: spec.approx_mb,
        is_custom: false,
        classes: vec![],
    })
}

#[tauri::command]
pub fn import_custom_autolabel_onnx(
    state: State<'_, AppState>,
    file_path: String,
) -> Result<AutoLabelModelInfo, String> {
    let storage = read_storage(&state)?;
    import_custom_autolabel_onnx_with_storage(&storage, Path::new(&file_path))
}

#[tauri::command]
pub fn read_binary_file(state: State<'_, AppState>, path: String) -> Result<Response, String> {
    let storage = read_storage(&state)?;
    let candidate = PathBuf::from(&path);
    let resolved = if candidate.is_absolute() {
        candidate
    } else {
        storage.data_dir().join(candidate)
    };
    let bytes = fs::read(&resolved)
        .map_err(|e| format!("Failed reading {}: {e}", resolved.display()))?;
    Ok(Response::new(bytes))
}

#[tauri::command]
pub fn register_custom_model(
    state: State<'_, AppState>,
    name: String,
    module: ProjectModule,
    task: ProjectTask,
    source_kind: String,
    local_weight_path: Option<String>,
    hf_repo_id: Option<String>,
) -> Result<CustomModel, String> {
    read_storage(&state)?.register_custom_model(
        name,
        module,
        task,
        source_kind,
        local_weight_path,
        hf_repo_id,
    )
}

#[tauri::command]
pub fn delete_custom_model(
    state: State<'_, AppState>,
    model_id: String,
) -> Result<Vec<CustomModel>, String> {
    read_storage(&state)?.delete_custom_model(&model_id)
}

#[tauri::command]
pub async fn start_training_run(
    app: AppHandle,
    state: State<'_, AppState>,
    config: TrainingConfig,
    offline_only: bool,
) -> Result<TrainingRun, String> {
    let storage = read_storage(&state)?;
    let pollers = state.pollers.clone();
    training::start_training_run(app, storage, pollers, config, offline_only).await
}

#[tauri::command]
pub fn list_training_runs(state: State<'_, AppState>) -> Result<Vec<TrainingRun>, String> {
    read_storage(&state)?.list_runs()
}

#[tauri::command]
pub async fn sync_training_run(
    app: AppHandle,
    state: State<'_, AppState>,
    run_id: String,
) -> Result<TrainingRun, String> {
    let storage = read_storage(&state)?;
    let updated = training::sync_run_now(&storage, &run_id).await?;
    let _ = app.emit("training-update", &updated);
    Ok(updated)
}

#[tauri::command]
pub async fn cancel_training_poll(
    app: AppHandle,
    state: State<'_, AppState>,
    run_id: String,
) -> Result<TrainingRun, String> {
    let storage = read_storage(&state)?;
    training::cancel_run_poller(&app, &storage, &state.pollers, &run_id).await
}

#[tauri::command]
pub fn rename_training_run(
    app: AppHandle,
    state: State<'_, AppState>,
    run_id: String,
    new_name: String,
) -> Result<TrainingRun, String> {
    let trimmed = new_name.trim();
    if trimmed.is_empty() {
        return Err("Model name cannot be empty.".into());
    }
    let storage = read_storage(&state)?;
    let mut run = storage.load_run(&run_id)?;
    run.model_label = trimmed.to_string();
    run.updated_at = Utc::now().to_rfc3339();
    storage.save_run(&run)?;
    let _ = app.emit("training-update", &run);
    Ok(run)
}

#[tauri::command]
pub async fn delete_training_run(
    state: State<'_, AppState>,
    run_id: String,
) -> Result<(), String> {
    let storage = read_storage(&state)?;
    training::delete_training_run(&storage, &state.pollers, &run_id).await
}

#[tauri::command]
pub fn export_run_notebook(
    state: State<'_, AppState>,
    run_id: String,
    dest_path: String,
) -> Result<String, String> {
    let storage = read_storage(&state)?;
    let run = storage.load_run(&run_id)?;
    let src = PathBuf::from(&run.local_bundle_dir).join(format!("damnisium_{run_id}.ipynb"));
    if !src.is_file() || fs::metadata(&src).map(|m| m.len() == 0).unwrap_or(true) {
        return Err("Local notebook file for this run was not found or is empty.".into());
    }
    let dest = PathBuf::from(&dest_path);
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::copy(&src, &dest).map_err(|e| format!("Failed saving notebook: {e}"))?;
    if fs::metadata(&dest).map(|m| m.len() == 0).unwrap_or(true) {
        return Err("Saved notebook file is empty.".into());
    }
    Ok(dest.to_string_lossy().to_string())
}

#[tauri::command]
pub fn open_local_path(app: AppHandle, path: String) -> Result<(), String> {
    app.opener()
        .open_path(path, None::<&str>)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn open_external_url(app: AppHandle, url: String) -> Result<(), String> {
    app.opener()
        .open_url(url, None::<&str>)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn list_colab_runtime_specs(
    state: State<'_, AppState>,
) -> Result<Vec<crate::models::ColabRuntimeSpec>, String> {
    let storage = read_storage(&state)?;
    let cfg = storage.load_config();
    let token = google_oauth::get_valid_colab_token(&storage, &cfg).await?;
    let project_id = google_oauth::resolve_gcp_project_id(&cfg);
    let mut client = crate::colab_api::ColabClient::new(token, project_id)?;
    match client.list_runtime_specs().await {
        Ok(specs) => Ok(specs),
        Err(err) if crate::colab_api::is_auth_expired_error(&err) => {
            let fresh = google_oauth::force_refresh_colab_token(&storage, &cfg).await?;
            client.set_access_token(fresh);
            client.list_runtime_specs().await
        }
        Err(err) => Err(err),
    }
}

#[tauri::command]
pub fn resolve_media_abs_path(
    state: State<'_, AppState>,
    rel_path: String,
) -> Result<String, String> {
    let storage = read_storage(&state)?;
    Ok(storage.data_dir().join(rel_path).to_string_lossy().to_string())
}

fn infer_runtime_kind(module: &ProjectModule, task: &ProjectTask) -> &'static str {
    match (module, task) {
        (ProjectModule::Image, ProjectTask::Detection) => "ort_yolo_det",
        (ProjectModule::Image, ProjectTask::Segmentation) => "ort_yolo_seg",
        (ProjectModule::Image, ProjectTask::Classification) => "ort_img_cls",
        (ProjectModule::Audio, ProjectTask::Classification) => "ort_audio_mel",
        (ProjectModule::Audio, ProjectTask::Transcription) => "hf_asr",
        (ProjectModule::Text, ProjectTask::Intent) => "hf_text_cls",
        _ => "hf_text_gen",
    }
}

fn read_labels_meta(path: &std::path::Path) -> Option<EvalLabelsMeta> {
    let bytes = fs::read(path).ok()?;
    if let Ok(meta) = serde_json::from_slice::<EvalLabelsMeta>(&bytes) {
        return Some(meta);
    }
    if let Ok(arr) = serde_json::from_slice::<Vec<String>>(&bytes) {
        return Some(EvalLabelsMeta {
            classes: arr,
            ..Default::default()
        });
    }
    None
}

fn read_metrics_meta(path: &std::path::Path) -> Option<EvalMetricsMeta> {
    let bytes = fs::read(path).ok()?;
    serde_json::from_slice::<EvalMetricsMeta>(&bytes).ok()
}

fn find_onnx_in_dir(dir: &std::path::Path) -> Option<PathBuf> {
    let preferred = ["best.onnx", "model.onnx"];
    for name in preferred {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("onnx") {
                return Some(p);
            }
        }
    }
    None
}

#[tauri::command]
pub fn list_eval_bundles(state: State<'_, AppState>) -> Result<Vec<EvalBundleInfo>, String> {
    let storage = read_storage(&state)?;
    let mut runs = storage.list_runs()?;
    runs.sort_by(|a, b| a.created_at.cmp(&b.created_at));
    let mut seen_labels: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut bundles = Vec::new();
    for run in runs {
        let wdir = PathBuf::from(&run.weights_dir);
        if !wdir.is_dir() {
            continue;
        }
        let onnx_dir_candidate = wdir.join("onnx_model");
        let onnx_file = find_onnx_in_dir(&wdir).or_else(|| {
            if onnx_dir_candidate.is_dir() {
                find_onnx_in_dir(&onnx_dir_candidate)
                    .or_else(|| find_onnx_in_dir(&onnx_dir_candidate.join("onnx")))
            } else {
                None
            }
        });
        if onnx_file.is_none() && !onnx_dir_candidate.is_dir() {
            continue;
        }
        let labels_path = wdir.join("labels.json");
        let metrics_path = wdir.join("metrics.json");
        let labels_meta = read_labels_meta(&labels_path);
        let metrics_meta = read_metrics_meta(&metrics_path);

        let runtime_kind = labels_meta
            .as_ref()
            .map(|m| m.runtime_kind.clone())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| infer_runtime_kind(&run.module, &run.task).to_string());

        let input_size = labels_meta
            .as_ref()
            .and_then(|m| m.img_size)
            .unwrap_or_else(|| {
                if run.task == ProjectTask::Classification && run.module == ProjectModule::Image {
                    224
                } else {
                    640
                }
            });

        let classes = labels_meta
            .as_ref()
            .map(|m| m.classes.clone())
            .unwrap_or_default();

        let base_label = if run
            .model_label
            .to_lowercase()
            .starts_with(&run.project_name.to_lowercase())
        {
            run.model_label.clone()
        } else {
            format!("{} · {}", run.project_name, run.model_label)
        };
        let count = seen_labels
            .entry(base_label.to_ascii_lowercase())
            .and_modify(|c| *c += 1)
            .or_insert(1);
        let display_label = if *count == 1 {
            base_label
        } else {
            format!("{base_label} (#{count})")
        };

        bundles.push(EvalBundleInfo {
            id: run.id.clone(),
            label: display_label,
            module: run.module.clone(),
            task: run.task.clone(),
            runtime_kind,
            onnx_path: onnx_file
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default(),
            onnx_dir: if onnx_dir_candidate.is_dir() {
                Some(onnx_dir_candidate.to_string_lossy().to_string())
            } else {
                None
            },
            labels_path: if labels_path.is_file() {
                Some(labels_path.to_string_lossy().to_string())
            } else {
                None
            },
            metrics_path: if metrics_path.is_file() {
                Some(metrics_path.to_string_lossy().to_string())
            } else {
                None
            },
            classes,
            input_size,
            metrics: metrics_meta,
            source: "colab_run".to_string(),
        });
    }
    bundles.reverse();

    if let Ok(ggufs) = storage.list_local_gguf_models() {
        for g in ggufs {
            bundles.push(EvalBundleInfo {
                id: g.id,
                label: g.name,
                module: ProjectModule::Text,
                task: ProjectTask::Qa,
                runtime_kind: "llama_gguf".to_string(),
                onnx_path: g.local_path,
                onnx_dir: g.mmproj_path,
                labels_path: None,
                metrics_path: None,
                classes: vec![],
                input_size: 2048,
                metrics: None,
                source: "gguf_hub".to_string(),
            });
        }
    }
    Ok(bundles)
}

#[tauri::command]
pub fn inspect_local_eval_bundle(
    app: AppHandle,
    path: String,
    module: ProjectModule,
    task: ProjectTask,
) -> Result<EvalBundleInfo, String> {
    let target = PathBuf::from(path.trim());
    if !target.exists() {
        return Err("Selected model file or bundle directory does not exist.".into());
    }

    if target.is_file()
        && target
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
    {
        let size = fs::metadata(&target).map(|m| m.len()).unwrap_or(0);
        if size == 0 {
            return Err("Selected GGUF file is empty.".into());
        }
        let mut magic = [0u8; 4];
        {
            let mut f = File::open(&target).map_err(|e| e.to_string())?;
            f.read_exact(&mut magic)
                .map_err(|_| "Selected file is too small to be a valid GGUF model.".to_string())?;
        }
        if &magic != b"GGUF" {
            return Err("Selected file does not start with a valid GGUF header.".into());
        }
        let parent = target.parent();
        if let Some(p) = parent {
            let _ = app.asset_protocol_scope().allow_directory(p, true);
        }
        let mmproj = parent.and_then(|p| {
            fs::read_dir(p).ok().and_then(|rd| {
                rd.flatten()
                    .map(|e| e.path())
                    .find(|f| {
                        f.is_file()
                            && f.extension().and_then(|e| e.to_str()) == Some("gguf")
                            && f.file_name()
                                .and_then(|n| n.to_str())
                                .map(|n| n.to_lowercase().contains("mmproj"))
                                .unwrap_or(false)
                    })
                    .map(|f| f.to_string_lossy().to_string())
            })
        });
        let label = target
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("Local GGUF")
            .to_string();
        return Ok(EvalBundleInfo {
            id: format!("gguf_{}", &Uuid::new_v4().to_string().replace('-', "")[..8]),
            label,
            module: ProjectModule::Text,
            task: ProjectTask::Qa,
            runtime_kind: "llama_gguf".to_string(),
            onnx_path: target.to_string_lossy().to_string(),
            onnx_dir: mmproj,
            labels_path: None,
            metrics_path: None,
            classes: vec![],
            input_size: 2048,
            metrics: None,
            source: "local_gguf".to_string(),
        });
    }

    let (base_dir, onnx_file, onnx_dir) = if target.is_dir() {
        let _ = app.asset_protocol_scope().allow_directory(&target, true);
        let parent = target.parent().unwrap_or(&target);
        let _ = app.asset_protocol_scope().allow_directory(parent, true);
        let found = find_onnx_in_dir(&target).or_else(|| find_onnx_in_dir(&target.join("onnx")));
        if found
            .as_ref()
            .map(|p| fs::metadata(p).map(|m| m.len() > 0).unwrap_or(false))
            != Some(true)
        {
            return Err("Selected directory does not contain a non-empty .onnx model file.".into());
        }
        (
            parent.to_path_buf(),
            found,
            Some(target.to_string_lossy().to_string()),
        )
    } else {
        if fs::metadata(&target).map(|m| m.len() == 0).unwrap_or(true) {
            return Err("Selected model file is empty.".into());
        }
        let parent = target
            .parent()
            .ok_or_else(|| "Invalid model file path.".to_string())?;
        let _ = app.asset_protocol_scope().allow_directory(parent, true);
        let sub_dir = parent.join("onnx_model");
        (
            parent.to_path_buf(),
            Some(target.clone()),
            if sub_dir.is_dir() {
                Some(sub_dir.to_string_lossy().to_string())
            } else {
                None
            },
        )
    };

    let labels_path = [target.join("labels.json"), base_dir.join("labels.json")]
        .into_iter()
        .find(|p| p.is_file());
    let metrics_path = [target.join("metrics.json"), base_dir.join("metrics.json")]
        .into_iter()
        .find(|p| p.is_file());

    let labels_meta = if let Some(lp) = labels_path.as_ref() {
        Some(
            read_labels_meta(lp)
                .ok_or_else(|| format!("Invalid labels.json format at {}.", lp.display()))?,
        )
    } else {
        None
    };
    let metrics_meta = metrics_path.as_ref().and_then(|p| read_metrics_meta(p));

    let mut classes = labels_meta
        .as_ref()
        .map(|m| m.classes.clone())
        .unwrap_or_default();

    if classes.is_empty() {
        let cfg_path = if target.is_dir() {
            target.join("config.json")
        } else {
            base_dir.join("config.json")
        };
        if cfg_path.is_file() {
            if let Ok(bytes) = fs::read(&cfg_path) {
                if let Ok(val) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                    if let Some(obj) = val.get("id2label").and_then(|v| v.as_object()) {
                        let mut pairs: Vec<(usize, String)> = obj
                            .iter()
                            .filter_map(|(k, v)| {
                                Some((k.parse::<usize>().ok()?, v.as_str()?.to_string()))
                            })
                            .collect();
                        pairs.sort_by_key(|p| p.0);
                        classes = pairs.into_iter().map(|p| p.1).collect();
                    }
                }
            }
        }
    }

    let runtime_kind = labels_meta
        .as_ref()
        .map(|m| m.runtime_kind.clone())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| infer_runtime_kind(&module, &task).to_string());

    let input_size = labels_meta
        .as_ref()
        .and_then(|m| m.img_size)
        .unwrap_or_else(|| {
            if task == ProjectTask::Classification && module == ProjectModule::Image {
                224
            } else {
                640
            }
        });

    let label = target
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("Local Model")
        .to_string();

    Ok(EvalBundleInfo {
        id: format!("local_{}", &Uuid::new_v4().to_string().replace('-', "")[..8]),
        label,
        module,
        task,
        runtime_kind,
        onnx_path: onnx_file
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default(),
        onnx_dir,
        labels_path: labels_path.map(|p| p.to_string_lossy().to_string()),
        metrics_path: metrics_path.map(|p| p.to_string_lossy().to_string()),
        classes,
        input_size,
        metrics: metrics_meta,
        source: "local".to_string(),
    })
}

#[tauri::command]
pub fn list_folder_media(
    app: AppHandle,
    folder_path: String,
    kind: String,
) -> Result<Vec<String>, String> {
    let dir = PathBuf::from(&folder_path);
    if !dir.is_dir() {
        return Err("Selected path is not a folder.".into());
    }
    let _ = app.asset_protocol_scope().allow_directory(&dir, true);
    let mut files = Vec::new();
    for entry in walkdir::WalkDir::new(&dir).max_depth(2).into_iter().flatten() {
        if !entry.file_type().is_file() {
            continue;
        }
        let ext = entry
            .path()
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        let valid = if kind == "audio" {
            matches!(ext.as_str(), "wav" | "mp3" | "ogg" | "flac" | "m4a" | "aac")
        } else {
            matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "webp" | "bmp" | "gif")
        };
        if valid {
            files.push(entry.path().to_string_lossy().to_string());
            if files.len() >= 2000 {
                break;
            }
        }
    }
    files.sort();
    Ok(files)
}

#[tauri::command]
pub fn read_text_eval_file(file_path: String) -> Result<Vec<TextEvalRow>, String> {
    let path = PathBuf::from(&file_path);
    if !path.is_file() {
        return Err("Evaluation file does not exist.".into());
    }
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let mut rows = Vec::new();

    if ext == "csv" {
        let mut reader = csv::ReaderBuilder::new()
            .flexible(true)
            .from_path(&path)
            .map_err(|e| format!("Failed reading CSV: {e}"))?;
        let headers = reader.headers().map_err(|e| e.to_string())?.clone();
        let find_col = |candidates: &[&str]| -> Option<usize> {
            headers.iter().position(|h| {
                let low = h.trim().to_lowercase();
                candidates.iter().any(|c| low == *c)
            })
        };
        let prompt_idx = find_col(&["prompt", "question", "text", "input", "utterance"]).unwrap_or(0);
        let expected_idx =
            find_col(&["expected", "answer", "completion", "output", "response", "label", "intent"])
                .unwrap_or(1);
        let sys_idx = find_col(&["system"]);

        for rec in reader.records() {
            let record = rec.map_err(|e| e.to_string())?;
            let prompt = record.get(prompt_idx).unwrap_or("").trim().to_string();
            if prompt.is_empty() {
                continue;
            }
            let expected = record.get(expected_idx).unwrap_or("").trim().to_string();
            let system = sys_idx
                .and_then(|i| record.get(i))
                .unwrap_or("")
                .trim()
                .to_string();
            rows.push(TextEvalRow {
                prompt,
                expected,
                system,
            });
        }
        return Ok(rows);
    }

    let raw = fs::read_to_string(&path).map_err(|e| e.to_string())?;
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let val: serde_json::Value =
            serde_json::from_str(trimmed).map_err(|e| format!("Invalid JSONL row: {e}"))?;
        let prompt = val
            .get("prompt")
            .or_else(|| val.get("question"))
            .or_else(|| val.get("text"))
            .or_else(|| val.get("input"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if prompt.is_empty() {
            continue;
        }
        let expected = val
            .get("expected")
            .or_else(|| val.get("answer"))
            .or_else(|| val.get("completion"))
            .or_else(|| val.get("output"))
            .or_else(|| val.get("label"))
            .or_else(|| val.get("intent"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let system = val
            .get("system")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        rows.push(TextEvalRow {
            prompt,
            expected,
            system,
        });
    }
    Ok(rows)
}

#[tauri::command]
pub async fn check_connectivity() -> Result<ConnectivityStatus, String> {
    let client = reqwest::Client::builder()
        .user_agent("Damnisium-App/v1-beta")
        .timeout(Duration::from_secs(4))
        .build()
        .map_err(|e| e.to_string())?;
    let started = Instant::now();
    let res = client
        .head("https://huggingface.co/api/models?limit=1")
        .send()
        .await;
    let checked_at = Utc::now().to_rfc3339();
    match res {
        Ok(resp) if resp.status().is_success() || resp.status().as_u16() == 429 => {
            Ok(ConnectivityStatus {
                online: true,
                latency_ms: Some(started.elapsed().as_millis() as u64),
                message: "Connected".into(),
                checked_at,
            })
        }
        _ => match client
            .head("https://www.googleapis.com/discovery/v1/apis")
            .send()
            .await
        {
            Ok(_) => Ok(ConnectivityStatus {
                online: true,
                latency_ms: Some(started.elapsed().as_millis() as u64),
                message: "Connected (Google API reachable)".into(),
                checked_at,
            }),
            Err(_) => Ok(ConnectivityStatus {
                online: false,
                latency_ms: None,
                message: "Offline Mode — local tools remain available".into(),
                checked_at,
            }),
        },
    }
}

#[tauri::command]
pub fn run_preflight_check(
    state: State<'_, AppState>,
    project_id: String,
) -> Result<PreflightReport, String> {
    let storage = read_storage(&state)?;
    let project = storage.load_project(&project_id)?;
    Ok(storage.preflight_check(&project))
}

#[tauri::command]
pub async fn fetch_uncensored_hub(
    state: State<'_, AppState>,
    module_filter: String,
    format_filter: String,
    sort_by: String,
    search_query: String,
    hide_over_4gb: bool,
) -> Result<HfHubResponse, String> {
    let cfg = read_storage(&state)?.load_config();
    Ok(hf_hub::fetch_uncensored_models(
        &module_filter,
        &format_filter,
        &sort_by,
        &search_query,
        hide_over_4gb,
        Some(cfg.hf_token.as_str()),
    )
    .await)
}

#[tauri::command]
pub async fn inspect_hf_repo(
    state: State<'_, AppState>,
    repo_input: String,
    force_refresh: Option<bool>,
) -> Result<HfHubModel, String> {
    let storage = read_storage(&state)?;
    let cfg = storage.load_config();
    let norm = hf_hub::normalize_repo_id(&repo_input)?;
    let cache_key = norm.to_lowercase();
    let force = force_refresh.unwrap_or(false);
    let mut cache = storage.load_hf_telemetry_cache();
    if !force {
        if let Some(cached) = cache.get(&cache_key) {
            if cached.total_weight_bytes.is_some() || cached.estimated_gb.is_some() {
                return Ok(cached.clone());
            }
        }
    }
    match hf_hub::inspect_hf_repo(&norm, Some(cfg.hf_token.as_str()), false).await {
        Ok(model) => {
            cache.insert(cache_key, model.clone());
            let _ = storage.save_hf_telemetry_cache(&cache);
            Ok(model)
        }
        Err(err) => {
            if let Some(cached) = cache.get(&cache_key) {
                Ok(cached.clone())
            } else {
                Err(err)
            }
        }
    }
}

#[tauri::command]
pub async fn download_hf_file(
    app: AppHandle,
    state: State<'_, AppState>,
    repo_input: String,
    rfilename: String,
) -> Result<String, String> {
    let storage = read_storage(&state)?;
    let cfg = storage.load_config();
    let dest = hf_hub::download_hf_file(
        app.clone(),
        &storage,
        &repo_input,
        &rfilename,
        Some(cfg.hf_token.as_str()),
    )
    .await?;
    if let Some(parent) = dest.parent() {
        let _ = app.asset_protocol_scope().allow_directory(parent, true);
    }
    Ok(dest.to_string_lossy().to_string())
}

#[tauri::command]
pub fn list_local_gguf_models(state: State<'_, AppState>) -> Result<Vec<LocalGgufModel>, String> {
    read_storage(&state)?.list_local_gguf_models()
}

#[tauri::command]
pub fn get_llama_runtime_status(
    state: State<'_, AppState>,
) -> Result<LlamaRuntimeStatus, String> {
    let storage = read_storage(&state)?;
    Ok(state.gguf_server.status(storage.data_dir()))
}

#[tauri::command]
pub async fn install_llama_runtime(
    state: State<'_, AppState>,
) -> Result<LlamaRuntimeStatus, String> {
    let storage = read_storage(&state)?;
    gguf_runtime::install_llama_runtime(storage.data_dir()).await
}

#[tauri::command]
pub async fn run_gguf_chat(
    state: State<'_, AppState>,
    model_path: String,
    mmproj_path: Option<String>,
    system: String,
    prompt: String,
    max_tokens: u32,
    temperature: f64,
) -> Result<GgufGenerationResult, String> {
    let storage = read_storage(&state)?;
    state
        .gguf_server
        .chat(
            storage.data_dir(),
            Path::new(&model_path),
            mmproj_path.as_deref().filter(|s| !s.is_empty()).map(Path::new),
            &system,
            &prompt,
            max_tokens,
            temperature,
        )
        .await
}

#[tauri::command]
pub fn unload_gguf_runtime(state: State<'_, AppState>) -> Result<(), String> {
    state.gguf_server.unload();
    Ok(())
}

fn build_inference_snippets(runtime_kind: &str, model_filename: &str) -> (String, String) {
    match runtime_kind {
        "llama_gguf" => (
            format!(
                r#"from llama_cpp import Llama

llm = Llama(model_path="{model_filename}", n_ctx=2048, n_threads=4)
out = llm.create_chat_completion(
    messages=[
        {{"role": "system", "content": "Be helpful and concise."}},
        {{"role": "user", "content": "Explain this dataset."}}
    ],
    max_tokens=256,
    temperature=0.7
)
print(out["choices"][0]["message"]["content"])
"#
            ),
            format!(
                r#"use std::process::Command;

fn main() {{
    let status = Command::new("llama-server.exe")
        .args(["-m", "{model_filename}", "--port", "8080", "-c", "2048"])
        .status()
        .expect("failed running llama-server.exe");
    println!("Exited with: {{status}}");
}}
"#
            ),
        ),
        "ort_yolo_det" | "ort_yolo_seg" => (
            format!(
                r#"import json
import numpy as np
import onnxruntime as ort
from PIL import Image

with open("labels.json", "r", encoding="utf-8") as f:
    meta = json.load(f)
classes = meta.get("classes", meta) if isinstance(meta, dict) else meta
size = meta.get("imgSize", 640) if isinstance(meta, dict) else 640

img = Image.open("sample.jpg").convert("RGB").resize((size, size))
arr = (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
sess = ort.InferenceSession("{model_filename}", providers=["CPUExecutionProvider"])
outputs = sess.run(None, {{sess.get_inputs()[0].name: arr}})
print(outputs[0].shape, classes)
"#
            ),
            format!(
                r#"use ort::session::Session;
use ort::value::Tensor;

fn main() -> ort::Result<()> {{
    let session = Session::builder()?.commit_from_file("{model_filename}")?;
    let input = Tensor::<f32>::from_array(([1usize, 3, 640, 640], vec![0.0f32; 3 * 640 * 640]))?;
    let outputs = session.run(ort::inputs![input]?)?;
    println!("Outputs count: {{}}", outputs.len());
    Ok(())
}}
"#
            ),
        ),
        "ort_img_cls" | "ort_audio_mel" => (
            format!(
                r#"import json
import numpy as np
import onnxruntime as ort

with open("labels.json", "r", encoding="utf-8") as f:
    meta = json.load(f)
classes = meta.get("classes", meta) if isinstance(meta, dict) else meta

sess = ort.InferenceSession("{model_filename}", providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
dummy = np.zeros(shape, dtype=np.float32)
logits = sess.run(None, {{inp.name: dummy}})[0][0]
idx = int(np.argmax(logits))
print(classes[idx] if idx < len(classes) else idx, float(logits[idx]))
"#
            ),
            format!(
                r#"use ort::session::Session;
use ort::value::Tensor;

fn main() -> ort::Result<()> {{
    let session = Session::builder()?.commit_from_file("{model_filename}")?;
    let shape = [1usize, 3, 224, 224];
    let n: usize = shape.iter().product();
    let input = Tensor::<f32>::from_array((shape, vec![0.0f32; n]))?;
    let outputs = session.run(ort::inputs![input]?)?;
    println!("Predictions tensor ready: {{}}", outputs.len());
    Ok(())
}}
"#
            ),
        ),
        _ => (
            r#"import json
from transformers import pipeline

with open("labels.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

task = "text-classification" if meta.get("runtimeKind") == "hf_text_cls" else "text-generation"
pipe = pipeline(task, model="onnx_model")
print(pipe("Sample prompt"))
"#
            .to_string(),
            r#"use std::fs;

fn main() {
    let labels = fs::read_to_string("labels.json").expect("missing labels.json");
    println!("Loaded Damnisium ONNX metadata:\n{labels}");
}
"#
            .to_string(),
        ),
    }
}

#[tauri::command]
pub fn export_model_bundle(
    state: State<'_, AppState>,
    onnx_path: String,
    labels_path: Option<String>,
    metrics_path: Option<String>,
    dest_dir: String,
    runtime_kind: String,
    as_zip: bool,
) -> Result<ExportBundleResult, String> {
    let _storage = read_storage(&state)?;
    let model_src = PathBuf::from(onnx_path.trim());
    if !model_src.is_file() {
        return Err("Active model weight file not found on disk.".into());
    }
    if fs::metadata(&model_src).map(|m| m.len() == 0).unwrap_or(true) {
        return Err("Active model weight file is empty (0 bytes).".into());
    }
    if model_src
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false)
    {
        let mut magic = [0u8; 4];
        let mut f = File::open(&model_src).map_err(|e| e.to_string())?;
        f.read_exact(&mut magic)
            .map_err(|_| "Model GGUF header could not be verified.".to_string())?;
        if &magic != b"GGUF" {
            return Err("Source file failed GGUF header verification.".into());
        }
    }
    if let Some(lp) = labels_path.as_deref().map(Path::new).filter(|p| p.is_file()) {
        if read_labels_meta(lp).is_none() {
            return Err(format!("labels.json failed JSON validation at {}.", lp.display()));
        }
    }
    if let Some(mp) = metrics_path.as_deref().map(Path::new).filter(|p| p.is_file()) {
        if read_metrics_meta(mp).is_none() {
            return Err(format!("metrics.json failed JSON validation at {}.", mp.display()));
        }
    }
    let target_root = PathBuf::from(dest_dir.trim());
    fs::create_dir_all(&target_root).map_err(|e| e.to_string())?;

    let model_filename = model_src
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model.onnx")
        .to_string();
    let (py_snippet, rs_snippet) = build_inference_snippets(&runtime_kind, &model_filename);

    let stem = model_src
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("damnisium_model")
        .replace(' ', "_");
    let stamp = Utc::now().format("%Y%m%d_%H%M%S");
    let bundle_name = format!("{stem}_{stamp}");

    let mut candidate_files: Vec<(String, PathBuf)> = vec![(model_filename.clone(), model_src.clone())];
    if let Some(lp) = labels_path.as_deref().map(Path::new).filter(|p| p.is_file()) {
        candidate_files.push(("labels.json".into(), lp.to_path_buf()));
    }
    if let Some(mp) = metrics_path.as_deref().map(Path::new).filter(|p| p.is_file()) {
        candidate_files.push(("metrics.json".into(), mp.to_path_buf()));
    }
    if let Some(parent) = model_src.parent() {
        let mut onnx_dir_opt = if parent.join("onnx_model").is_dir() {
            Some(parent.join("onnx_model"))
        } else {
            None
        };
        if onnx_dir_opt.is_none() {
            let mut cur = Some(parent);
            for _ in 0..2 {
                if let Some(dir_cand) = cur {
                    if dir_cand
                        .file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.eq_ignore_ascii_case("onnx_model"))
                        .unwrap_or(false)
                    {
                        onnx_dir_opt = Some(dir_cand.to_path_buf());
                        break;
                    }
                    cur = dir_cand.parent();
                }
            }
        }
        if let Some(onnx_dir) = onnx_dir_opt {
            if let Some(strip_root) = onnx_dir.parent() {
                let model_canon = fs::canonicalize(&model_src).unwrap_or_else(|_| model_src.clone());
                for entry in walkdir::WalkDir::new(&onnx_dir).max_depth(3).into_iter().flatten() {
                    if entry.file_type().is_file() {
                        let entry_canon =
                            fs::canonicalize(entry.path()).unwrap_or_else(|_| entry.path().to_path_buf());
                        if entry_canon == model_canon {
                            continue;
                        }
                        if let Ok(rel) = entry.path().strip_prefix(strip_root) {
                            let rel_str = rel.to_string_lossy().replace('\\', "/");
                            if !candidate_files.iter().any(|(name, _)| name == &rel_str) {
                                candidate_files.push((rel_str, entry.path().to_path_buf()));
                            }
                        }
                    }
                }
            }
        }
    }

    let mut exported_names: Vec<String> = candidate_files.iter().map(|(n, _)| n.clone()).collect();
    exported_names.push("infer_python.py".into());
    exported_names.push("infer_rust.rs".into());

    if as_zip {
        let zip_path = target_root.join(format!("{bundle_name}.zip"));
        let file = File::create(&zip_path).map_err(|e| e.to_string())?;
        let mut zip = ZipWriter::new(file);
        let opts = SimpleFileOptions::default()
            .compression_method(CompressionMethod::Deflated)
            .compression_level(Some(6));

        for (rel, src) in &candidate_files {
            zip.start_file(rel, opts).map_err(|e| e.to_string())?;
            let mut src_f = File::open(src).map_err(|e| e.to_string())?;
            std::io::copy(&mut src_f, &mut zip).map_err(|e| e.to_string())?;
        }
        zip.start_file("infer_python.py", opts).map_err(|e| e.to_string())?;
        zip.write_all(py_snippet.as_bytes()).map_err(|e| e.to_string())?;
        zip.start_file("infer_rust.rs", opts).map_err(|e| e.to_string())?;
        zip.write_all(rs_snippet.as_bytes()).map_err(|e| e.to_string())?;
        let out_file = zip.finish().map_err(|e| e.to_string())?;
        out_file.sync_all().map_err(|e| e.to_string())?;
        drop(out_file);

        let verify_file = File::open(&zip_path).map_err(|e| e.to_string())?;
        let mut archive = ZipArchive::new(verify_file).map_err(|e| e.to_string())?;
        for name in &exported_names {
            let entry = archive
                .by_name(name)
                .map_err(|_| format!("Export verification failed: missing {name} inside ZIP."))?;
            if entry.size() == 0 {
                return Err(format!("Export verification failed: {name} is 0 bytes inside ZIP."));
            }
        }

        return Ok(ExportBundleResult {
            output_path: zip_path.to_string_lossy().to_string(),
            is_zip: true,
            files: exported_names,
            python_snippet: py_snippet,
            rust_snippet: rs_snippet,
        });
    }

    let bundle_dir = target_root.join(&bundle_name);
    fs::create_dir_all(&bundle_dir).map_err(|e| e.to_string())?;
    for (rel, src) in &candidate_files {
        let dest = bundle_dir.join(rel);
        if let Some(p) = dest.parent() {
            fs::create_dir_all(p).map_err(|e| e.to_string())?;
        }
        fs::copy(src, dest).map_err(|e| e.to_string())?;
    }
    fs::write(bundle_dir.join("infer_python.py"), py_snippet.as_bytes())
        .map_err(|e| e.to_string())?;
    fs::write(bundle_dir.join("infer_rust.rs"), rs_snippet.as_bytes())
        .map_err(|e| e.to_string())?;

    for name in &exported_names {
        let check_path = bundle_dir.join(name);
        if fs::metadata(&check_path).map(|m| m.len() == 0).unwrap_or(true) {
            return Err(format!("Export verification failed: {} is missing or empty.", check_path.display()));
        }
    }

    Ok(ExportBundleResult {
        output_path: bundle_dir.to_string_lossy().to_string(),
        is_zip: false,
        files: exported_names,
        python_snippet: py_snippet,
        rust_snippet: rs_snippet,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autolabel_sidecar_labels_and_meta_parsing() {
        let p = Path::new("C:/tmp/models/autolabel/run_123_best.onnx");
        assert_eq!(
            autolabel_sidecar_labels_path(p),
            PathBuf::from("C:/tmp/models/autolabel/run_123_best.labels.json")
        );

        let tmp = std::env::temp_dir().join(format!("damnisium_cmd_lbl_{}", Uuid::new_v4()));
        fs::create_dir_all(&tmp).unwrap();
        let obj_path = tmp.join("obj_labels.json");
        fs::write(
            &obj_path,
            br#"{"runtimeKind":"ort_yolo_det","classes":["cluster","defect"],"imgSize":512}"#,
        )
        .unwrap();
        let meta_obj = read_labels_meta(&obj_path).unwrap();
        assert_eq!(meta_obj.classes, vec!["cluster", "defect"]);
        assert_eq!(meta_obj.img_size, Some(512));

        let arr_path = tmp.join("arr_labels.json");
        fs::write(&arr_path, br#"["alpha","beta"]"#).unwrap();
        let meta_arr = read_labels_meta(&arr_path).unwrap();
        assert_eq!(meta_arr.classes, vec!["alpha", "beta"]);
        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn test_custom_autolabel_onnx_preserves_run_prefix_and_sidecar() {
        let tmp = std::env::temp_dir().join(format!("damnisium_cmd_imp_{}", Uuid::new_v4()));
        let mgr = StorageManager::from_data_dir(tmp.clone()).unwrap();

        let run_weights = tmp.join("runs").join("run_abc123").join("weights");
        fs::create_dir_all(&run_weights).unwrap();
        let onnx_src = run_weights.join("best.onnx");
        fs::write(&onnx_src, vec![0x42u8; 2048]).unwrap();
        fs::write(
            run_weights.join("labels.json"),
            br#"{"runtimeKind":"ort_yolo_det","classes":["cluster"],"imgSize":640}"#,
        )
        .unwrap();

        let info = import_custom_autolabel_onnx_with_storage(&mgr, &onnx_src).unwrap();
        assert_eq!(info.filename, "run_abc123_best.onnx");
        assert_eq!(info.classes, vec!["cluster".to_string()]);
        assert_eq!(info.input_size, 640);
        let sidecar = autolabel_sidecar_labels_path(Path::new(&info.local_path));
        assert!(sidecar.is_file());
        assert_eq!(read_labels_meta(&sidecar).unwrap().classes, vec!["cluster".to_string()]);

        let _ = fs::remove_dir_all(tmp);
    }
}
