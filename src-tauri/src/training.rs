use crate::colab_api::{
    format_mcp_execution_error, is_auth_expired_error, is_transient_allocation_error,
    select_best_runtime_spec, strip_ansi_codes, ColabClient, ColabRuntime, CreateRuntimeStart,
};
use crate::colab_notebook::{
    build_colab_notebook, build_headless_stages, HeadlessStagePlan, NotebookDriveSyncContext,
};
use crate::constants::{DRIVE_ROOT_FOLDER, DRIVE_RUNS_FOLDER};
use crate::google_drive;
use crate::google_oauth::{
    force_refresh_colab_token, get_valid_access_token, get_valid_colab_token,
    resolve_gcp_project_id, SCOPE_DRIVE_FILE,
};
use crate::models::{
    ColabRemoteStatus, ColabRuntimeSpec, Project, RunState, TrainingConfig, TrainingRun,
};
use crate::storage::StorageManager;
use chrono::Utc;
use reqwest::Client;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tauri::{AppHandle, Emitter};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use uuid::Uuid;

const EPOCH_CHUNK_SIZE: u32 = 5;
const MAX_CHUNK_SECS: u64 = 450;

pub struct PollerRegistry {
    pub active: Mutex<HashMap<String, JoinHandle<()>>>,
    pub cancellations: std::sync::Mutex<HashMap<String, Arc<AtomicBool>>>,
    pub launch_cooldowns: std::sync::Mutex<HashMap<String, (bool, Instant)>>,
}

impl Default for PollerRegistry {
    fn default() -> Self {
        Self {
            active: Mutex::new(HashMap::new()),
            cancellations: std::sync::Mutex::new(HashMap::new()),
            launch_cooldowns: std::sync::Mutex::new(HashMap::new()),
        }
    }
}

struct LaunchCooldownGuard<'a> {
    registry: &'a PollerRegistry,
    project_id: String,
    apply_cooldown: bool,
}

impl<'a> Drop for LaunchCooldownGuard<'a> {
    fn drop(&mut self) {
        if let Ok(mut map) = self.registry.launch_cooldowns.lock() {
            if self.apply_cooldown {
                map.insert(self.project_id.clone(), (false, Instant::now()));
            } else {
                map.remove(&self.project_id);
            }
        }
    }
}

#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct ChunkDoneMarker {
    completed_epoch: u32,
    #[allow(dead_code)]
    done: bool,
}

#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct OutputsDoneMarker {
    #[serde(default)]
    output_dir: String,
}

#[derive(Debug, Default)]
struct McpOutputMarker {
    chunk_done: Option<ChunkDoneMarker>,
    output_dir: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct SavedMetricsFile {
    #[serde(default)]
    loss_curve: Vec<crate::models::TrainingMetricPoint>,
}

fn jupyter_path_from_linux_output(
    output_dir: &str,
    remote_work_base: &str,
    fallback: &str,
) -> String {
    let trimmed = output_dir.trim().replace('\\', "/");
    if trimmed.is_empty() {
        return fallback.to_string();
    }
    let clean = trimmed.trim_matches('/');
    if remote_work_base.starts_with("content/") {
        if clean.starts_with("content/") {
            return clean.to_string();
        }
        return fallback.to_string();
    }
    if let Some(rest) = clean.strip_prefix("content/") {
        return rest.to_string();
    }
    clean.to_string()
}

pub fn resolve_unique_run_model_label(
    storage: &StorageManager,
    project_id: &str,
    cfg: &TrainingConfig,
) -> String {
    let base_raw = cfg
        .custom_model_name
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .or_else(|| {
            cfg.custom_hf_repo
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_else(|| cfg.model_id.trim());
    let base = if base_raw.is_empty() {
        "damnisium-model"
    } else {
        base_raw
    };
    let existing: Vec<String> = storage
        .list_runs()
        .unwrap_or_default()
        .into_iter()
        .filter(|r| r.project_id == project_id)
        .map(|r| r.model_label)
        .collect();
    if !existing.iter().any(|lbl| lbl.eq_ignore_ascii_case(base)) {
        return base.to_string();
    }
    let mut suffix = 2u32;
    loop {
        let candidate = format!("{base} #{suffix}");
        if !existing
            .iter()
            .any(|lbl| lbl.eq_ignore_ascii_case(&candidate))
        {
            return candidate;
        }
        suffix += 1;
    }
}

pub async fn start_training_run(
    app: AppHandle,
    storage: StorageManager,
    pollers: Arc<PollerRegistry>,
    cfg: TrainingConfig,
    offline_only: bool,
) -> Result<TrainingRun, String> {
    let mut launch_guard = if !offline_only {
        {
            let mut map = pollers
                .launch_cooldowns
                .lock()
                .map_err(|_| "Launch state lock poisoned".to_string())?;
            if let Some(&(in_flight, ts)) = map.get(&cfg.project_id) {
                if in_flight {
                    return Err(
                        "Training run initialization is already in progress for this project."
                            .into(),
                    );
                }
                let elapsed = ts.elapsed().as_secs();
                if elapsed < 8 {
                    let wait = 8 - elapsed;
                    return Err(format!(
                        "Training launch cooling down — please wait {wait}s before starting another run."
                    ));
                }
            }
            map.insert(cfg.project_id.clone(), (true, Instant::now()));
        }
        Some(LaunchCooldownGuard {
            registry: &pollers,
            project_id: cfg.project_id.clone(),
            apply_cooldown: false,
        })
    } else {
        None
    };

    let mut project = storage.load_project(&cfg.project_id)?;
    if cfg.auto_apply_split {
        storage.apply_split_ratios(&mut project);
        storage.save_project(&project)?;
    }

    let pf = storage.preflight_check(&project);
    if !pf.ok {
        let msg = pf
            .issues
            .iter()
            .find(|i| i.severity == "error")
            .map(|i| i.message.clone())
            .unwrap_or_else(|| "Dataset pre-flight check failed.".to_string());
        return Err(msg);
    }

    if !offline_only {
        let app_cfg = storage.load_config();
        let _ = get_valid_colab_token(&storage, &app_cfg).await?;
    }

    if let Some(ref mut g) = launch_guard {
        g.apply_cooldown = true;
    }

    let run_id = format!("run_{}", &Uuid::new_v4().to_string().replace('-', "")[..12]);
    let run_dir = storage.run_dir(&run_id);
    let weights_dir = run_dir.join("weights");
    fs::create_dir_all(&weights_dir).map_err(|e| e.to_string())?;

    let dataset_zip = run_dir.join("dataset.zip");
    storage.package_dataset_zip(&project, &dataset_zip)?;
    if !dataset_zip.is_file()
        || fs::metadata(&dataset_zip)
            .map(|m| m.len() == 0)
            .unwrap_or(true)
    {
        return Err("Packaged dataset.zip is missing or empty.".into());
    }

    let cfg_bytes = serde_json::to_vec_pretty(&cfg).map_err(|e| e.to_string())?;
    fs::write(run_dir.join("config.json"), &cfg_bytes).map_err(|e| e.to_string())?;

    let custom_weight_path = cfg
        .custom_weight_rel_path
        .as_deref()
        .map(|rel| storage.data_dir().join(rel))
        .filter(|p| p.is_file());

    let nb_filename = format!("damnisium_{run_id}.ipynb");
    let offline_ctx = NotebookDriveSyncContext::default();
    let offline_nb = build_colab_notebook(
        &run_id,
        &project,
        &cfg,
        custom_weight_path.is_some(),
        &offline_ctx,
    )?;
    fs::write(run_dir.join(&nb_filename), &offline_nb).map_err(|e| e.to_string())?;

    let model_label = resolve_unique_run_model_label(&storage, &project.id, &cfg);
    let now = Utc::now().to_rfc3339();
    let run = TrainingRun {
        id: run_id.clone(),
        project_id: project.id.clone(),
        project_name: project.name.clone(),
        module: project.module.clone(),
        task: project.task.clone(),
        model_id: cfg.model_id.clone(),
        model_label,
        state: if offline_only {
            RunState::WaitingForColab
        } else {
            RunState::Uploading
        },
        gpu_name: None,
        current_epoch: 0.0,
        total_epochs: cfg.epochs,
        progress_pct: 0.0,
        eta_seconds: None,
        message: if offline_only {
            "Offline bundle ready — download .ipynb or open Colab manually".into()
        } else {
            "Discovering Colab runtime hardware".into()
        },
        error: None,
        metrics: vec![],
        log_tail: vec![format!(
            "[Damnisium] Prepared local bundle for {run_id} ({} items packaged)",
            pf.total_items
        )],
        colab_url: None,
        drive_run_folder_id: None,
        drive_output_folder_id: None,
        drive_notebook_id: None,
        drive_status_file_id: None,
        drive_log_file_id: None,
        drive_output_bundle_id: None,
        colab_runtime_name: None,
        colab_session_name: None,
        colab_accelerator: cfg.preferred_accelerator.clone(),
        colab_last_execution_id: None,
        local_bundle_dir: run_dir.to_string_lossy().to_string(),
        weights_dir: weights_dir.to_string_lossy().to_string(),
        downloaded_weights: vec![],
        created_at: now.clone(),
        updated_at: now,
    };
    storage.save_run(&run)?;
    let _ = app.emit("training-update", &run);

    if offline_only {
        return Ok(run);
    }

    let cancel_flag = Arc::new(AtomicBool::new(false));
    if let Ok(mut cmap) = pollers.cancellations.lock() {
        cmap.insert(run_id.clone(), cancel_flag.clone());
    }

    let app_clone = app.clone();
    let storage_clone = storage.clone();
    let pollers_clone = pollers.clone();
    let run_clone = run.clone();
    let handle = tokio::spawn(async move {
        execute_headless_colab_job(
            app_clone,
            storage_clone,
            pollers_clone,
            run_clone,
            project,
            cfg,
            cfg_bytes,
            custom_weight_path,
            dataset_zip,
            cancel_flag,
        )
        .await;
    });
    let mut active = pollers.active.lock().await;
    active.insert(run_id, handle);
    Ok(run)
}

fn persist_and_emit(app: &AppHandle, storage: &StorageManager, run: &mut TrainingRun) {
    run.updated_at = Utc::now().to_rfc3339();
    let _ = storage.save_run(run);
    let _ = app.emit("training-update", &*run);
}

fn is_noisy_console_line(trimmed: &str) -> bool {
    if trimmed.is_empty() || trimmed.starts_with("[DAMNISIUM_") {
        return true;
    }
    trimmed.contains("0%|")
        || trimmed.contains("100%|")
        || trimmed.contains("Requesting secret HF_TOKEN")
        || trimmed.contains("Error while fetching HF_TOKEN")
        || trimmed.contains("pip's dependency resolver")
        || trimmed.contains("which is incompatible")
        || trimmed.contains("requires huggingface-hub")
}

fn append_run_logs(run: &mut TrainingRun, lines: impl IntoIterator<Item = String>) {
    for line in lines {
        let stripped = strip_ansi_codes(line.trim_end_matches(&['\r', '\n'][..]));
        let trimmed = stripped.trim();
        if is_noisy_console_line(trimmed) {
            continue;
        }
        if run.log_tail.last().map(|l| l.as_str() == trimmed).unwrap_or(false) {
            continue;
        }
        run.log_tail.push(trimmed.to_string());
    }
    if run.log_tail.len() > 600 {
        let drop_cnt = run.log_tail.len() - 400;
        run.log_tail.drain(..drop_cnt);
    }
}

fn process_mcp_line(run: &mut TrainingRun, line: &str, markers: &mut McpOutputMarker) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }
    if let Some(json_str) = trimmed.strip_prefix("[DAMNISIUM_STATUS]") {
        if let Ok(remote) = serde_json::from_str::<ColabRemoteStatus>(json_str.trim()) {
            apply_remote_status(run, remote);
            return true;
        }
        return false;
    }
    if let Some(json_str) = trimmed.strip_prefix("[DAMNISIUM_CHUNK_DONE]") {
        if let Ok(m) = serde_json::from_str::<ChunkDoneMarker>(json_str.trim()) {
            markers.chunk_done = Some(m);
        }
        return false;
    }
    if let Some(json_str) = trimmed.strip_prefix("[DAMNISIUM_OUTPUTS]") {
        if let Ok(m) = serde_json::from_str::<OutputsDoneMarker>(json_str.trim()) {
            if !m.output_dir.trim().is_empty() {
                markers.output_dir = Some(m.output_dir);
            }
        }
        return false;
    }
    append_run_logs(run, [line.to_string()]);
    false
}

fn process_mcp_output(
    run: &mut TrainingRun,
    stdout: &str,
    stderr: &str,
) -> McpOutputMarker {
    let mut markers = McpOutputMarker::default();
    for line in stdout.lines() {
        process_mcp_line(run, line, &mut markers);
    }
    append_run_logs(run, stderr.lines().map(ToString::to_string));
    markers
}

fn apply_remote_status(run: &mut TrainingRun, remote: ColabRemoteStatus) {
    if let Some(gpu) = remote.gpu {
        if !gpu.trim().is_empty() {
            run.gpu_name = Some(gpu);
        }
    }
    if let Some(ep) = remote.current_epoch {
        run.current_epoch = ep;
    }
    if let Some(tot) = remote.total_epochs {
        run.total_epochs = tot;
    }
    if let Some(pct) = remote.progress_pct {
        if pct > run.progress_pct || run.state == RunState::Training {
            run.progress_pct = pct.max(run.progress_pct);
        }
    }
    if remote.eta_seconds.is_some() {
        run.eta_seconds = remote.eta_seconds;
    }
    if let Some(msg) = remote.message {
        if !msg.trim().is_empty() {
            run.message = msg;
        }
    }
    if remote.error.is_some() {
        run.error = remote.error;
    }
    if !remote.metrics.is_empty() {
        run.metrics = remote.metrics;
    }
    match remote.state.as_str() {
        "installing" => run.state = RunState::Installing,
        "training" => run.state = RunState::Training,
        _ => {}
    }
}

async fn refresh_runtime_proxy_token(
    storage: &StorageManager,
    colab: &mut ColabClient,
    runtime: &mut ColabRuntime,
) -> Result<(), String> {
    let app_cfg = storage.load_config();
    if let Ok(tok) = get_valid_colab_token(storage, &app_cfg).await {
        colab.set_access_token(tok);
    }
    match colab.get_runtime(&runtime.name).await {
        Ok(refreshed) => {
            *runtime = refreshed;
            Ok(())
        }
        Err(err) if is_auth_expired_error(&err) => {
            let fresh = force_refresh_colab_token(storage, &app_cfg).await?;
            colab.set_access_token(fresh);
            *runtime = colab.get_runtime(&runtime.name).await?;
            Ok(())
        }
        Err(err) => Err(err),
    }
}

async fn refresh_tokens_for_stage(
    storage: &StorageManager,
    colab: &mut ColabClient,
    runtime: &mut ColabRuntime,
) -> Result<(), String> {
    let app_cfg = storage.load_config();
    let fresh_tok = get_valid_colab_token(storage, &app_cfg).await?;
    colab.set_access_token(fresh_tok);
    let _ = colab.ensure_runtime_token_fresh(runtime).await?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn allocate_or_reuse_colab_runtime(
    app: &AppHandle,
    storage: &StorageManager,
    colab: &mut ColabClient,
    app_cfg: &crate::models::AppConfig,
    cfg: &TrainingConfig,
    run: &mut TrainingRun,
    cancel_flag: &AtomicBool,
) -> Result<(ColabRuntime, bool), String> {
    let initial_bundle_line = run.log_tail.first().cloned();
    let mut spec_label = String::from("GPU");

    for attempt in 1..=6 {
        if cancel_flag.load(Ordering::Relaxed) {
            return Err("Training run cancelled by user.".into());
        }

        if attempt > 1 {
            run.progress_pct = 1.0;
            run.log_tail.clear();
            if let Some(ref first_line) = initial_bundle_line {
                run.log_tail.push(first_line.clone());
            }
            append_run_logs(
                run,
                [format!(
                    "[Damnisium] VM wait reached 15s — automatically restarting Colab run (attempt {attempt}/6)..."
                )],
            );
            run.message = format!("Restarting Colab runtime allocation (attempt {attempt}/6)...");
            persist_and_emit(app, storage, run);
            let fresh_tok = get_valid_colab_token(storage, app_cfg).await?;
            *colab = ColabClient::new(fresh_tok, resolve_gcp_project_id(app_cfg))?;
        } else {
            run.progress_pct = 1.0;
            run.message = "Checking eligible Colab hardware & active runtimes".into();
            append_run_logs(
                run,
                ["[Damnisium] Checking eligible Colab hardware & active runtimes...".to_string()],
            );
            persist_and_emit(app, storage, run);
        }

        let specs = match colab.list_runtime_specs().await {
            Ok(s) => s,
            Err(err) if is_auth_expired_error(&err) => {
                let fresh = force_refresh_colab_token(storage, app_cfg).await?;
                colab.set_access_token(fresh);
                colab.list_runtime_specs().await?
            }
            Err(err) => return Err(err),
        };

        let target_spec = select_best_runtime_spec(
            &specs,
            cfg.preferred_accelerator.as_deref(),
            cfg.runtime_variant.as_deref(),
        )
        .unwrap_or_else(|| ColabRuntimeSpec {
            variant: cfg
                .runtime_variant
                .as_deref()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or("VARIANT_GPU")
                .to_string(),
            accelerator: cfg
                .preferred_accelerator
                .as_deref()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or("T4")
                .to_string(),
            shape: "SHAPE_STANDARD".to_string(),
            eligible: true,
        });

        spec_label = if target_spec.accelerator.is_empty()
            || target_spec.accelerator.eq_ignore_ascii_case("NONE")
        {
            target_spec.variant.replace("VARIANT_", "")
        } else {
            format!(
                "{} {}",
                target_spec.variant.replace("VARIANT_", ""),
                target_spec.accelerator
            )
        };

        if let Some(rt) = colab.find_reusable_runtime(&target_spec).await {
            append_run_logs(
                run,
                [format!(
                    "[Damnisium] Reusing active Colab {spec_label} runtime: {}",
                    rt.name
                )],
            );
            persist_and_emit(app, storage, run);
            return Ok((rt, false));
        }

        run.progress_pct = 2.0;
        run.message = format!("Requesting Colab {spec_label} runtime (attempt {attempt}/6)...");
        append_run_logs(
            run,
            [format!(
                "[Damnisium] Requesting Colab {spec_label} runtime (attempt {attempt}/6)..."
            )],
        );
        persist_and_emit(app, storage, run);

        let attempt_res: Result<ColabRuntime, String> = async {
            let start = match colab.start_create_runtime(&target_spec).await {
                Ok(s) => s,
                Err(err) if is_auth_expired_error(&err) => {
                    let fresh = force_refresh_colab_token(storage, app_cfg).await?;
                    colab.set_access_token(fresh);
                    colab.start_create_runtime(&target_spec).await?
                }
                Err(err) => return Err(err),
            };

            match start {
                CreateRuntimeStart::Ready(rt) => Ok(rt),
                CreateRuntimeStart::Operation(mut op) => {
                    let started = Instant::now();
                    let mut last_ui_emit = Instant::now();
                    let mut logged_mid_wait = false;
                    let mut probed_reuse = false;
                    while !op.done.unwrap_or(false) {
                        if cancel_flag.load(Ordering::Relaxed) {
                            return Err("Training run cancelled during runtime allocation.".into());
                        }
                        let elapsed = started.elapsed().as_secs();
                        if elapsed >= 15 {
                            return Err("DEADLINE_EXCEEDED: 15s fast restart threshold".into());
                        }
                        if last_ui_emit.elapsed().as_secs() >= 3 {
                            run.progress_pct = (2.0 + (elapsed as f64 / 5.0)).min(5.0);
                            run.message = format!(
                                "Waiting for Colab {spec_label} VM... ({elapsed}s / 15s)"
                            );
                            persist_and_emit(app, storage, run);
                            last_ui_emit = Instant::now();
                        }
                        if !logged_mid_wait && elapsed >= 10 {
                            logged_mid_wait = true;
                            append_run_logs(
                                run,
                                [format!(
                                    "[Damnisium] Waiting for Colab {spec_label} VM... ({elapsed}s)"
                                )],
                            );
                            persist_and_emit(app, storage, run);
                        }
                        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                        match colab.poll_operation(&op.name).await {
                            Ok(next_op) => op = next_op,
                            Err(err) if is_auth_expired_error(&err) => {
                                let fresh = force_refresh_colab_token(storage, app_cfg).await?;
                                colab.set_access_token(fresh);
                            }
                            Err(err) => return Err(err),
                        }
                        if !op.done.unwrap_or(false) && !probed_reuse && started.elapsed().as_secs() >= 7 {
                            probed_reuse = true;
                            if let Some(rt) = colab.find_reusable_runtime(&target_spec).await {
                                append_run_logs(
                                    run,
                                    [format!(
                                        "[Damnisium] Attached to active Colab {spec_label} runtime: {}",
                                        rt.name
                                    )],
                                );
                                persist_and_emit(app, storage, run);
                                return Ok(rt);
                            }
                        }
                    }
                    colab.finalize_runtime_from_op(op).await
                }
            }
        }
        .await;

        match attempt_res {
            Ok(rt) => return Ok((rt, true)),
            Err(err) if is_transient_allocation_error(&err) => {
                if let Ok(fresh) = get_valid_colab_token(storage, app_cfg).await {
                    colab.set_access_token(fresh);
                }
                if let Some(rt) = colab.find_reusable_runtime(&target_spec).await {
                    append_run_logs(
                        run,
                        [format!(
                            "[Damnisium] Attached to provisioned Colab {spec_label} runtime: {}",
                            rt.name
                        )],
                    );
                    persist_and_emit(app, storage, run);
                    return Ok((rt, true));
                }
                if attempt < 6 {
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    continue;
                }
                return Err(format!(
                    "Colab {spec_label} allocation timed out after 6 automatic restarts ({err}). Try again shortly or pick another accelerator in Colab Runtime Hardware."
                ));
            }
            Err(err) => return Err(err),
        }
    }

    Err(format!("Colab {spec_label} allocation failed after 6 automatic restarts."))
}

#[allow(clippy::too_many_arguments)]
async fn execute_headless_colab_job(
    app: AppHandle,
    storage: StorageManager,
    pollers: Arc<PollerRegistry>,
    mut run: TrainingRun,
    project: Project,
    cfg: TrainingConfig,
    cfg_bytes: Vec<u8>,
    custom_weight_path: Option<PathBuf>,
    dataset_zip: PathBuf,
    cancel_flag: Arc<AtomicBool>,
) {
    let mut active_runtime: Option<ColabRuntime> = None;
    let mut active_session: Option<String> = None;
    let mut client_opt: Option<ColabClient> = None;

    let outcome: Result<(), String> = async {
        let app_cfg = storage.load_config();
        let access_token = get_valid_colab_token(&storage, &app_cfg).await?;
        let gcp_project = resolve_gcp_project_id(&app_cfg);
        let mut colab = ColabClient::new(access_token, gcp_project)?;

        let (mut runtime, was_created) = allocate_or_reuse_colab_runtime(
            &app,
            &storage,
            &mut colab,
            &app_cfg,
            &cfg,
            &mut run,
            &cancel_flag,
        )
        .await?;
        active_runtime = Some(runtime.clone());
        client_opt = Some(ColabClient::new(
            get_valid_colab_token(&storage, &app_cfg).await?,
            resolve_gcp_project_id(&app_cfg),
        )?);

        run.colab_runtime_name = Some(runtime.name.clone());
        if let Some(ref rs) = runtime.runtime_spec {
            let accel = if rs.accelerator.is_empty() {
                rs.variant.clone()
            } else {
                rs.accelerator.clone()
            };
            run.colab_accelerator = Some(accel.clone());
            run.gpu_name = Some(format!(
                "{} · {}",
                rs.variant.replace("VARIANT_", ""),
                accel
            ));
        }
        run.progress_pct = 6.0;
        run.message = format!(
            "{} Colab runtime {} ({})",
            if was_created {
                "Allocated"
            } else {
                "Attached to"
            },
            runtime.name,
            run.colab_accelerator.as_deref().unwrap_or("GPU")
        );
        append_run_logs(
            &mut run,
            [format!("[Damnisium] Runtime ready: {}", runtime.name)],
        );
        persist_and_emit(&app, &storage, &mut run);

        if cancel_flag.load(Ordering::Relaxed) {
            return Err("Training run cancelled by user.".into());
        }

        let remote_work_base = colab.resolve_remote_work_base(&runtime).await;
        let remote_run_dir = format!("{remote_work_base}/{}", run.id);

        run.message = "Streaming dataset.zip directly to Colab runtime".into();
        persist_and_emit(&app, &storage, &mut run);

        let mut last_pct_emit = 6.0;
        let app_prog = app.clone();
        let st_prog = storage.clone();
        let remote_ds_path = format!("{remote_run_dir}/dataset.zip");
        let upload_res = colab
            .upload_file_streaming(
                &runtime,
                &dataset_zip,
                &remote_ds_path,
                |sent, tot| {
                    let frac = if tot > 0 {
                        sent as f64 / tot as f64
                    } else {
                        1.0
                    };
                    let pct = 6.0 + frac * 10.0;
                    if pct - last_pct_emit >= 2.0 || sent >= tot {
                        last_pct_emit = pct;
                        run.progress_pct = pct;
                        run.message = format!(
                            "Streaming dataset.zip ({:.1} / {:.1} MB)",
                            sent as f64 / (1024.0 * 1024.0),
                            tot as f64 / (1024.0 * 1024.0)
                        );
                        persist_and_emit(&app_prog, &st_prog, &mut run);
                    }
                },
            )
            .await;
        if let Err(err) = upload_res {
            let low = err.to_ascii_lowercase();
            if low.contains("http 401") || low.contains("http 403") {
                refresh_runtime_proxy_token(&storage, &mut colab, &mut runtime).await?;
                colab
                    .upload_file_streaming(&runtime, &dataset_zip, &remote_ds_path, |_, _| {})
                    .await?;
            } else {
                return Err(err);
            }
        }

        let remote_cfg_path = format!("{remote_run_dir}/config.json");
        if let Err(err) = colab.upload_bytes(&runtime, &cfg_bytes, &remote_cfg_path).await {
            let low = err.to_ascii_lowercase();
            if low.contains("http 401") || low.contains("http 403") {
                refresh_runtime_proxy_token(&storage, &mut colab, &mut runtime).await?;
                colab.upload_bytes(&runtime, &cfg_bytes, &remote_cfg_path).await?;
            } else {
                return Err(err);
            }
        }
        run.progress_pct = run.progress_pct.max(18.0);
        persist_and_emit(&app, &storage, &mut run);

        if let Some(ref cw_path) = custom_weight_path {
            run.message = "Streaming custom weights to Colab runtime".into();
            persist_and_emit(&app, &storage, &mut run);
            let remote_cw_path = format!("{remote_run_dir}/custom_weight.bin");
            if let Err(err) = colab
                .upload_file_streaming(&runtime, cw_path, &remote_cw_path, |_, _| {})
                .await
            {
                let low = err.to_ascii_lowercase();
                if low.contains("http 401") || low.contains("http 403") {
                    refresh_runtime_proxy_token(&storage, &mut colab, &mut runtime).await?;
                    colab
                        .upload_file_streaming(&runtime, cw_path, &remote_cw_path, |_, _| {})
                        .await?;
                } else {
                    return Err(err);
                }
            }
        }

        if cancel_flag.load(Ordering::Relaxed) {
            return Err("Training run cancelled by user.".into());
        }

        let stages = build_headless_stages(
            &run.id,
            &project,
            &cfg,
            custom_weight_path.is_some(),
            &app_cfg.hf_token,
        )?;

        run.state = RunState::Installing;
        run.progress_pct = 20.0;
        run.message = "Opening headless Colab MCP kernel session".into();
        persist_and_emit(&app, &storage, &mut run);

        let session_id = format!("damnisium-{}", run.id.replace('_', "-"));
        let mut session_name = String::new();
        for s_try in 1..=3 {
            match colab.create_session(&runtime.name, &session_id).await {
                Ok(s) => {
                    session_name = s;
                    break;
                }
                Err(err) if is_auth_expired_error(&err) && s_try < 3 => {
                    let fresh = force_refresh_colab_token(&storage, &app_cfg).await?;
                    colab.set_access_token(fresh);
                }
                Err(err) if is_transient_allocation_error(&err) && s_try < 3 => {
                    let wait_s = s_try * 3;
                    tokio::time::sleep(std::time::Duration::from_secs(wait_s)).await;
                }
                Err(err) => return Err(err),
            }
        }
        active_session = Some(session_name.clone());
        run.colab_session_name = Some(session_name.clone());
        append_run_logs(
            &mut run,
            [format!("[Damnisium] MCP session active: {session_name}")],
        );
        persist_and_emit(&app, &storage, &mut run);

        run_mcp_stage(
            &app,
            &storage,
            &mut colab,
            &mut runtime,
            &session_name,
            &mut run,
            &stages.setup_script,
            "Setting up Colab runtime & extracting dataset",
            &cancel_flag,
        )
        .await?;

        run_mcp_stage(
            &app,
            &storage,
            &mut colab,
            &mut runtime,
            &session_name,
            &mut run,
            &stages.prepare_script,
            "Installing packages & preparing trainer",
            &cancel_flag,
        )
        .await?;
        run.progress_pct = run.progress_pct.max(25.0);
        persist_and_emit(&app, &storage, &mut run);

        let total_epochs = cfg.epochs.max(1);
        let mut next_ep = 0u32;
        while next_ep < total_epochs {
            if cancel_flag.load(Ordering::Relaxed) {
                return Err("Training run cancelled by user.".into());
            }
            let batch_end = (next_ep + EPOCH_CHUNK_SIZE).min(total_epochs);
            let chunk_msg = format!(
                "Running epoch chunk {}..{} / {total_epochs}",
                next_ep + 1,
                batch_end
            );
            run.state = RunState::Training;
            run.message = chunk_msg.clone();
            persist_and_emit(&app, &storage, &mut run);

            let chunk_code = HeadlessStagePlan::chunk_script(next_ep, batch_end, MAX_CHUNK_SECS);
            let marker = run_mcp_stage(
                &app,
                &storage,
                &mut colab,
                &mut runtime,
                &session_name,
                &mut run,
                &chunk_code,
                &chunk_msg,
                &cancel_flag,
            )
            .await?;

            next_ep = marker
                .chunk_done
                .map(|m| m.completed_epoch.max(next_ep + 1))
                .unwrap_or(batch_end);
            run.current_epoch = run.current_epoch.max(next_ep as f64);
            let floor_pct = 25.0 + (next_ep as f64 / total_epochs as f64) * 67.0;
            run.progress_pct = run.progress_pct.max(floor_pct);
            persist_and_emit(&app, &storage, &mut run);
        }

        run.state = RunState::Training;
        run.progress_pct = run.progress_pct.max(93.0);
        run.message = "Exporting ONNX model & evaluation metadata on Colab".into();
        persist_and_emit(&app, &storage, &mut run);

        let export_marker = run_mcp_stage(
            &app,
            &storage,
            &mut colab,
            &mut runtime,
            &session_name,
            &mut run,
            &stages.export_script,
            "Exporting ONNX & evaluation bundle",
            &cancel_flag,
        )
        .await?;

        if cancel_flag.load(Ordering::Relaxed) {
            return Err("Training run cancelled by user.".into());
        }

        refresh_tokens_for_stage(&storage, &mut colab, &mut runtime).await?;
        run.state = RunState::DownloadingWeights;
        run.progress_pct = run.progress_pct.max(95.5);
        run.message = "Streaming trained ONNX weights & metrics from Colab runtime".into();
        persist_and_emit(&app, &storage, &mut run);

        let weights_dir = PathBuf::from(&run.weights_dir);
        fs::create_dir_all(&weights_dir).map_err(|e| e.to_string())?;
        let default_out = format!("{remote_run_dir}/output");
        let remote_root = jupyter_path_from_linux_output(
            export_marker.output_dir.as_deref().unwrap_or(""),
            &remote_work_base,
            &default_out,
        );
        let remote_files = match colab
            .list_remote_files_recursive(&runtime, &remote_root)
            .await
        {
            Ok(list) => list,
            Err(err) if err.to_ascii_lowercase().contains("http 401") || err.to_ascii_lowercase().contains("http 403") => {
                refresh_runtime_proxy_token(&storage, &mut colab, &mut runtime).await?;
                colab
                    .list_remote_files_recursive(&runtime, &remote_root)
                    .await?
            }
            Err(err) => return Err(err),
        };
        let file_items: Vec<_> = remote_files
            .into_iter()
            .filter(|it| it.item_type != "directory")
            .collect();
        if file_items.is_empty() {
            return Err("Colab runtime finished without producing output artifacts.".into());
        }

        let clean_prefix = format!("{}/", remote_root.trim_matches('/'));
        let total_files = file_items.len();
        let mut downloaded = Vec::new();
        for (idx, item) in file_items.into_iter().enumerate() {
            let rpath = if item.path.trim().is_empty() {
                format!("{remote_root}/{}", item.name)
            } else {
                item.path.clone()
            };
            let raw_rel = rpath
                .trim_start_matches('/')
                .strip_prefix(&clean_prefix)
                .unwrap_or(&item.name);
            let rel = if raw_rel.split('/').any(|seg| seg == "..") || raw_rel.trim().is_empty() {
                item.name.as_str()
            } else {
                raw_rel
            };
            run.progress_pct = 95.5 + ((idx as f64 + 1.0) / total_files as f64) * 4.0;
            run.message = format!("Downloading {rel} ({}/{})", idx + 1, total_files);
            persist_and_emit(&app, &storage, &mut run);

            let local_dest = weights_dir.join(rel);
            if let Err(err) = colab
                .download_file_streaming(&runtime, &rpath, &local_dest)
                .await
            {
                let low = err.to_ascii_lowercase();
                if low.contains("http 401") || low.contains("http 403") {
                    refresh_runtime_proxy_token(&storage, &mut colab, &mut runtime).await?;
                    colab
                        .download_file_streaming(&runtime, &rpath, &local_dest)
                        .await?;
                } else {
                    return Err(err);
                }
            }
            if !local_dest.is_file()
                || fs::metadata(&local_dest)
                    .map(|m| m.len() == 0)
                    .unwrap_or(true)
            {
                return Err(format!(
                    "Downloaded weight artifact {} failed size verification.",
                    local_dest.display()
                ));
            }
            downloaded.push(local_dest.to_string_lossy().to_string());
        }

        if downloaded.is_empty() {
            return Err("No non-empty weight files were downloaded from Colab.".into());
        }

        let metrics_json_path = weights_dir.join("metrics.json");
        if let Ok(bytes) = fs::read(&metrics_json_path) {
            if let Ok(saved) = serde_json::from_slice::<SavedMetricsFile>(&bytes) {
                if !saved.loss_curve.is_empty() && saved.loss_curve.len() >= run.metrics.len() {
                    run.metrics = saved.loss_curve;
                }
            }
        }

        let latest_cfg = storage.load_config();
        if latest_cfg.enable_drive_backup {
            backup_outputs_to_drive_if_authorized(&storage, &latest_cfg, &mut run, &downloaded)
                .await;
        }

        run.downloaded_weights = downloaded;
        run.state = RunState::Completed;
        run.current_epoch = cfg.epochs as f64;
        run.progress_pct = 100.0;
        run.eta_seconds = Some(0);
        run.message = "Training completed and ONNX weights saved locally".into();
        persist_and_emit(&app, &storage, &mut run);
        Ok(())
    }
    .await;

    drop(client_opt);
    let was_cancelled = cancel_flag.load(Ordering::Relaxed);
    if active_session.is_some() || (was_cancelled && active_runtime.is_some()) {
        let rt_name = if was_cancelled {
            active_runtime.as_ref().map(|rt| rt.name.as_str())
        } else {
            None
        };
        cleanup_colab_resources(&storage, active_session.as_deref(), rt_name).await;
    }

    if let Err(err) = outcome {
        if run.state != RunState::Completed {
            run.state = RunState::Failed;
            run.error = Some(err.clone());
            run.message = if was_cancelled {
                "Training run cancelled".into()
            } else {
                "Headless Colab run failed".into()
            };
            let headline = err.lines().next().unwrap_or(&err);
            append_run_logs(&mut run, [format!("[Damnisium Error] {headline}")]);
            persist_and_emit(&app, &storage, &mut run);
        }
    }

    if let Ok(mut cmap) = pollers.cancellations.lock() {
        cmap.remove(&run.id);
    }
    let mut active = pollers.active.lock().await;
    active.remove(&run.id);
}

#[allow(clippy::too_many_arguments)]
async fn run_mcp_stage(
    app: &AppHandle,
    storage: &StorageManager,
    colab: &mut ColabClient,
    runtime: &mut ColabRuntime,
    session_name: &str,
    run: &mut TrainingRun,
    code: &str,
    stage_desc: &str,
    cancel_flag: &AtomicBool,
) -> Result<McpOutputMarker, String> {
    refresh_tokens_for_stage(storage, colab, runtime).await?;
    let exec_id = Uuid::new_v4().to_string();
    run.colab_last_execution_id = Some(exec_id.clone());
    persist_and_emit(app, storage, run);

    let mut streamed_buf = String::new();
    let mut streamed_any = false;
    let mut markers = McpOutputMarker::default();
    let mut last_emit = Instant::now();

    let exec_outcome = colab
        .execute_code_resumable_streaming(session_name, code, Some(&exec_id), cancel_flag, |chunk| {
            streamed_any = true;
            streamed_buf.push_str(chunk);
            let mut force_emit = false;
            while let Some(pos) = streamed_buf.find('\n') {
                let line: String = streamed_buf.drain(..=pos).collect();
                if process_mcp_line(run, &line, &mut markers) {
                    force_emit = true;
                }
            }
            if force_emit || last_emit.elapsed().as_millis() >= 500 {
                persist_and_emit(app, storage, run);
                last_emit = Instant::now();
            }
        })
        .await;

    let (_used_eid, res) = match exec_outcome {
        Ok(ok) => ok,
        Err(err) if is_auth_expired_error(&err) => {
            let app_cfg = storage.load_config();
            let fresh = force_refresh_colab_token(storage, &app_cfg).await?;
            colab.set_access_token(fresh);
            streamed_buf.clear();
            streamed_any = false;
            markers = McpOutputMarker::default();
            colab
                .execute_code_resumable_streaming(
                    session_name,
                    code,
                    Some(&exec_id),
                    cancel_flag,
                    |chunk| {
                        streamed_any = true;
                        streamed_buf.push_str(chunk);
                        let mut force_emit = false;
                        while let Some(pos) = streamed_buf.find('\n') {
                            let line: String = streamed_buf.drain(..=pos).collect();
                            if process_mcp_line(run, &line, &mut markers) {
                                force_emit = true;
                            }
                        }
                        if force_emit || last_emit.elapsed().as_millis() >= 500 {
                            persist_and_emit(app, storage, run);
                            last_emit = Instant::now();
                        }
                    },
                )
                .await?
        }
        Err(err) => return Err(err),
    };

    if !streamed_buf.trim().is_empty() {
        let tail = std::mem::take(&mut streamed_buf);
        process_mcp_line(run, &tail, &mut markers);
    }

    if !streamed_any {
        markers = process_mcp_output(run, &res.stdout, &res.stderr);
    } else {
        let mut last_remote: Option<ColabRemoteStatus> = None;
        for line in res.stdout.lines() {
            let trimmed = line.trim();
            if let Some(json_str) = trimmed.strip_prefix("[DAMNISIUM_STATUS]") {
                if let Ok(remote) = serde_json::from_str::<ColabRemoteStatus>(json_str.trim()) {
                    last_remote = Some(remote);
                }
            }
            if markers.chunk_done.is_none() {
                if let Some(json_str) = trimmed.strip_prefix("[DAMNISIUM_CHUNK_DONE]") {
                    if let Ok(m) = serde_json::from_str::<ChunkDoneMarker>(json_str.trim()) {
                        markers.chunk_done = Some(m);
                    }
                }
            }
            if markers.output_dir.is_none() {
                if let Some(json_str) = trimmed.strip_prefix("[DAMNISIUM_OUTPUTS]") {
                    if let Ok(m) = serde_json::from_str::<OutputsDoneMarker>(json_str.trim()) {
                        if !m.output_dir.trim().is_empty() {
                            markers.output_dir = Some(m.output_dir);
                        }
                    }
                }
            }
        }
        if let Some(remote) = last_remote {
            let more_metrics = remote.metrics.len() > run.metrics.len();
            let newer_epoch = remote.current_epoch.unwrap_or(0.0) > run.current_epoch;
            if more_metrics || newer_epoch {
                apply_remote_status(run, remote);
            }
        }
        let unseen_stderr: Vec<String> = res
            .stderr
            .lines()
            .map(|l| l.trim_end_matches('\r'))
            .filter(|l| !l.trim().is_empty() && !run.log_tail.iter().any(|existing| existing == l))
            .map(ToString::to_string)
            .collect();
        append_run_logs(run, unseen_stderr);
    }

    if let Some(ref res_val) = res.result {
        if !res_val.is_null() && res.stdout.trim().is_empty() && !streamed_any {
            let is_bare_num = res_val
                .as_str()
                .map(|s| s.trim().chars().all(|c| c.is_ascii_digit()))
                .unwrap_or(false)
                || res_val.is_number();
            if !is_bare_num {
                append_run_logs(run, [format!("[Colab Result] {res_val}")]);
            }
        }
    }
    if res.output_truncated {
        append_run_logs(
            run,
            ["[Damnisium] Colab truncated verbose stdout for this step.".to_string()],
        );
    }
    if let Some(ref exec_err) = res.execution_error {
        let formatted = format_mcp_execution_error(exec_err);
        persist_and_emit(app, storage, run);
        return Err(format!("{stage_desc} error: {formatted}"));
    }
    persist_and_emit(app, storage, run);
    Ok(markers)
}

async fn backup_outputs_to_drive_if_authorized(
    storage: &StorageManager,
    app_cfg: &crate::models::AppConfig,
    run: &mut TrainingRun,
    downloaded_paths: &[String],
) {
    let Ok(stored_auth) = storage.load_auth() else {
        return;
    };
    if !stored_auth
        .scopes
        .split_whitespace()
        .any(|s| s == SCOPE_DRIVE_FILE)
    {
        append_run_logs(
            run,
            ["[Damnisium] Drive backup enabled, but sign in again in Settings to grant drive.file scope.".into()],
        );
        return;
    }
    let Ok(token) = get_valid_access_token(storage, app_cfg).await else {
        return;
    };
    let client = Client::new();
    let root_id = match stored_auth.drive_root_folder_id.clone() {
        Some(id) if !id.is_empty() => id,
        _ => match google_drive::ensure_folder(&client, &token, DRIVE_ROOT_FOLDER, None).await {
            Ok(id) => id,
            Err(e) => {
                append_run_logs(run, [format!("[Damnisium] Drive backup notice: {e}")]);
                return;
            }
        },
    };
    let Ok(runs_id) =
        google_drive::ensure_folder(&client, &token, DRIVE_RUNS_FOLDER, Some(&root_id)).await
    else {
        return;
    };
    let Ok(run_fld) = google_drive::ensure_folder(&client, &token, &run.id, Some(&runs_id)).await
    else {
        return;
    };
    run.drive_run_folder_id = Some(run_fld.clone());
    for p in downloaded_paths {
        let pb = Path::new(p);
        let Some(fname) = pb.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if let Ok(fid) = google_drive::upload_file_resumable(
            &client,
            &token,
            &run_fld,
            fname,
            "application/octet-stream",
            pb,
        )
        .await
        {
            append_run_logs(
                run,
                [format!("[Damnisium] Backed up {fname} to Drive ({fid})")],
            );
        }
    }
}

pub async fn sync_run_now(storage: &StorageManager, run_id: &str) -> Result<TrainingRun, String> {
    let current = storage.load_run(run_id)?;
    let Some(folder_id) = current.drive_run_folder_id.as_deref() else {
        return Ok(current);
    };
    let Some(status_id) = current.drive_status_file_id.as_deref() else {
        return Ok(current);
    };
    if status_id.is_empty() || folder_id.is_empty() {
        return Ok(current);
    }
    let app_cfg = storage.load_config();
    let token = get_valid_access_token(storage, &app_cfg).await?;
    let client = Client::new();
    let mut run = current.clone();
    if let Ok(raw) = google_drive::download_text(&client, &token, status_id).await {
        if let Ok(remote) = serde_json::from_str::<ColabRemoteStatus>(&raw) {
            apply_remote_status(&mut run, remote);
            run.updated_at = Utc::now().to_rfc3339();
            storage.save_run(&run)?;
        }
    }
    Ok(run)
}

async fn cleanup_colab_resources(
    storage: &StorageManager,
    session_name: Option<&str>,
    runtime_name: Option<&str>,
) {
    if session_name.is_none() && runtime_name.is_none() {
        return;
    }
    let app_cfg = storage.load_config();
    let tok = match get_valid_colab_token(storage, &app_cfg).await {
        Ok(t) => t,
        Err(_) => return,
    };
    let mut client = match ColabClient::new(tok, resolve_gcp_project_id(&app_cfg)) {
        Ok(c) => c,
        Err(_) => return,
    };
    let mut refreshed = false;
    if let Some(sname) = session_name.filter(|s| !s.trim().is_empty()) {
        if let Err(err) = client.delete_session(sname).await {
            if is_auth_expired_error(&err) {
                if let Ok(fresh) = force_refresh_colab_token(storage, &app_cfg).await {
                    client.set_access_token(fresh);
                    refreshed = true;
                    let _ = client.delete_session(sname).await;
                }
            }
        }
    }
    if let Some(rname) = runtime_name.filter(|s| !s.trim().is_empty()) {
        if let Err(err) = client.delete_runtime(rname).await {
            if is_auth_expired_error(&err) && !refreshed {
                if let Ok(fresh) = force_refresh_colab_token(storage, &app_cfg).await {
                    client.set_access_token(fresh);
                    let _ = client.delete_runtime(rname).await;
                }
            }
        }
    }
}

pub async fn resume_active_pollers(
    app: AppHandle,
    storage: StorageManager,
    _pollers: Arc<PollerRegistry>,
) {
    let Ok(runs) = storage.list_runs() else {
        return;
    };
    for mut run in runs {
        let active_headless = matches!(
            run.state,
            RunState::Uploading
                | RunState::Installing
                | RunState::Training
                | RunState::DownloadingWeights
        );
        if active_headless {
            if run.colab_runtime_name.is_some() || run.colab_session_name.is_some() {
                cleanup_colab_resources(
                    &storage,
                    run.colab_session_name.as_deref(),
                    run.colab_runtime_name.as_deref(),
                )
                .await;
            }
            run.state = RunState::Failed;
            run.error = Some("Application closed while remote Colab job was active.".into());
            run.message = "Interrupted by application restart".into();
            run.updated_at = Utc::now().to_rfc3339();
            let _ = storage.save_run(&run);
            let _ = app.emit("training-update", &run);
        }
    }
}

pub async fn cancel_run_poller(
    app: &AppHandle,
    storage: &StorageManager,
    pollers: &Arc<PollerRegistry>,
    run_id: &str,
) -> Result<TrainingRun, String> {
    if let Ok(cmap) = pollers.cancellations.lock() {
        if let Some(flag) = cmap.get(run_id) {
            flag.store(true, Ordering::Relaxed);
        }
    }
    let mut active = pollers.active.lock().await;
    if let Some(handle) = active.remove(run_id) {
        handle.abort();
    }
    drop(active);

    let mut run = storage.load_run(run_id)?;
    cleanup_colab_resources(
        storage,
        run.colab_session_name.as_deref(),
        run.colab_runtime_name.as_deref(),
    )
    .await;
    if !matches!(run.state, RunState::Completed | RunState::Failed) {
        run.state = RunState::Failed;
        run.error = Some("Training run cancelled by user.".into());
        run.message = "Cancelled by user".into();
        run.updated_at = Utc::now().to_rfc3339();
        storage.save_run(&run)?;
        let _ = app.emit("training-update", &run);
    }
    Ok(run)
}

pub async fn delete_training_run(
    storage: &StorageManager,
    pollers: &Arc<PollerRegistry>,
    run_id: &str,
) -> Result<(), String> {
    if let Ok(mut cmap) = pollers.cancellations.lock() {
        if let Some(flag) = cmap.remove(run_id) {
            flag.store(true, Ordering::Relaxed);
        }
    }
    let mut active = pollers.active.lock().await;
    if let Some(handle) = active.remove(run_id) {
        handle.abort();
    }
    drop(active);

    if let Ok(run) = storage.load_run(run_id) {
        let is_active = matches!(
            run.state,
            RunState::Uploading
                | RunState::Installing
                | RunState::Training
                | RunState::DownloadingWeights
        );
        if run.colab_session_name.is_some() || (is_active && run.colab_runtime_name.is_some()) {
            let rt_name = if is_active {
                run.colab_runtime_name.as_deref()
            } else {
                None
            };
            cleanup_colab_resources(
                storage,
                run.colab_session_name.as_deref(),
                rt_name,
            )
            .await;
        }
    }

    storage.delete_run(run_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_mcp_status_and_chunk_marker() {
        let now = Utc::now().to_rfc3339();
        let mut run = TrainingRun {
            id: "run_test".into(),
            project_id: "p1".into(),
            project_name: "Demo".into(),
            module: crate::models::ProjectModule::Image,
            task: crate::models::ProjectTask::Detection,
            model_id: "yolov8n".into(),
            model_label: "yolov8n".into(),
            state: RunState::Installing,
            gpu_name: None,
            current_epoch: 0.0,
            total_epochs: 10,
            progress_pct: 0.0,
            eta_seconds: None,
            message: String::new(),
            error: None,
            metrics: vec![],
            log_tail: vec![],
            colab_url: None,
            drive_run_folder_id: None,
            drive_output_folder_id: None,
            drive_notebook_id: None,
            drive_status_file_id: None,
            drive_log_file_id: None,
            drive_output_bundle_id: None,
            colab_runtime_name: Some("runtimes/rt1".into()),
            colab_session_name: Some("runtimes/rt1/sessions/s1".into()),
            colab_accelerator: Some("T4".into()),
            colab_last_execution_id: None,
            local_bundle_dir: ".".into(),
            weights_dir: ".".into(),
            downloaded_weights: vec![],
            created_at: now.clone(),
            updated_at: now,
        };
        let stdout = "[10:00:01] Starting YOLO\n0%|          | 0.00/801 [00:00<?, ?B/s]\nRequesting secret HF_TOKEN timed out.\n[DAMNISIUM_STATUS] {\"state\":\"training\",\"gpu\":\"Tesla T4\",\"currentEpoch\":3.0,\"totalEpochs\":10,\"progressPct\":30.0,\"message\":\"Epoch 3 / 10\",\"metrics\":[]}\n[DAMNISIUM_CHUNK_DONE] {\"completedEpoch\":3,\"done\":false}\n[DAMNISIUM_OUTPUTS] {\"outputDir\":\"/content/damnisium_work/run_test/output\",\"artifacts\":[\"best.onnx\"]}\n";
        let marker = process_mcp_output(&mut run, stdout, "warn line");
        assert_eq!(run.state, RunState::Training);
        assert_eq!(run.gpu_name.as_deref(), Some("Tesla T4"));
        assert_eq!(run.current_epoch, 3.0);
        assert_eq!(marker.chunk_done.unwrap().completed_epoch, 3);
        assert_eq!(
            marker.output_dir.as_deref(),
            Some("/content/damnisium_work/run_test/output")
        );
        assert_eq!(run.log_tail.len(), 2);
        assert!(is_noisy_console_line("pip's dependency resolver does not currently"));
        assert!(!is_noisy_console_line("[12:31:02] Epoch 1/50 loss=0.4512"));
        assert_eq!(
            jupyter_path_from_linux_output(
                "/content/damnisium_work/run_test/output",
                "content/damnisium_work",
                "fallback"
            ),
            "content/damnisium_work/run_test/output"
        );
        assert_eq!(
            jupyter_path_from_linux_output(
                "/content/damnisium_work/run_test/output",
                "damnisium_work",
                "fallback"
            ),
            "damnisium_work/run_test/output"
        );

        let saved_json = r#"{"primaryName":"mAP50","lossCurve":[{"step":1,"epoch":1.0,"trainLoss":1.2,"valLoss":0.9,"score":0.4,"scoreName":"mAP50","lr":0.001},{"step":2,"epoch":2.0,"trainLoss":0.8,"valLoss":0.6,"score":0.7,"scoreName":"mAP50","lr":0.0008}]}"#;
        let parsed_saved: SavedMetricsFile = serde_json::from_str(saved_json).unwrap();
        assert_eq!(parsed_saved.loss_curve.len(), 2);
        let remote_final: ColabRemoteStatus = serde_json::from_str(
            r#"{"state":"training","currentEpoch":4.0,"progressPct":50.0,"metrics":[{"step":1,"epoch":1.0,"trainLoss":1.1,"valLoss":0.85,"score":0.42,"scoreName":"mAP50","lr":0.001}]}"#,
        )
        .unwrap();
        apply_remote_status(&mut run, remote_final);
        assert_eq!(run.current_epoch, 4.0);
        assert_eq!(run.metrics.len(), 1);
    }

    #[test]
    fn test_unique_run_model_label_deduplication() {
        let tmp = std::env::temp_dir().join(format!("damnisium_lbl_{}", Uuid::new_v4()));
        let mgr = StorageManager::from_data_dir(tmp.clone()).unwrap();
        let cfg = TrainingConfig {
            project_id: "p1".into(),
            model_id: "yolov8n".into(),
            is_custom_model: false,
            custom_hf_repo: None,
            custom_weight_rel_path: None,
            epochs: 10,
            batch_size: 8,
            learning_rate: 0.001,
            img_size: 640,
            max_seq_len: 512,
            export_onnx: true,
            auto_apply_split: true,
            resume_from_run_id: None,
            use_qlora: false,
            preferred_accelerator: None,
            runtime_variant: None,
            custom_model_name: None,
        };
        assert_eq!(resolve_unique_run_model_label(&mgr, "p1", &cfg), "yolov8n");
        let now = Utc::now().to_rfc3339();
        let make_run = |id: &str, lbl: &str| TrainingRun {
            id: id.into(),
            project_id: "p1".into(),
            project_name: "Proj".into(),
            module: crate::models::ProjectModule::Image,
            task: crate::models::ProjectTask::Detection,
            model_id: "yolov8n".into(),
            model_label: lbl.into(),
            state: RunState::Completed,
            gpu_name: None,
            current_epoch: 10.0,
            total_epochs: 10,
            progress_pct: 100.0,
            eta_seconds: Some(0),
            message: String::new(),
            error: None,
            metrics: vec![],
            log_tail: vec![],
            colab_url: None,
            drive_run_folder_id: None,
            drive_output_folder_id: None,
            drive_notebook_id: None,
            drive_status_file_id: None,
            drive_log_file_id: None,
            drive_output_bundle_id: None,
            colab_runtime_name: None,
            colab_session_name: None,
            colab_accelerator: None,
            colab_last_execution_id: None,
            local_bundle_dir: ".".into(),
            weights_dir: ".".into(),
            downloaded_weights: vec![],
            created_at: now.clone(),
            updated_at: now.clone(),
        };
        mgr.save_run(&make_run("run_1", "yolov8n")).unwrap();
        assert_eq!(resolve_unique_run_model_label(&mgr, "p1", &cfg), "yolov8n #2");
        mgr.save_run(&make_run("run_2", "yolov8n #2")).unwrap();
        assert_eq!(resolve_unique_run_model_label(&mgr, "p1", &cfg), "yolov8n #3");
        let mut named_cfg = cfg;
        named_cfg.custom_model_name = Some("bottle-detector-v1".into());
        assert_eq!(resolve_unique_run_model_label(&mgr, "p1", &named_cfg), "bottle-detector-v1");
        let _ = fs::remove_dir_all(tmp);
    }
}
