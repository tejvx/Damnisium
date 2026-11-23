use crate::constants::APP_VERSION;
use crate::models::{Project, ProjectModule, ProjectTask, TrainingConfig};
use base64::{engine::general_purpose::STANDARD as BASE64_STD, Engine as _};
use serde_json::{json, Value};

#[derive(Debug, Clone, Default)]
pub struct NotebookDriveSyncContext {
    pub run_folder_id: String,
    pub output_folder_id: String,
    pub dataset_file_id: String,
    pub custom_weight_file_id: String,
    pub status_file_id: String,
    pub log_file_id: String,
    pub output_bundle_id: String,
}

#[derive(Debug, Clone)]
pub struct HeadlessStagePlan {
    pub setup_script: String,
    pub prepare_script: String,
    pub export_script: String,
}

impl HeadlessStagePlan {
    pub fn chunk_script(start_epoch: u32, end_epoch: u32, max_chunk_secs: u64) -> String {
        format!("_ = damnisium_train_chunk({start_epoch}, {end_epoch}, {max_chunk_secs})")
    }
}

fn build_train_module_code(
    project: &Project,
    cfg: &TrainingConfig,
    has_custom_weight_file: bool,
) -> Result<String, String> {
    match (&project.module, &project.task) {
        (ProjectModule::Image, ProjectTask::Detection)
        | (ProjectModule::Image, ProjectTask::Segmentation) => {
            Ok(build_yolo_cell(project, cfg, has_custom_weight_file))
        }
        (ProjectModule::Image, ProjectTask::Classification) => {
            Ok(build_image_cls_cell(project, cfg, has_custom_weight_file))
        }
        (ProjectModule::Audio, ProjectTask::Classification) => Ok(build_yamnet_cell(project, cfg)),
        (ProjectModule::Audio, ProjectTask::Transcription) => Ok(build_whisper_cell(project, cfg)),
        (ProjectModule::Text, ProjectTask::Intent) => Ok(build_distilbert_cell(project, cfg)),
        (ProjectModule::Text, ProjectTask::Qa) | (ProjectModule::Text, ProjectTask::Completion) => {
            Ok(build_qwen_lora_cell(project, cfg))
        }
        _ => Err("Unsupported module/task combination for Colab training.".into()),
    }
}

pub fn build_headless_stages(
    run_id: &str,
    project: &Project,
    cfg: &TrainingConfig,
    has_custom_weight_file: bool,
    hf_token: &str,
) -> Result<HeadlessStagePlan, String> {
    let sync_ctx = NotebookDriveSyncContext::default();
    let setup_script =
        build_setup_cell(run_id, cfg, has_custom_weight_file, &sync_ctx, true, hf_token)?;
    let prepare_script = build_train_module_code(project, cfg, has_custom_weight_file)?;
    Ok(HeadlessStagePlan {
        setup_script,
        prepare_script,
        export_script: "damnisium_export_outputs()".to_string(),
    })
}

pub fn build_colab_notebook(
    run_id: &str,
    project: &Project,
    cfg: &TrainingConfig,
    has_custom_weight_file: bool,
    sync_ctx: &NotebookDriveSyncContext,
) -> Result<Vec<u8>, String> {
    let setup_cell = build_setup_cell(run_id, cfg, has_custom_weight_file, sync_ctx, false, "")?;
    let module_code = build_train_module_code(project, cfg, has_custom_weight_file)?;
    let train_cell = format!(
        "{module_code}\ndamnisium_train_chunk(0, int(TRAIN_CFG[\"epochs\"]), 0)\ndamnisium_export_outputs()\n"
    );

    let notebook = json!({
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {
                "provenance": [],
                "gpuType": "T4"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            },
            "accelerator": "GPU"
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    format!("## Damnisium {APP_VERSION} Remote Training Runner\n"),
                    format!("Project: `{}` | Run: `{run_id}` | Model: `{}`\n\n", project.name, cfg.model_id),
                    "Click **Runtime -> Run all** (`Ctrl+F9`). Works across Chrome profiles and Drive accounts. Live logs, charts, ONNX weights, and metrics sync automatically back to Damnisium."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": Value::Null,
                "metadata": {},
                "outputs": [],
                "source": split_source_lines(&setup_cell)
            },
            {
                "cell_type": "code",
                "execution_count": Value::Null,
                "metadata": {},
                "outputs": [],
                "source": split_source_lines(&train_cell)
            }
        ]
    });

    serde_json::to_vec_pretty(&notebook).map_err(|e| e.to_string())
}

fn split_source_lines(code: &str) -> Vec<String> {
    let lines: Vec<&str> = code.lines().collect();
    let total = lines.len();
    lines
        .into_iter()
        .enumerate()
        .map(|(i, line)| {
            if i + 1 < total {
                format!("{line}\n")
            } else {
                line.to_string()
            }
        })
        .collect()
}

fn build_setup_cell(
    run_id: &str,
    cfg: &TrainingConfig,
    has_custom_weight_file: bool,
    sync_ctx: &NotebookDriveSyncContext,
    headless: bool,
    hf_token: &str,
) -> Result<String, String> {
    let cfg_json = serde_json::to_string(cfg).map_err(|e| e.to_string())?;
    let cfg_b64 = BASE64_STD.encode(cfg_json.as_bytes());
    let hf_tok_b64 = BASE64_STD.encode(hf_token.trim().as_bytes());
    let drive_json = serde_json::to_string(&json!({
        "runFolderId": sync_ctx.run_folder_id,
        "outputFolderId": sync_ctx.output_folder_id,
        "datasetId": sync_ctx.dataset_file_id,
        "customWeightId": sync_ctx.custom_weight_file_id,
        "statusId": sync_ctx.status_file_id,
        "logId": sync_ctx.log_file_id,
        "outputBundleId": sync_ctx.output_bundle_id,
    }))
    .map_err(|e| e.to_string())?;
    let drive_b64 = BASE64_STD.encode(drive_json.as_bytes());
    let headless_py = if headless { "True" } else { "False" };

    let custom_block = if has_custom_weight_file {
        r#"custom_weight_src = os.path.join(RUN_DIR, "custom_weight.bin")
custom_weight_dst = os.path.join(WORK_DIR, "custom_weight.bin")
if os.path.exists(custom_weight_src):
    if os.path.abspath(custom_weight_src) != os.path.abspath(custom_weight_dst):
        shutil.copy2(custom_weight_src, custom_weight_dst)
elif not HEADLESS_MODE and DRIVE_FILES.get("customWeightId") and not os.path.exists(custom_weight_dst):
    download_drive_input(DRIVE_FILES["customWeightId"], custom_weight_dst, require_zip=False)"#
    } else {
        ""
    };

    Ok(format!(
        r#"import base64
import io
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import warnings
import zipfile

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

HEADLESS_MODE = {headless_py}
RUN_ID = "{run_id}"
TRAIN_CFG = json.loads(base64.b64decode("{cfg_b64}").decode("utf-8"))
DRIVE_FILES = json.loads(base64.b64decode("{drive_b64}").decode("utf-8"))
_HF_TOK = base64.b64decode("{hf_tok_b64}").decode("utf-8").strip()
if _HF_TOK:
    os.environ["HF_TOKEN"] = _HF_TOK
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _HF_TOK
if HEADLESS_MODE:
    try:
        import google.colab.userdata as _colab_ud
        _colab_ud.get = lambda k: os.environ.get(k)
    except Exception:
        pass

def _live_stdout():
    if HEADLESS_MODE:
        return sys.stdout
    return getattr(sys.stdout, "orig", sys.stdout)

def _live_stderr():
    if HEADLESS_MODE:
        return sys.stderr
    return getattr(sys.stderr, "orig", sys.stderr)

LOCAL_RUN_DIR = f"/content/drive/MyDrive/Damnisium/runs/{{RUN_ID}}"
has_fuse_run = False
if not HEADLESS_MODE:
    from google.colab import drive
    has_fuse_run = os.path.isdir("/content/drive/MyDrive") and os.path.exists(os.path.join(LOCAL_RUN_DIR, "dataset.zip"))
    if not has_fuse_run and not DRIVE_FILES.get("datasetId"):
        try:
            drive.mount("/content/drive", force_remount=False)
            has_fuse_run = os.path.exists(os.path.join(LOCAL_RUN_DIR, "dataset.zip"))
        except Exception as mount_err:
            _live_stdout().write(f"Drive mount notice: {{mount_err}}\n")
            _live_stdout().flush()

if HEADLESS_MODE:
    _candidates = [
        f"/content/damnisium_work/{{RUN_ID}}",
        os.path.abspath(os.path.join("damnisium_work", RUN_ID)),
        f"/damnisium_work/{{RUN_ID}}",
        "/content/damnisium_work",
        os.path.abspath("damnisium_work"),
        "/damnisium_work",
    ]
    WORK_DIR = next((d for d in _candidates if os.path.exists(os.path.join(d, "dataset.zip"))), _candidates[0])
else:
    WORK_DIR = "/content/damnisium_work"
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)

DATASET_DIR = os.path.join(WORK_DIR, "dataset")
if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)
os.makedirs(DATASET_DIR, exist_ok=True)

if HEADLESS_MODE:
    RUN_DIR = WORK_DIR
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    OUTPUT_DIR = os.path.join(WORK_DIR, "output")
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
elif has_fuse_run:
    RUN_DIR = LOCAL_RUN_DIR
    CHECKPOINT_DIR = os.path.join(RUN_DIR, "checkpoints")
    OUTPUT_DIR = os.path.join(RUN_DIR, "output")
else:
    RUN_DIR = os.path.join(WORK_DIR, "run_sync")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    OUTPUT_DIR = os.path.join(WORK_DIR, "output")

os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
STATUS_PATH = os.path.join(RUN_DIR, "status.json")
LOG_PATH = os.path.join(RUN_DIR, "train.log")

cfg_disk_path = os.path.join(RUN_DIR, "config.json")
if os.path.exists(cfg_disk_path):
    with open(cfg_disk_path, "r", encoding="utf-8") as f:
        TRAIN_CFG = json.load(f)

RESUME_FROM_RUN_ID = (TRAIN_CFG.get("resumeFromRunId") or "").strip()
RESUME_CKPT_DIR = os.path.join("/content/drive/MyDrive/Damnisium/runs", RESUME_FROM_RUN_ID, "checkpoints") if (has_fuse_run and RESUME_FROM_RUN_ID) else ""

DRIVE_SVC = None
_LOG_LOCK = threading.Lock()
_SYNC_LOCK = threading.Lock()
LOG_LINES = []
_LOG_DIRTY = False
_LAST_LOG_SYNC = 0.0

def ensure_drive_service():
    global DRIVE_SVC
    if DRIVE_SVC is not None:
        return DRIVE_SVC
    try:
        from google.colab import auth
        from googleapiclient.discovery import build
        auth.authenticate_user()
        DRIVE_SVC = build("drive", "v3", cache_discovery=False)
    except Exception as auth_err:
        _live_stdout().write(f"Drive API sync unavailable: {{auth_err}}\n")
        _live_stdout().flush()
    return DRIVE_SVC

def sync_drive_file(file_key, local_path, mime_type):
    fid = DRIVE_FILES.get(file_key) or ""
    if not fid or not os.path.exists(local_path):
        return
    svc = ensure_drive_service()
    if svc is None:
        return
    try:
        from googleapiclient.http import MediaFileUpload
        resumable = os.path.getsize(local_path) > (2 * 1024 * 1024)
        with _SYNC_LOCK:
            media = MediaFileUpload(local_path, mimetype=mime_type, resumable=resumable)
            svc.files().update(fileId=fid, media_body=media, supportsAllDrives=True).execute()
    except Exception as sync_err:
        _live_stderr().write(f"Drive sync warning ({{file_key}}): {{sync_err}}\n")
        _live_stderr().flush()

def download_drive_input(file_id, dest_path, require_zip=False):
    if not file_id:
        raise FileNotFoundError(f"Missing Drive file ID for {{dest_path}}")
    svc = ensure_drive_service()
    if svc is not None:
        try:
            from googleapiclient.http import MediaIoBaseDownload
            with _SYNC_LOCK:
                req = svc.files().get_media(fileId=file_id, supportsAllDrives=True)
                with open(dest_path, "wb") as fh:
                    downloader = MediaIoBaseDownload(fh, req)
                    done = False
                    while not done:
                        _, done = downloader.next_chunk()
            if os.path.getsize(dest_path) > 0 and (not require_zip or zipfile.is_zipfile(dest_path)):
                return dest_path
        except Exception as dl_err:
            _live_stdout().write(f"Drive API download fallback for {{file_id}}: {{dl_err}}\n")
            _live_stdout().flush()
    url = f"https://drive.usercontent.google.com/download?id={{file_id}}&export=download&confirm=t"
    urllib.request.urlretrieve(url, dest_path)
    if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
        raise RuntimeError(f"Downloaded empty file from Drive ID {{file_id}}")
    if require_zip and not zipfile.is_zipfile(dest_path):
        raise RuntimeError(f"Drive ID {{file_id}} did not download a valid zip archive. Sign into Colab or link matching account.")
    return dest_path

def _write_log_disk_locked():
    global _LOG_DIRTY, _LAST_LOG_SYNC
    with open(LOG_PATH, "w", encoding="utf-8") as lf:
        lf.write("\n".join(LOG_LINES) + ("\n" if LOG_LINES else ""))
    _LOG_DIRTY = False
    _LAST_LOG_SYNC = time.time()

def flush_logs(force=False):
    global _LOG_DIRTY
    did_write = False
    with _LOG_LOCK:
        if not LOG_LINES:
            return
        if force or (_LOG_DIRTY and (time.time() - _LAST_LOG_SYNC) >= 2.5):
            _write_log_disk_locked()
            did_write = True
    if did_write and not has_fuse_run:
        sync_drive_file("logId", LOG_PATH, "text/plain")

def _record_log_line(raw_line):
    global _LOG_DIRTY
    clean = raw_line.rstrip("\r\n")
    if not clean.strip():
        return
    with _LOG_LOCK:
        if len(LOG_LINES) >= 1200:
            del LOG_LINES[:400]
        LOG_LINES.append(clean)
        _LOG_DIRTY = True
    if (time.time() - _LAST_LOG_SYNC) >= 2.5:
        flush_logs(force=False)

class TeeStream(io.TextIOBase):
    def __init__(self, orig):
        self.orig = orig
        self.buf = ""
    def write(self, s):
        if not s:
            return 0
        self.orig.write(s)
        text = s.replace("\r\n", "\n")
        if "\r" in text and "\n" not in text:
            return len(s)
        text = text.replace("\r", "\n")
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            _record_log_line(line)
        return len(s)
    @property
    def encoding(self):
        return getattr(self.orig, "encoding", "utf-8")
    @property
    def errors(self):
        return getattr(self.orig, "errors", "replace")
    def flush(self):
        self.orig.flush()
        if self.buf.strip():
            _record_log_line(self.buf)
            self.buf = ""
    def isatty(self):
        return False

if not HEADLESS_MODE:
    if "_DAMNISIUM_ORIG_STDOUT" not in globals():
        _DAMNISIUM_ORIG_STDOUT = sys.stdout
        _DAMNISIUM_ORIG_STDERR = sys.stderr
    sys.stdout = TeeStream(_DAMNISIUM_ORIG_STDOUT)
    sys.stderr = TeeStream(_DAMNISIUM_ORIG_STDERR)

def append_log(line):
    stamp = time.strftime("%H:%M:%S")
    print(f"[{{stamp}}] {{line}}", flush=True)

def format_clean_exc(exc):
    msg = str(exc).strip()
    lines = [ln.strip() for ln in msg.splitlines() if ln.strip()]
    head = lines[0] if lines else type(exc).__name__
    if head.startswith(type(exc).__name__):
        return head
    return f"{{type(exc).__name__}}: {{head}}"

_NOISY_CMD_TOKENS = (
    "pip's dependency resolver",
    "which is incompatible",
    "requires huggingface-hub",
    "WARNING: Running pip",
    "0%|",
    "100%|",
    "Requesting secret HF_TOKEN",
)

def run_logged_cmd(cmd):
    append_log("Running: " + " ".join(str(c) for c in cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )
    tail_lines = []
    if proc.stdout is not None:
        for raw_line in proc.stdout:
            txt = raw_line.rstrip("\r\n")
            stripped = txt.strip()
            if not stripped:
                continue
            if any(tok in stripped for tok in _NOISY_CMD_TOKENS):
                continue
            tail_lines.append(stripped)
            if len(tail_lines) > 25:
                tail_lines.pop(0)
            print(txt, flush=True)
    ret = proc.wait()
    flush_logs(force=True)
    if ret != 0:
        err_tail = tail_lines[-1] if tail_lines else f"exit code {{ret}}"
        raise subprocess.CalledProcessError(ret, cmd, output=err_tail)

def find_latest_ckpt_dir(parent_dir):
    if not parent_dir or not os.path.isdir(parent_dir):
        return ""
    candidates = []
    for entry in os.listdir(parent_dir):
        full = os.path.join(parent_dir, entry)
        if os.path.isdir(full) and entry.startswith("checkpoint-"):
            try:
                step_num = int(entry.split("-")[-1])
            except Exception:
                step_num = 0
            candidates.append((step_num, full))
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1] if candidates else ""

def resolve_hf_train_repo(repo_id):
    if not repo_id:
        return repo_id
    low = repo_id.lower()
    if "gguf" not in low:
        return repo_id
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id)
        card = info.card_data.to_dict() if info.card_data else {{}}
        base_m = card.get("base_model")
        if isinstance(base_m, list) and base_m:
            return str(base_m[0])
        if isinstance(base_m, str) and base_m.strip():
            return base_m.strip()
    except Exception:
        pass
    return repo_id

METRICS_HISTORY = []
TRAIN_START_TS = time.time()

def _safe_float(v):
    if v is None:
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None

def _sanitize_json_val(v):
    if isinstance(v, dict):
        return {{k: _sanitize_json_val(val) for k, val in v.items()}}
    if isinstance(v, (list, tuple)):
        return [_sanitize_json_val(x) for x in v]
    if isinstance(v, bool) or v is None or isinstance(v, (int, str)):
        return v
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return str(v)

def _torch_load_compat(path, map_location="cpu"):
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def detect_gpu_name():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "CPU"

def write_status(state, message, current_epoch=0.0, total_epochs=1, progress_pct=0.0, eta_seconds=None, error=None, artifacts=None):
    payload = _sanitize_json_val({{
        "state": state,
        "gpu": detect_gpu_name(),
        "currentEpoch": current_epoch,
        "totalEpochs": total_epochs,
        "progressPct": progress_pct,
        "etaSeconds": eta_seconds,
        "message": message,
        "error": error,
        "metrics": METRICS_HISTORY,
        "artifacts": artifacts or []
    }})
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, allow_nan=False)
    flush_logs(force=True)
    if HEADLESS_MODE:
        out_stream = _live_stdout()
        out_stream.write("[DAMNISIUM_STATUS] " + json.dumps(payload, allow_nan=False) + "\n")
        out_stream.flush()
    elif not has_fuse_run:
        sync_drive_file("statusId", STATUS_PATH, "application/json")

def emit_chunk_done(completed_epoch, total_epochs):
    payload = _sanitize_json_val({{
        "completedEpoch": int(completed_epoch),
        "done": int(completed_epoch) >= int(total_epochs)
    }})
    if HEADLESS_MODE:
        out_stream = _live_stdout()
        out_stream.write("[DAMNISIUM_CHUNK_DONE] " + json.dumps(payload, allow_nan=False) + "\n")
        out_stream.flush()

def emit_outputs_done(artifacts):
    payload = _sanitize_json_val({{
        "outputDir": OUTPUT_DIR,
        "artifacts": list(artifacts)
    }})
    if HEADLESS_MODE:
        out_stream = _live_stdout()
        out_stream.write("[DAMNISIUM_OUTPUTS] " + json.dumps(payload, allow_nan=False) + "\n")
        out_stream.flush()

def export_torch_onnx(model, dummy, out_path, input_names, output_names, dynamic_axes):
    try:
        torch.onnx.export(
            model,
            dummy,
            out_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            model,
            dummy,
            out_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
        )

def write_eval_bundle_meta(labels_meta, metrics_meta):
    with open(os.path.join(OUTPUT_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(_sanitize_json_val(labels_meta), f, indent=2, allow_nan=False)
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(_sanitize_json_val(metrics_meta), f, indent=2, allow_nan=False)

def finalize_completed_run(message, total_epochs, artifacts):
    if HEADLESS_MODE:
        append_log(message)
        emit_outputs_done(artifacts)
        write_status("training", message, current_epoch=float(total_epochs), total_epochs=total_epochs, progress_pct=95.0, eta_seconds=0, artifacts=artifacts)
        return
    bundle_path = os.path.join(OUTPUT_DIR, "output_bundle.zip")
    with open(bundle_path, "wb") as bf:
        with zipfile.ZipFile(bf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root_d, _, fnames in os.walk(OUTPUT_DIR):
                for fn in fnames:
                    if fn == "output_bundle.zip":
                        continue
                    abs_p = os.path.join(root_d, fn)
                    rel_p = os.path.relpath(abs_p, OUTPUT_DIR).replace("\\", "/")
                    zf.write(abs_p, rel_p)
    if not os.path.exists(bundle_path) or os.path.getsize(bundle_path) <= 30:
        raise RuntimeError("Failed packaging non-empty output_bundle.zip")
    if not has_fuse_run:
        sync_drive_file("outputBundleId", bundle_path, "application/zip")
        if os.path.isdir("/content/drive/MyDrive"):
            try:
                backup_dir = f"/content/drive/MyDrive/Damnisium_Outputs/{{RUN_ID}}"
                os.makedirs(backup_dir, exist_ok=True)
                shutil.copy2(bundle_path, os.path.join(backup_dir, "output_bundle.zip"))
            except Exception:
                pass
    else:
        time.sleep(2.5)
    append_log(message)
    write_status("completed", message, current_epoch=float(total_epochs), total_epochs=total_epochs, progress_pct=100.0, eta_seconds=0, artifacts=artifacts)

write_status("installing", "Preparing dataset and Colab workspace", total_epochs=int(TRAIN_CFG["epochs"]), progress_pct=21.0)
dataset_zip_path = os.path.join(RUN_DIR, "dataset.zip")
if not os.path.exists(dataset_zip_path) and not HEADLESS_MODE:
    append_log("Downloading dataset.zip from shared Google Drive run")
    download_drive_input(DRIVE_FILES.get("datasetId", ""), dataset_zip_path, require_zip=True)
if not os.path.exists(dataset_zip_path):
    raise FileNotFoundError(f"dataset.zip not found at {{dataset_zip_path}}")
append_log("Extracting dataset.zip")
with zipfile.ZipFile(dataset_zip_path, "r") as zf:
    zf.extractall(DATASET_DIR)
{custom_block}
append_log("Dataset extracted cleanly")
write_status("installing", "Dataset extracted cleanly", total_epochs=int(TRAIN_CFG["epochs"]), progress_pct=23.0)
"#
    ))
}

fn build_yolo_cell(project: &Project, cfg: &TrainingConfig, has_custom_weight_file: bool) -> String {
    let is_seg = project.task == ProjectTask::Segmentation;
    let runtime_kind = if is_seg { "ort_yolo_seg" } else { "ort_yolo_det" };
    let base_weight = if has_custom_weight_file {
        "os.path.join(WORK_DIR, \"custom_weight.bin\")".to_string()
    } else if is_seg {
        format!("\"{}-seg.pt\"", cfg.model_id)
    } else {
        format!("\"{}.pt\"", cfg.model_id)
    };
    let aug = &project.image_aug;
    let fliplr = if aug.flip_horizontal { 0.5 } else { 0.0 };
    let flipud = if aug.flip_vertical { 0.5 } else { 0.0 };
    let degrees = aug.rotate_deg;
    let hsv_v = (aug.brightness_pct / 100.0).clamp(0.0, 0.9);
    let mosaic = if aug.mosaic { 1.0 } else { 0.0 };
    let mixup = if aug.mixup { 0.15 } else { 0.0 };
    let metric_key = if is_seg { "metrics/mAP50(M)" } else { "metrics/mAP50(B)" };
    let map95_key = if is_seg { "metrics/mAP50-95(M)" } else { "metrics/mAP50-95(B)" };
    let prec_key = if is_seg { "metrics/precision(M)" } else { "metrics/precision(B)" };
    let rec_key = if is_seg { "metrics/recall(M)" } else { "metrics/recall(B)" };

    format!(
        r#"import subprocess
import sys

write_status("installing", "Installing Ultralytics YOLO and ONNX export tools", total_epochs=int(TRAIN_CFG["epochs"]), progress_pct=24.0)
append_log("Installing ultralytics and onnx")
run_logged_cmd([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "onnx", "onnxslim", "onnxscript", "onnxruntime"])

from ultralytics import YOLO

try:
    total_epochs = int(TRAIN_CFG["epochs"])
    batch_size = int(TRAIN_CFG["batchSize"])
    img_size = int(TRAIN_CFG["imgSize"])
    lr0 = float(TRAIN_CFG["learningRate"])
    model_source = {base_weight}
    resume_ckpt = os.path.join(RESUME_CKPT_DIR, "last.pt") if RESUME_CKPT_DIR else ""
    if resume_ckpt and os.path.exists(resume_ckpt):
        append_log(f"Resuming YOLO checkpoint from {{resume_ckpt}}")
        model_source = resume_ckpt

    with open(os.path.join(DATASET_DIR, "classes.json"), "r", encoding="utf-8") as f:
        class_list = json.load(f)

    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    with open(yaml_path, "r", encoding="utf-8") as yf:
        yaml_lines = yf.read().splitlines()
    val_img_dir = os.path.join(DATASET_DIR, "images", "val")
    train_img_dir = os.path.join(DATASET_DIR, "images", "train")
    has_val_imgs = os.path.isdir(val_img_dir) and len(os.listdir(val_img_dir)) > 0
    rewritten_yaml = []
    for yl in yaml_lines:
        if yl.startswith("path:"):
            rewritten_yaml.append(f"path: {{DATASET_DIR}}")
        elif yl.startswith("val:") and not has_val_imgs and os.path.isdir(train_img_dir):
            rewritten_yaml.append("val: images/train")
        else:
            rewritten_yaml.append(yl)
    if not any(yl.startswith("path:") for yl in rewritten_yaml):
        rewritten_yaml.insert(0, f"path: {{DATASET_DIR}}")
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write("\n".join(rewritten_yaml) + "\n")

    last_eval = {{"map50": None, "map50_95": None, "precision": None, "recall": None}}
    _yolo_state = {{"lr": lr0, "best_score": None, "conf": None, "save_dir": os.path.join(WORK_DIR, "runs", "damnisium_yolo")}}

    def damnisium_train_chunk(start_ep, end_ep, max_secs=450):
        try:
            start_ep = int(start_ep)
            target_end = min(total_epochs, int(end_ep))
            if start_ep >= total_epochs:
                emit_chunk_done(total_epochs, total_epochs)
                return total_epochs
            chunk_epochs = max(1, target_end - start_ep)
            chunk_t0 = time.time()
            start_frac = max(0.01, 1.0 - 0.99 * (start_ep / max(1, total_epochs)))
            end_frac = max(0.01, 1.0 - 0.99 * (target_end / max(1, total_epochs)))
            chunk_lr0 = lr0 * start_frac
            chunk_lrf = max(0.01, min(1.0, end_frac / start_frac))
            chunk_warmup = min(3.0, max(0.0, 3.0 - float(start_ep)))
            close_mos = min(chunk_epochs, max(0, target_end - max(0, total_epochs - 10))) if {mosaic} > 0 else 0
            last_pt_ckpt = os.path.join(CHECKPOINT_DIR, "last.pt")
            src_weights = last_pt_ckpt if (start_ep > 0 and os.path.exists(last_pt_ckpt)) else model_source
            append_log(f"Loading YOLO weights {{src_weights}} for epochs {{start_ep + 1}}..{{target_end}}")
            chunk_model = YOLO(src_weights)
            state_box = {{"last_ep": start_ep, "seen_local": 0}}

            def _extract_loss(loss_items):
                if loss_items is None:
                    return None
                if hasattr(loss_items, "detach") and hasattr(loss_items, "mean"):
                    return _safe_float(loss_items.detach().float().mean().item())
                if hasattr(loss_items, "mean"):
                    return _safe_float(loss_items.mean().item())
                if isinstance(loss_items, (list, tuple)):
                    vals = [_safe_float(x) for x in loss_items]
                    vals = [x for x in vals if x is not None]
                    return (sum(vals) / len(vals)) if vals else None
                return _safe_float(loss_items)

            def _extract_trainer_train_loss(trainer, metrics):
                for attr in ("tloss", "loss_items", "loss"):
                    val = _extract_loss(getattr(trainer, attr, None))
                    if val is not None:
                        return val
                try:
                    tloss_raw = getattr(trainer, "tloss", None)
                    if tloss_raw is not None and hasattr(trainer, "label_loss_items"):
                        labeled = trainer.label_loss_items(tloss_raw, prefix="train")
                        if isinstance(labeled, dict):
                            vals = [_safe_float(v) for v in labeled.values()]
                            vals = [v for v in vals if v is not None]
                            if vals:
                                return sum(vals) / len(vals)
                except Exception:
                    pass
                metric_losses = [
                    _safe_float(metrics.get(k))
                    for k in ("train/box_loss", "train/cls_loss", "train/dfl_loss", "train/seg_loss")
                ]
                metric_losses = [v for v in metric_losses if v is not None]
                return (sum(metric_losses) / len(metric_losses)) if metric_losses else None

            def on_epoch_end(trainer):
                local_ep = int(trainer.epoch + 1)
                if local_ep > chunk_epochs or local_ep <= state_box["seen_local"]:
                    return
                state_box["seen_local"] = local_ep
                epoch_num = float(start_ep + local_ep)
                state_box["last_ep"] = int(epoch_num)
                elapsed = max(1e-3, time.time() - TRAIN_START_TS)
                eta = int((elapsed / max(1.0, epoch_num)) * max(0.0, total_epochs - epoch_num))
                metrics = trainer.metrics or {{}}
                train_loss = _extract_trainer_train_loss(trainer, metrics)
                val_loss = _safe_float(metrics.get("val/box_loss") or metrics.get("val/seg_loss") or metrics.get("val/cls_loss"))
                score = _safe_float(metrics.get("{metric_key}"))
                if score is not None:
                    last_eval["map50"] = score
                v95 = _safe_float(metrics.get("{map95_key}"))
                if v95 is not None:
                    last_eval["map50_95"] = v95
                vp = _safe_float(metrics.get("{prec_key}"))
                if vp is not None:
                    last_eval["precision"] = vp
                vr = _safe_float(metrics.get("{rec_key}"))
                if vr is not None:
                    last_eval["recall"] = vr
                lr_val = None
                try:
                    lr_val = _safe_float(trainer.optimizer.param_groups[0]["lr"])
                    if lr_val is not None:
                        _yolo_state["lr"] = lr_val
                except Exception:
                    pass
                METRICS_HISTORY.append({{
                    "step": int(epoch_num),
                    "epoch": epoch_num,
                    "trainLoss": train_loss,
                    "valLoss": val_loss,
                    "score": score,
                    "scoreName": "mAP50",
                    "lr": lr_val
                }})
                sdir = str(trainer.save_dir)
                _yolo_state["save_dir"] = sdir
                try:
                    cm = getattr(getattr(trainer, "validator", None), "confusion_matrix", None)
                    if cm is not None and hasattr(cm, "matrix"):
                        _yolo_state["conf"] = {{"labels": class_list + ["background"], "matrix": cm.matrix.astype(int).tolist()}}
                except Exception:
                    pass
                pct = 25.0 + (epoch_num / max(1, total_epochs)) * 67.0
                loss_s = f"{{train_loss:.4f}}" if train_loss is not None else "n/a"
                score_s = f"{{score:.4f}}" if score is not None else "n/a"
                append_log(f"Epoch {{int(epoch_num)}}/{{total_epochs}} loss={{loss_s}} mAP50={{score_s}}")
                write_status("training", f"Epoch {{int(epoch_num)}} / {{total_epochs}}", current_epoch=epoch_num, total_epochs=total_epochs, progress_pct=pct, eta_seconds=eta)
                if max_secs > 0 and (time.time() - chunk_t0) >= max_secs and int(epoch_num) < total_epochs:
                    trainer.stop = True

            chunk_model.add_callback("on_fit_epoch_end", on_epoch_end)
            if start_ep == 0:
                write_status("training", "Starting YOLO training", current_epoch=0.0, total_epochs=total_epochs, progress_pct=25.0)
            chunk_model.train(
                data=yaml_path,
                epochs=chunk_epochs,
                imgsz=img_size,
                batch=batch_size,
                lr0=chunk_lr0,
                lrf=chunk_lrf,
                warmup_epochs=chunk_warmup,
                close_mosaic=close_mos,
                fliplr={fliplr},
                flipud={flipud},
                degrees={degrees},
                hsv_v={hsv_v},
                mosaic={mosaic},
                mixup={mixup},
                project=os.path.join(WORK_DIR, "runs"),
                name="damnisium_yolo",
                exist_ok=True
            )
            sdir = _yolo_state["save_dir"]
            src_last = os.path.join(sdir, "weights", "last.pt")
            if os.path.exists(src_last):
                shutil.copy2(src_last, os.path.join(CHECKPOINT_DIR, "last.pt"))
            src_best = os.path.join(sdir, "weights", "best.pt")
            best_overall = os.path.join(CHECKPOINT_DIR, "best_overall.pt")
            if os.path.exists(src_best):
                prev_b = _yolo_state["best_score"]
                cur_b = last_eval["map50"]
                if not os.path.exists(best_overall) or (cur_b is not None and (prev_b is None or cur_b >= prev_b)):
                    _yolo_state["best_score"] = cur_b
                    shutil.copy2(src_best, best_overall)
            completed_ep = min(total_epochs, max(start_ep + 1, state_box["last_ep"]))
            emit_chunk_done(completed_ep, total_epochs)
            return completed_ep
        except Exception as exc:
            clean_err = format_clean_exc(exc)
            append_log(f"Error: {{clean_err}}")
            write_status("failed", "Training failed", total_epochs=total_epochs, error=clean_err)
            raise RuntimeError(clean_err) from None

    def damnisium_export_outputs():
        try:
            save_dir = _yolo_state["save_dir"]
            artifacts = []
            best_overall = os.path.join(CHECKPOINT_DIR, "best_overall.pt")
            if os.path.exists(best_overall):
                shutil.copy2(best_overall, os.path.join(OUTPUT_DIR, "best.pt"))
                artifacts.append("best.pt")
            else:
                src_best = os.path.join(save_dir, "weights", "best.pt")
                if os.path.exists(src_best):
                    shutil.copy2(src_best, os.path.join(OUTPUT_DIR, "best.pt"))
                    artifacts.append("best.pt")
            for last_cand in (os.path.join(CHECKPOINT_DIR, "last.pt"), os.path.join(save_dir, "weights", "last.pt")):
                if os.path.exists(last_cand):
                    shutil.copy2(last_cand, os.path.join(OUTPUT_DIR, "last.pt"))
                    artifacts.append("last.pt")
                    break

            best_pt = os.path.join(OUTPUT_DIR, "best.pt")
            if not os.path.exists(best_pt):
                best_pt = os.path.join(OUTPUT_DIR, "last.pt")
            if TRAIN_CFG.get("exportOnnx", True) and os.path.exists(best_pt):
                append_log("Exporting best.onnx")
                write_status("training", "Exporting YOLO ONNX model", current_epoch=float(total_epochs), total_epochs=total_epochs, progress_pct=94.0)
                best_model = YOLO(best_pt)
                onnx_path = None
                for exp_kw in (
                    {{"format": "onnx", "imgsz": img_size, "opset": 17, "simplify": True}},
                    {{"format": "onnx", "imgsz": img_size, "opset": 17, "simplify": False}},
                    {{"format": "onnx", "imgsz": img_size, "simplify": False}},
                ):
                    try:
                        onnx_path = best_model.export(**exp_kw)
                        if onnx_path and os.path.exists(onnx_path) and os.path.getsize(onnx_path) > 0:
                            break
                    except Exception as exp_err:
                        append_log(f"ONNX export attempt ({{exp_kw}}) notice: {{exp_err}}")
                        onnx_path = None
                dst_onnx = os.path.join(OUTPUT_DIR, "best.onnx")
                if not onnx_path or not os.path.exists(onnx_path) or os.path.getsize(onnx_path) == 0:
                    raise RuntimeError("YOLO ONNX export did not produce a valid model file.")
                if os.path.abspath(str(onnx_path)) != os.path.abspath(dst_onnx):
                    shutil.copy2(onnx_path, dst_onnx)
                    if os.path.dirname(os.path.abspath(str(onnx_path))) == os.path.abspath(OUTPUT_DIR):
                        try:
                            os.remove(str(onnx_path))
                        except Exception:
                            pass
                artifacts.append("best.onnx")

            last_val_loss = next((m["valLoss"] for m in reversed(METRICS_HISTORY) if m.get("valLoss") is not None), None)
            write_eval_bundle_meta(
                {{
                    "runtimeKind": "{runtime_kind}",
                    "classes": class_list,
                    "imgSize": img_size
                }},
                {{
                    "primaryName": "mAP50",
                    "primaryValue": last_eval["map50"],
                    "valAccuracy": None,
                    "valLoss": last_val_loss,
                    "map50": last_eval["map50"],
                    "map50_95": last_eval["map50_95"],
                    "precision": last_eval["precision"],
                    "recall": last_eval["recall"],
                    "lossCurve": METRICS_HISTORY,
                    "confusion": _yolo_state["conf"]
                }}
            )
            artifacts.extend(["labels.json", "metrics.json"])
            finalize_completed_run("Training completed and ONNX evaluation bundle saved", total_epochs, artifacts)
        except Exception as exc:
            clean_err = format_clean_exc(exc)
            append_log(f"Error: {{clean_err}}")
            write_status("failed", "YOLO export failed", total_epochs=total_epochs, error=clean_err)
            raise RuntimeError(clean_err) from None
except Exception as exc:
    clean_err = format_clean_exc(exc)
    append_log(f"Error: {{clean_err}}")
    write_status("failed", "YOLO setup failed", total_epochs=int(TRAIN_CFG["epochs"]), error=clean_err)
    raise RuntimeError(clean_err) from None
"#
    )
}

fn build_image_cls_cell(project: &Project, cfg: &TrainingConfig, has_custom_weight_file: bool) -> String {
    let model_arch = match cfg.model_id.as_str() {
        "efficientnet_b0" => "efficientnet_b0",
        "resnet18" => "resnet18",
        _ => "mobilenet_v3_large",
    };
    let custom_repo = cfg
        .custom_hf_repo
        .as_deref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .unwrap_or("");
    let timm_pkg = if custom_repo.is_empty() { "" } else { "\"timm\", " };
    let aug = &project.image_aug;
    let load_custom = if has_custom_weight_file { "True" } else { "False" };

    format!(
        r#"import subprocess
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

write_status("installing", "Preparing PyTorch Vision classifier and ONNX exporter", total_epochs=int(TRAIN_CFG["epochs"]), progress_pct=24.0)
run_logged_cmd([sys.executable, "-m", "pip", "install", "-q", {timm_pkg}"onnx", "onnxscript"])

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_epochs = int(TRAIN_CFG["epochs"])
    batch_size = int(TRAIN_CFG["batchSize"])
    img_size = int(TRAIN_CFG["imgSize"])
    lr = float(TRAIN_CFG["learningRate"])

    train_tfms = [transforms.Resize((img_size, img_size))]
    if {flip_h}:
        train_tfms.append(transforms.RandomHorizontalFlip(0.5))
    if {flip_v}:
        train_tfms.append(transforms.RandomVerticalFlip(0.5))
    if {rot_deg} > 0:
        train_tfms.append(transforms.RandomRotation({rot_deg}))
    if {bright} > 0 or {contrast} > 0:
        train_tfms.append(transforms.ColorJitter(brightness={bright}, contrast={contrast}))
    train_tfms.extend([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir = os.path.join(DATASET_DIR, "val")
    if not os.path.exists(val_dir) or len(os.listdir(val_dir)) == 0:
        val_dir = train_dir

    train_ds = datasets.ImageFolder(train_dir, transform=transforms.Compose(train_tfms))
    if len(train_ds.samples) == 0:
        raise RuntimeError("No training images found in dataset.")
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)
    _val_samples = []
    for _vp, _ in val_ds.samples:
        _cls_name = os.path.basename(os.path.dirname(_vp))
        if _cls_name in train_ds.class_to_idx:
            _val_samples.append((_vp, train_ds.class_to_idx[_cls_name]))
    if _val_samples:
        val_ds.samples = _val_samples
        val_ds.targets = [s[1] for s in _val_samples]
    else:
        val_ds = datasets.ImageFolder(train_dir, transform=val_tfms)
    num_classes = len(train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=(len(train_ds) > batch_size and len(train_ds) % batch_size == 1))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    custom_repo = "{custom_repo}".strip()
    if custom_repo:
        import timm
        try:
            src_name = custom_repo if custom_repo.startswith("hf-hub:") else f"hf-hub:{{custom_repo}}"
            model = timm.create_model(src_name, pretrained=True, num_classes=num_classes)
        except Exception:
            model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif "{model_arch}" == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif "{model_arch}" == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    if {load_custom} and os.path.exists(os.path.join(WORK_DIR, "custom_weight.bin")):
        state = _torch_load_compat(os.path.join(WORK_DIR, "custom_weight.bin"), map_location="cpu")
        if isinstance(state, dict):
            model.load_state_dict(state, strict=False)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs))

    best_acc = -1.0
    best_vloss = 0.0
    best_matrix = [[0] * num_classes for _ in range(num_classes)]
    best_path = os.path.join(OUTPUT_DIR, "best_model.pt")
    start_epoch = 1
    resume_ckpt = os.path.join(RESUME_CKPT_DIR, "checkpoint_last.pt") if RESUME_CKPT_DIR else ""
    if resume_ckpt and os.path.exists(resume_ckpt):
        ckpt = _torch_load_compat(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            pass
        start_epoch = min(total_epochs, int(ckpt.get("epoch", 0)) + 1)
        best_acc = float(ckpt.get("best_acc", -1.0))
        append_log(f"Resumed image classifier checkpoint at epoch {{start_epoch}}")

    def damnisium_train_chunk(start_ep, end_ep, max_secs=450):
        global best_acc, best_vloss, best_matrix
        try:
            start_ep = int(start_ep)
            eff_start = max(start_ep + 1, start_epoch) if start_ep == 0 else (start_ep + 1)
            target_end = min(total_epochs, int(end_ep))
            if start_ep >= total_epochs or eff_start > total_epochs:
                emit_chunk_done(total_epochs, total_epochs)
                return total_epochs
            if eff_start > target_end:
                emit_chunk_done(eff_start - 1, total_epochs)
                return eff_start - 1
            if start_ep == 0:
                write_status("training", "Starting classifier training", total_epochs=total_epochs, progress_pct=25.0)
            chunk_t0 = time.time()
            completed_ep = start_ep
            for epoch in range(eff_start, target_end + 1):
                model.train()
                running_loss = 0.0
                steps = 0
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    out = model(x)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()
                    running_loss += float(loss.item())
                    steps += 1
                scheduler.step()
                train_loss = running_loss / max(1, steps)

                model.eval()
                val_loss = 0.0
                val_steps = 0
                correct = 0
                total = 0
                matrix = [[0] * num_classes for _ in range(num_classes)]
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        out = model(x)
                        val_loss += float(criterion(out, y).item())
                        val_steps += 1
                        preds = out.argmax(dim=1)
                        correct += int((preds == y).sum().item())
                        total += int(y.size(0))
                        for t_idx, p_idx in zip(y.cpu().tolist(), preds.cpu().tolist()):
                            matrix[t_idx][p_idx] += 1

                acc = correct / max(1, total)
                v_loss = val_loss / max(1, val_steps)
                if acc >= best_acc:
                    best_acc = acc
                    best_vloss = v_loss
                    best_matrix = matrix
                    torch.save(model.state_dict(), best_path)

                torch.save({{
                    "epoch": epoch,
                    "best_acc": best_acc,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                }}, os.path.join(CHECKPOINT_DIR, "checkpoint_last.pt"))

                elapsed = max(1e-3, time.time() - TRAIN_START_TS)
                eta = int((elapsed / epoch) * max(0, total_epochs - epoch))
                METRICS_HISTORY.append({{
                    "step": epoch,
                    "epoch": float(epoch),
                    "trainLoss": train_loss,
                    "valLoss": v_loss,
                    "score": acc,
                    "scoreName": "val_acc",
                    "lr": float(optimizer.param_groups[0]["lr"])
                }})
                append_log(f"Epoch {{epoch}}/{{total_epochs}} train_loss={{train_loss:.4f}} val_loss={{v_loss:.4f}} val_acc={{acc:.4f}}")
                write_status("training", f"Epoch {{epoch}} / {{total_epochs}}", current_epoch=float(epoch), total_epochs=total_epochs, progress_pct=25.0 + (epoch / total_epochs) * 67.0, eta_seconds=eta)
                completed_ep = epoch
                if max_secs > 0 and (time.time() - chunk_t0) >= max_secs and epoch < total_epochs:
                    break
            emit_chunk_done(completed_ep, total_epochs)
            return completed_ep
        except Exception as exc:
            clean_err = format_clean_exc(exc)
            append_log(f"Error: {{clean_err}}")
            write_status("failed", "Classification training failed", total_epochs=total_epochs, error=clean_err)
            raise RuntimeError(clean_err) from None

    def damnisium_export_outputs():
        global model
        try:
            if not os.path.exists(best_path):
                torch.save(model.state_dict(), best_path)
            model.load_state_dict(_torch_load_compat(best_path, map_location="cpu"))
            model = model.cpu().eval()
            artifacts = ["best_model.pt"]
            if TRAIN_CFG.get("exportOnnx", True):
                dummy = torch.randn(1, 3, img_size, img_size)
                onnx_out = os.path.join(OUTPUT_DIR, "model.onnx")
                export_torch_onnx(model, dummy, onnx_out, ["images"], ["logits"], {{"images": {{0: "batch"}}, "logits": {{0: "batch"}}}})
                artifacts.append("model.onnx")

            write_eval_bundle_meta(
                {{
                    "runtimeKind": "ort_img_cls",
                    "classes": train_ds.classes,
                    "imgSize": img_size
                }},
                {{
                    "primaryName": "val_acc",
                    "primaryValue": best_acc,
                    "valAccuracy": best_acc,
                    "valLoss": best_vloss,
                    "map50": None,
                    "map50_95": None,
                    "precision": None,
                    "recall": None,
                    "lossCurve": METRICS_HISTORY,
                    "confusion": {{"labels": train_ds.classes, "matrix": best_matrix}}
                }}
            )
            artifacts.extend(["labels.json", "metrics.json"])
            finalize_completed_run("Classification training completed", total_epochs, artifacts)
        except Exception as exc:
            clean_err = format_clean_exc(exc)
            append_log(f"Error: {{clean_err}}")
            write_status("failed", "Classification export failed", total_epochs=total_epochs, error=clean_err)
            raise RuntimeError(clean_err) from None
except Exception as exc:
    clean_err = format_clean_exc(exc)
    append_log(f"Error: {{clean_err}}")
    write_status("failed", "Classification setup failed", total_epochs=int(TRAIN_CFG["epochs"]), error=clean_err)
    raise RuntimeError(clean_err) from None
"#,
        flip_h = if aug.flip_horizontal { "True" } else { "False" },
        flip_v = if aug.flip_vertical { "True" } else { "False" },
        rot_deg = aug.rotate_deg,
        bright = (aug.brightness_pct / 100.0).clamp(0.0, 0.9),
        contrast = (aug.contrast_pct / 100.0).clamp(0.0, 0.9),
    )
}

fn build_yamnet_cell(project: &Project, _cfg: &TrainingConfig) -> String {
    let aug = &project.audio_aug;
    format!(
        r#"import subprocess
import sys

write_status("installing", "Installing TensorFlow Hub, Librosa, and ONNX tools", total_epochs=int(TRAIN_CFG["epochs"]), progress_pct=24.0)
run_logged_cmd([sys.executable, "-m", "pip", "install", "-q", "tensorflow-hub", "librosa", "soundfile", "numpy", "onnx", "onnxscript"])

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import torch
import torch.nn as nn

try:
    try:
        for _g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass
    total_epochs = int(TRAIN_CFG["epochs"])
    batch_size = int(TRAIN_CFG["batchSize"])
    lr = float(TRAIN_CFG["learningRate"])

    with open(os.path.join(DATASET_DIR, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(os.path.join(DATASET_DIR, "classes.json"), "r", encoding="utf-8") as f:
        class_list = json.load(f)

    label_to_idx = {{name: idx for idx, name in enumerate(class_list)}}
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    def augment_wave(y, sr):
        if {noise} > 0:
            y = y + np.random.normal(0.0, {noise}, size=y.shape).astype(np.float32)
        if abs({gain_db}) > 0:
            gain = 10.0 ** (np.random.uniform(-{gain_db}, {gain_db}) / 20.0)
            y = y * gain
        if abs({pitch}) > 0:
            steps = np.random.uniform(-{pitch}, {pitch})
            y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=steps)
        if abs({speed} - 1.0) > 1e-3:
            y = librosa.effects.time_stretch(y=y, rate=float({speed}))
        if {shift_pct} > 0 and len(y) > 0:
            max_shift = int(len(y) * ({shift_pct} / 100.0))
            if max_shift > 0:
                y = np.roll(y, np.random.randint(-max_shift, max_shift + 1))
        return np.clip(y, -1.0, 1.0).astype(np.float32)

    def to_log_mel_patch(y):
        mel = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=512, hop_length=160, n_mels=64, fmin=40, fmax=7500)
        log_mel = np.log(mel + 1e-6).T
        if log_mel.shape[0] < 96:
            pad_rows = 96 - log_mel.shape[0]
            log_mel = np.pad(log_mel, ((0, pad_rows), (0, 0)), mode="constant")
        else:
            log_mel = log_mel[:96, :]
        mean = np.mean(log_mel)
        std = np.std(log_mel) + 1e-5
        return ((log_mel - mean) / std).astype(np.float32)

    def extract_pair(audio_path, start_sec, end_sec, do_aug):
        dur = max(0.2, float(end_sec) - float(start_sec))
        y, _ = librosa.load(os.path.join(DATASET_DIR, audio_path), sr=16000, offset=float(start_sec), duration=dur, mono=True)
        if len(y) < 15600:
            y = np.pad(y, (0, 15600 - len(y)))
        if do_aug:
            y = augment_wave(y, 16000)
        _, embeddings, _ = yamnet(y)
        return np.mean(embeddings.numpy(), axis=0), to_log_mel_patch(y)

    X_train, M_train, y_train = [], [], []
    X_val, M_val, y_val = [], [], []
    for row in manifest:
        lbl = row.get("label", "")
        if lbl not in label_to_idx:
            continue
        idx = label_to_idx[lbl]
        is_val = row.get("split") == "val"
        emb, mel_patch = extract_pair(row["audio_path"], row["start_sec"], row["end_sec"], not is_val)
        if is_val:
            X_val.append(emb)
            M_val.append(mel_patch)
            y_val.append(idx)
        else:
            X_train.append(emb)
            M_train.append(mel_patch)
            y_train.append(idx)
            emb_aug, mel_aug = extract_pair(row["audio_path"], row["start_sec"], row["end_sec"], True)
            X_train.append(emb_aug)
            M_train.append(mel_aug)
            y_train.append(idx)

    if len(X_train) == 0:
        raise RuntimeError("No labeled audio slices found for training.")
    X_train = np.array(X_train, dtype=np.float32)
    M_train = np.array(M_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    if len(X_val) == 0:
        X_val, M_val, y_val = X_train, M_train, y_train
    else:
        X_val = np.array(X_val, dtype=np.float32)
        M_val = np.array(M_val, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.int32)

    clf = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(len(class_list), activation="softmax")
    ])
    initial_epoch = 0
    resume_keras = os.path.join(RESUME_CKPT_DIR, "yamnet_last.keras") if RESUME_CKPT_DIR else ""
    resume_meta = os.path.join(RESUME_CKPT_DIR, "ckpt_state.json") if RESUME_CKPT_DIR else ""
    if resume_keras and os.path.exists(resume_keras):
        clf = tf.keras.models.load_model(resume_keras)
        if os.path.exists(resume_meta):
            with open(resume_meta, "r", encoding="utf-8") as mf:
                initial_epoch = int(json.load(mf).get("epoch", 0))
        append_log(f"Resumed YAMNet checkpoint at epoch {{initial_epoch}}")

    clf.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    _ym_state = {{"last_ep": min(initial_epoch, max(0, total_epochs)), "t0": time.time(), "max_secs": 0}}

    class DriveProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {{}}
            ep = epoch + 1
            _ym_state["last_ep"] = ep
            elapsed = max(1e-3, time.time() - TRAIN_START_TS)
            eta = int((elapsed / ep) * max(0, total_epochs - ep))
            clf.save(os.path.join(CHECKPOINT_DIR, "yamnet_last.keras"))
            with open(os.path.join(CHECKPOINT_DIR, "ckpt_state.json"), "w", encoding="utf-8") as mf:
                json.dump({{"epoch": ep}}, mf)
            METRICS_HISTORY.append({{
                "step": ep,
                "epoch": float(ep),
                "trainLoss": _safe_float(logs.get("loss")),
                "valLoss": _safe_float(logs.get("val_loss")),
                "score": _safe_float(logs.get("val_accuracy", logs.get("accuracy"))),
                "scoreName": "val_acc",
                "lr": lr
            }})
            loss_v = _safe_float(logs.get("loss"))
            vacc_v = _safe_float(logs.get("val_accuracy", logs.get("accuracy")))
            loss_s = f"{{loss_v:.4f}}" if loss_v is not None else "n/a"
            vacc_s = f"{{vacc_v:.4f}}" if vacc_v is not None else "n/a"
            append_log(f"Epoch {{ep}}/{{total_epochs}} loss={{loss_s}} val_acc={{vacc_s}}")
            write_status("training", f"Epoch {{ep}} / {{total_epochs}}", current_epoch=float(ep), total_epochs=total_epochs, progress_pct=25.0 + (ep / total_epochs) * 63.0, eta_seconds=eta)
            if _ym_state["max_secs"] > 0 and (time.time() - _ym_state["t0"]) >= _ym_state["max_secs"] and ep < total_epochs:
                self.model.stop_training = True

    def damnisium_train_chunk(start_ep, end_ep, max_secs=450):
        try:
            start_ep = int(start_ep)
            target_ep = min(total_epochs, int(end_ep))
            eff_init = _ym_state["last_ep"] if start_ep == 0 else start_ep
            if eff_init >= total_epochs:
                emit_chunk_done(total_epochs, total_epochs)
                return total_epochs
            if eff_init >= target_ep:
                emit_chunk_done(eff_init, total_epochs)
                return eff_init
            _ym_state["t0"] = time.time()
            _ym_state["max_secs"] = float(max_secs)
            if start_ep == 0:
                write_status("training", "Training YAMNet classifier head", total_epochs=total_epochs, progress_pct=25.0)
            clf.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=target_ep, initial_epoch=eff_init, batch_size=batch_size, callbacks=[DriveProgressCallback()], verbose=0)
            completed_ep = min(total_epochs, max(eff_init + 1, _ym_state["last_ep"]))
            emit_chunk_done(completed_ep, total_epochs)
            return completed_ep
        except Exception as exc:
            clean_err = format_clean_exc(exc)
            append_log(f"Error: {{clean_err}}")
            write_status("failed", "YAMNet training failed", total_epochs=total_epochs, error=clean_err)
            raise RuntimeError(clean_err) from None

    def damnisium_export_outputs():
        try:
            keras_path = os.path.join(OUTPUT_DIR, "yamnet_classifier.keras")
            clf.save(keras_path)

            append_log("Distilling local WASM-compatible ONNX audio classifier")
            write_status("training", "Exporting WASM-compatible audio ONNX", current_epoch=float(total_epochs), total_epochs=total_epochs, progress_pct=90.0)
            teacher_probs = clf.predict(X_train, verbose=0)

            class AudioMelNet(nn.Module):
                def __init__(self, num_cls):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Conv2d(1, 32, kernel_size=3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, kernel_size=3, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(128, num_cls)
                    )
                def forward(self, x):
                    return self.net(x)

            mel_net = AudioMelNet(len(class_list))
            opt = torch.optim.AdamW(mel_net.parameters(), lr=1e-3)
            ce = nn.CrossEntropyLoss()
            kl = nn.KLDivLoss(reduction="batchmean")
            M_t = torch.tensor(M_train[:, None, :, :], dtype=torch.float32)
            Y_t = torch.tensor(y_train, dtype=torch.long)
            P_t = torch.tensor(teacher_probs, dtype=torch.float32)
            if M_t.shape[0] == 1:
                M_t = M_t.repeat(2, 1, 1, 1)
                Y_t = Y_t.repeat(2)
                P_t = P_t.repeat(2, 1)

            mel_net.train()
            for _ in range(40):
                opt.zero_grad()
                logits = mel_net(M_t)
                loss = 0.7 * ce(logits, Y_t) + 0.3 * kl(torch.log_softmax(logits, dim=1), P_t)
                loss.backward()
                opt.step()

            mel_net.eval()
            yamnet_preds = np.argmax(clf.predict(X_val, verbose=0), axis=1)
            num_cls = len(class_list)
            conf_mat = [[0] * num_cls for _ in range(num_cls)]
            for t_i, p_i in zip(y_val.tolist(), yamnet_preds.tolist()):
                conf_mat[t_i][p_i] += 1
            final_acc = float(np.mean(yamnet_preds == y_val)) if len(y_val) > 0 else 0.0

            dummy_mel = torch.randn(1, 1, 96, 64)
            export_torch_onnx(mel_net, dummy_mel, os.path.join(OUTPUT_DIR, "model.onnx"), ["mel"], ["logits"], {{"mel": {{0: "batch"}}, "logits": {{0: "batch"}}}})

            write_eval_bundle_meta(
                {{
                    "runtimeKind": "ort_audio_mel",
                    "classes": class_list,
                    "sampleRate": 16000,
                    "nMels": 64,
                    "patchFrames": 96
                }},
                {{
                    "primaryName": "val_acc",
                    "primaryValue": final_acc,
                    "valAccuracy": final_acc,
                    "valLoss": METRICS_HISTORY[-1]["valLoss"] if METRICS_HISTORY else None,
                    "map50": None,
                    "map50_95": None,
                    "precision": None,
                    "recall": None,
                    "lossCurve": METRICS_HISTORY,
                    "confusion": {{"labels": class_list, "matrix": conf_mat}}
                }}
            )
            finalize_completed_run("YAMNet audio training and ONNX export completed", total_epochs, ["yamnet_classifier.keras", "model.onnx", "labels.json", "metrics.json"])
        except Exception as exc:
            clean_err = format_clean_exc(exc)
            append_log(f"Error: {{clean_err}}")
            write_status("failed", "YAMNet export failed", total_epochs=total_epochs, error=clean_err)
            raise RuntimeError(clean_err) from None
except Exception as exc:
    clean_err = format_clean_exc(exc)
    append_log(f"Error: {{clean_err}}")
    write_status("failed", "YAMNet setup failed", total_epochs=int(TRAIN_CFG["epochs"]), error=clean_err)
    raise RuntimeError(clean_err) from None
"#,
        noise = aug.noise_level,
        gain_db = aug.gain_db,
        pitch = aug.pitch_semitones,
        speed = aug.speed_factor,
        shift_pct = aug.time_shift_pct,
    )
}

fn build_whisper_cell(project: &Project, cfg: &TrainingConfig) -> String {
    let default_repo = match cfg.model_id.as_str() {
        "whisper-base" => "openai/whisper-base",
        "whisper-small" => "openai/whisper-small",
        _ => "openai/whisper-tiny",
    };
    let base_repo = cfg
        .custom_hf_repo
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or(default_repo);
    let aug = &project.audio_aug;

    format!(
        r#"import subprocess
import sys

write_status("installing", "Installing Transformers, Librosa, and Optimum ONNX exporter", total_epochs=int(TRAIN_CFG["epochs"]), progress_pct=24.0)
run_logged_cmd([sys.executable, "-m", "pip", "install", "-q", "transformers", "accelerate", "datasets", "librosa", "soundfile", "onnx", "optimum[onnxruntime]"])

import librosa
import numpy as np
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback

try:
    total_epochs = int(TRAIN_CFG["epochs"])
    batch_size = int(TRAIN_CFG["batchSize"])
    lr = float(TRAIN_CFG["learningRate"])

    with open(os.path.join(DATASET_DIR, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    target_repo = resolve_hf_train_repo("{base_repo}")
    processor = WhisperProcessor.from_pretrained(target_repo, trust_remote_code=True)
    model = WhisperForConditionalGeneration.from_pretrained(target_repo, trust_remote_code=True)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    class WhisperSliceDataset(Dataset):
        def __init__(self, rows, augment):
            self.rows = [r for r in rows if r.get("transcript", "").strip()]
            self.augment = augment
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, idx):
            r = self.rows[idx]
            dur = max(0.2, float(r["end_sec"]) - float(r["start_sec"]))
            y, _ = librosa.load(os.path.join(DATASET_DIR, r["audio_path"]), sr=16000, offset=float(r["start_sec"]), duration=dur, mono=True)
            if self.augment and {noise} > 0:
                y = y + np.random.normal(0.0, {noise}, size=y.shape).astype(np.float32)
            feats = processor.feature_extractor(y, sampling_rate=16000).input_features[0]
            labels = processor.tokenizer(r["transcript"]).input_ids
            return {{"input_features": feats, "labels": labels}}

    @dataclass
    class DataCollatorSpeech:
        def __call__(self, features):
            input_features = [{{"input_features": f["input_features"]}} for f in features]
            batch = processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{{"input_ids": f["labels"]}} for f in features]
            labels_batch = processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels
            return batch

    train_rows = [r for r in manifest if r.get("split") != "val"]
    val_rows = [r for r in manifest if r.get("split") == "val"] or train_rows[:5]
    train_ds = WhisperSliceDataset(train_rows, True)
    if len(train_ds) == 0:
        raise RuntimeError("No labeled audio transcripts found for Whisper training.")
    val_ds = WhisperSliceDataset(val_rows, False) or train_ds

    _wh_state = {{"ran": False, "t0": time.time(), "max_secs": 0, "target_ep": total_epochs}}

    class DriveCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or ("loss" not in logs and "eval_loss" not in logs):
                return
            ep = float(state.epoch or 0.0)
            pct = 25.0 + min(1.0, ep / max(1.0, float(total_epochs))) * 63.0
            tloss = _safe_float(logs.get("loss"))
            vloss = _safe_float(logs.get("eval_loss"))
            METRICS_HISTORY.append({{
                "step": int(state.global_step),
                "epoch": ep,
                "trainLoss": tloss,
                "valLoss": vloss,
                "score": None,
                "scoreName": None,
                "lr": _safe_float(logs.get("learning_rate"))
            }})
            parts = []
            if tloss is not None:
                parts.append(f"loss={{tloss:.4f}}")
            if vloss is not None:
                parts.append(f"eval_loss={{vloss:.4f}}")
            append_log(f"Step {{state.global_step}} epoch={{ep:.2f}} " + " ".join(parts))
            write_status("training", f"Step {{state.global_step}} (Epoch {{ep:.2f}} / {{total_epochs}})", current_epoch=ep, total_epochs=total_epochs, progress_pct=pct)
        def on_epoch_end(self, args, state, control, **kwargs):
            ep_done = round(float(state.epoch or 0.0))
            if ep_done >= _wh_state["target_ep"] or (_wh_state["max_secs"] > 0 and (time.time() - _wh_state["t0"]) >= _wh_state["max_secs"]):
                control.should_save = True
                control.should_training_stop = True

    args = Seq2SeqTrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=total_epochs,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=1,
        report_to=[]
    )
    trainer = Seq2SeqTrainer(
        args=args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorSpeech(),
        callbacks=[DriveCallback()]
    )

    def damnisium_train_chunk(start_ep, end_ep, max_secs=450):
        try:
            start_ep = int(start_ep)
            target_ep = min(total_epochs, int(end_ep))
            if start_ep >= total_epochs:
                emit_chunk_done(total_epochs, total_epochs)
                return total_epochs
            _wh_state["t0"] = time.time()
            _wh_state["max_secs"] = float(max_secs)
            _wh_state["target_ep"] = target_ep
            if not _wh_state["ran"]:
                write_status("training", "Fine-tuning Whisper Tiny", total_epochs=total_epochs, progress_pct=25.0)
                resume_ckpt = find_latest_ckpt_dir(RESUME_CKPT_DIR)
                _wh_state["ran"] = True
            else:
                resume_ckpt = find_latest_ckpt_dir(CHECKPOINT_DIR)
            trainer.train(resume_from_checkpoint=resume_ckpt or None)
            completed_ep = min(total_epochs, max(start_ep + 1, int(round(float(getattr(trainer.state, "epoch", target_ep) or target_ep)))))
            emit_chunk_done(completed_ep, total_epochs)
            return completed_ep
        except Exception as exc:
            clean_err = format_clean_exc(exc)
            append_log(f"Error: {{clean_err}}")
            write_status("failed", "Whisper training failed", total_epochs=total_epochs, error=clean_err)
            raise RuntimeError(clean_err) from None

    def damnisium_export_outputs():
        try:
            save_dir = os.path.join(WORK_DIR, "whisper_saved")
            model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            shutil.make_archive(os.path.join(OUTPUT_DIR, "whisper_finetuned"), "zip", save_dir)
            artifacts = ["whisper_finetuned.zip"]

            if TRAIN_CFG.get("exportOnnx", True):
                append_log("Exporting Whisper ONNX runtime bundle")
                write_status("training", "Exporting Whisper ONNX", current_epoch=float(total_epochs), total_epochs=total_epochs, progress_pct=93.0)
                onnx_out_dir = os.path.join(OUTPUT_DIR, "onnx_model")
                _opt_cli = shutil.which("optimum-cli")
                if _opt_cli:
                    run_logged_cmd([_opt_cli, "export", "onnx", "--model", save_dir, "--task", "automatic-speech-recognition", onnx_out_dir])
                else:
                    run_logged_cmd([sys.executable, "-m", "optimum.exporters.onnx", "--model", save_dir, "--task", "automatic-speech-recognition", onnx_out_dir])
                artifacts.append("onnx_model")

            last_val_loss = next((m["valLoss"] for m in reversed(METRICS_HISTORY) if m.get("valLoss") is not None), None)
            write_eval_bundle_meta(
                {{
                    "runtimeKind": "hf_asr",
                    "classes": []
                }},
                {{
                    "primaryName": "eval_loss",
                    "primaryValue": last_val_loss,
                    "valAccuracy": None,
                    "valLoss": last_val_loss,
                    "map50": None,
                    "map50_95": None,
                    "precision": None,
                    "recall": None,
                    "lossCurve": METRICS_HISTORY,
                    "confusion": None
                }}
            )
            artifacts.extend(["labels.json", "metrics.json"])
            finalize_completed_run("Whisper training and ONNX export completed", total_epochs, artifacts)
        except Exception as exc:
            clean_err = format_clean_exc(exc)
            append_log(f"Error: {{clean_err}}")
            write_status("failed", "Whisper export failed", total_epochs=total_epochs, error=clean_err)
            raise RuntimeError(clean_err) from None
except Exception as exc:
    clean_err = format_clean_exc(exc)
    append_log(f"Error: {{clean_err}}")
    write_status("failed", "Whisper setup failed", total_epochs=int(TRAIN_CFG["epochs"]), error=clean_err)
    raise RuntimeError(clean_err) from None
"#,
        noise = aug.noise_level
    )
}

fn build_distilbert_cell(_project: &Project, cfg: &TrainingConfig) -> String {
    let default_repo = match cfg.model_id.as_str() {
        "roberta-base" => "FacebookAI/roberta-base",
        "minilm-l12" => "microsoft/MiniLM-L12-H384-uncased",
        _ => "distilbert-base-uncased",
    };
    let base_repo = cfg
        .custom_hf_repo
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or(default_repo);

    format!(
        r#"import subprocess
import sys

write_status("installing", "Installing Transformers and Optimum ONNX exporter", total_epochs=int(TRAIN_CFG["epochs"]), progress_pct=24.0)
run_logged_cmd([sys.executable, "-m", "pip", "install", "-q", "transformers", "accelerate", "onnx", "optimum[onnxruntime]"])

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback

try:
    total_epochs = int(TRAIN_CFG["epochs"])
    batch_size = int(TRAIN_CFG["batchSize"])
    max_len = int(TRAIN_CFG["maxSeqLen"])
    lr = float(TRAIN_CFG["learningRate"])

    with open(os.path.join(DATASET_DIR, "classes.json"), "r", encoding="utf-8") as f:
        class_list = json.load(f)
    label_to_id = {{c: i for i, c in enumerate(class_list)}}
    id_to_label = {{i: c for i, c in enumerate(class_list)}}

    def load_jsonl(fname):
        rows = []
        p = os.path.join(DATASET_DIR, fname)
        if not os.path.exists(p):
            return rows
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                lbl = obj.get("label")
                if lbl in label_to_id and obj.get("prompt", "").strip():
                    rows.append((obj["prompt"], label_to_id[lbl]))
        return rows

    train_rows = load_jsonl("train.jsonl")
    if len(train_rows) == 0:
        raise RuntimeError("No labeled intent samples found for DistilBERT training.")
    val_rows = load_jsonl("val.jsonl") or train_rows[:5]

    target_repo = resolve_hf_train_repo("{base_repo}")
    tokenizer = AutoTokenizer.from_pretrained(target_repo, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        target_repo,
        num_labels=len(class_list),
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
        trust_remote_code=True
    )

    class IntentDataset(Dataset):
        def __init__(self, rows):
            self.enc = tokenizer([r[0] for r in rows], truncation=True, padding=True, max_length=max_len)
            self.labels = [r[1] for r in rows]
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            item = {{k: torch.tensor(v[idx]) for k, v in self.enc.items()}}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {{"accuracy": float((preds == labels).mean())}}

    _db_state = {{"ran": False, "t0": time.time(), "max_secs": 0, "target_ep": total_epochs}}

    class DriveCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or ("loss" not in logs and "eval_loss" not in logs):
                return
            ep = float(state.epoch or 0.0)
            pct = 25.0 + min(1.0, ep / max(1.0, float(total_epochs))) * 63.0
            tloss = _safe_float(logs.get("loss"))
            vloss = _safe_float(logs.get("eval_loss"))
            eacc = _safe_float(logs.get("eval_accuracy"))
            METRICS_HISTORY.append({{
                "step": int(state.global_step),
                "epoch": ep,
                "trainLoss": tloss,
                "valLoss": vloss,
                "score": eacc,
                "scoreName": "val_acc" if "eval_accuracy" in logs else None,
                "lr": _safe_float(logs.get("learning_rate"))
            }})
            parts = []
            if tloss is not None:
                parts.append(f"loss={{tloss:.4f}}")
            if vloss is not None:
                parts.append(f"eval_loss={{vloss:.4f}}")
            if eacc is not None:
                parts.append(f"eval_acc={{eacc:.4f}}")
            append_log(f"Step {{state.global_step}} epoch={{ep:.2f}} " + " ".join(parts))
            write_status("training", f"Epoch {{ep:.2f}} / {{total_epochs}}", current_epoch=ep, total_epochs=total_epochs, progress_pct=pct)
        def on_epoch_end(self, args, state, control, **kwargs):
            ep_done = round(float(state.epoch or 0.0))
            if ep_done >= _db_state["target_ep"] or (_db_state["max_secs"] > 0 and (time.time() - _db_state["t0"]) >= _db_state["max_secs"]):
                control.should_save = True
                control.should_training_stop = True

    val_ds = IntentDataset(val_rows)
    args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=total_epochs,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=1,
        report_to=[]
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=IntentDataset(train_rows),
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[DriveCallback()]
    )

    def damnisium_train_chunk(start_ep, end_ep, max_secs=450):
        try:
            start_ep = int(start_ep)
            target_ep = min(total_epochs, int(end_ep))
            if start_ep >= total_epochs:
                emit_chunk_done(total_epochs, total_epochs)
                return total_epochs
            _db_state["t0"] = time.time()
            _db_state["max_secs"] = float(max_secs)
            _db_state["target_ep"] = target_ep
            if not _db_state["ran"]:
                write_status("training", "Training DistilBERT intent classifier", total_epochs=total_epochs, progress_pct=25.0)
                resume_ckpt = find_latest_ckpt_dir(RESUME_CKPT_DIR)
                _db_state["ran"] = True
            else:
                resume_ckpt = find_latest_ckpt_dir(CHECKPOINT_DIR)
            trainer.train(resume_from_checkpoint=resume_ckpt or None)
            completed_ep = min(total_epochs, max(start_ep + 1, int(round(float(getattr(trainer.state, "epoch", target_ep) or target_ep)))))
            emit_chunk_done(completed_ep, total_epochs)
            return completed_ep
        except Exception as exc:
            clean_err = format_clean_exc(exc)
            append_log(f"Error: {{clean_err}}")
            write_status("failed", "DistilBERT training failed", total_epochs=total_epochs, error=clean_err)
            raise RuntimeError(clean_err) from None

    def damnisium_export_outputs():
        try:
            pred_out = trainer.predict(val_ds)
            preds_arr = pred_out.predictions[0] if isinstance(pred_out.predictions, tuple) else pred_out.predictions
            pred_cls = np.argmax(preds_arr, axis=-1)
            true_cls = np.array([r[1] for r in val_rows])
            num_cls = len(class_list)
            conf_mat = [[0] * num_cls for _ in range(num_cls)]
            for t_i, p_i in zip(true_cls.tolist(), pred_cls.tolist()):
                conf_mat[t_i][p_i] += 1
            val_acc = float((pred_cls == true_cls).mean()) if len(true_cls) > 0 else 0.0

            save_dir = os.path.join(WORK_DIR, "distilbert_saved")
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            shutil.make_archive(os.path.join(OUTPUT_DIR, "distilbert_finetuned"), "zip", save_dir)
            artifacts = ["distilbert_finetuned.zip"]

            if TRAIN_CFG.get("exportOnnx", True):
                append_log("Exporting DistilBERT ONNX runtime bundle")
                write_status("training", "Exporting DistilBERT ONNX", current_epoch=float(total_epochs), total_epochs=total_epochs, progress_pct=93.0)
                onnx_out_dir = os.path.join(OUTPUT_DIR, "onnx_model")
                _opt_cli = shutil.which("optimum-cli")
                if _opt_cli:
                    run_logged_cmd([_opt_cli, "export", "onnx", "--model", save_dir, "--task", "text-classification", onnx_out_dir])
                else:
                    run_logged_cmd([sys.executable, "-m", "optimum.exporters.onnx", "--model", save_dir, "--task", "text-classification", onnx_out_dir])
                artifacts.append("onnx_model")

            last_val_loss = next((m["valLoss"] for m in reversed(METRICS_HISTORY) if m.get("valLoss") is not None), None)
            write_eval_bundle_meta(
                {{
                    "runtimeKind": "hf_text_cls",
                    "classes": class_list
                }},
                {{
                    "primaryName": "val_acc",
                    "primaryValue": val_acc,
                    "valAccuracy": val_acc,
                    "valLoss": last_val_loss,
                    "map50": None,
                    "map50_95": None,
                    "precision": None,
                    "recall": None,
                    "lossCurve": METRICS_HISTORY,
                    "confusion": {{"labels": class_list, "matrix": conf_mat}}
                }}
            )
            artifacts.extend(["labels.json", "metrics.json"])
            finalize_completed_run("DistilBERT training and ONNX export completed", total_epochs, artifacts)
        except Exception as exc:
            clean_err = format_clean_exc(exc)
            append_log(f"Error: {{clean_err}}")
            write_status("failed", "DistilBERT export failed", total_epochs=total_epochs, error=clean_err)
            raise RuntimeError(clean_err) from None
except Exception as exc:
    clean_err = format_clean_exc(exc)
    append_log(f"Error: {{clean_err}}")
    write_status("failed", "DistilBERT setup failed", total_epochs=int(TRAIN_CFG["epochs"]), error=clean_err)
    raise RuntimeError(clean_err) from None
"#
    )
}

fn build_qwen_lora_cell(_project: &Project, cfg: &TrainingConfig) -> String {
    let default_repo = match cfg.model_id.as_str() {
        "qwen2.5-1.5b" => "Qwen/Qwen2.5-1.5B-Instruct",
        "dolphin3-llama3.2-1b" => "dphn/Dolphin3.0-Llama3.2-1B",
        "smollm2-360m" => "HuggingFaceTB/SmolLM2-360M-Instruct",
        _ => "Qwen/Qwen2.5-0.5B-Instruct",
    };
    let base_repo = cfg
        .custom_hf_repo
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or(default_repo);
    let use_qlora = if cfg.use_qlora { "True" } else { "False" };
    let bnb_pkg = if cfg.use_qlora { "\"bitsandbytes\", " } else { "" };

    format!(
        r#"import subprocess
import sys

write_status("installing", "Installing Transformers, PEFT, and Optimum ONNX exporter", total_epochs=int(TRAIN_CFG["epochs"]), progress_pct=24.0)
run_logged_cmd([sys.executable, "-m", "pip", "install", "-q", "-U", {bnb_pkg}"transformers", "peft", "accelerate", "onnx", "optimum[onnxruntime]"])

import torch
from torch.utils.data import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback, default_data_collator, BitsAndBytesConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

def _load_hf_causal_lm_resilient(repo_id, use_qlora_flag, dtype_val):
    load_kw = {{"trust_remote_code": True}}
    if use_qlora_flag and torch.cuda.is_available():
        load_kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        load_kw["device_map"] = "auto"
    else:
        load_kw["torch_dtype"] = dtype_val
    try:
        tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(repo_id, **load_kw)
        if use_qlora_flag and torch.cuda.is_available():
            mdl = prepare_model_for_kbit_training(mdl)
        return tok, mdl
    except (KeyError, ValueError) as exc:
        err_s = str(exc)
        low_s = err_s.lower()
        if "does not recognize this architecture" not in low_s and "model type" not in low_s and not isinstance(exc, KeyError):
            raise
        mtype = "unknown"
        raw_cfg = {{}}
        try:
            from huggingface_hub import hf_hub_download
            cfg_file = hf_hub_download(repo_id, "config.json")
            with open(cfg_file, "r", encoding="utf-8") as cf:
                raw_cfg = json.load(cf)
            mtype = str(raw_cfg.get("model_type") or mtype)
        except Exception:
            pass
        mlow = mtype.lower()
        fam_candidates = []
        if "deepseek" in mlow:
            fam_candidates.extend(["deepseek_v3", "deepseek_v2"])
        elif "qwen" in mlow:
            fam_candidates.extend(["qwen3", "qwen2"])
        elif "llama" in mlow:
            fam_candidates.append("llama")
        elif "mistral" in mlow:
            fam_candidates.append("mistral")
        elif "gemma" in mlow:
            fam_candidates.extend(["gemma3", "gemma2", "gemma"])
        for fam in fam_candidates:
            if fam in CONFIG_MAPPING:
                try:
                    patched = dict(raw_cfg)
                    patched["model_type"] = fam
                    cfg_cls = CONFIG_MAPPING[fam]
                    compat_cfg = cfg_cls.from_dict(patched)
                    tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
                    mdl = AutoModelForCausalLM.from_pretrained(repo_id, config=compat_cfg, **load_kw)
                    if use_qlora_flag and torch.cuda.is_available():
                        mdl = prepare_model_for_kbit_training(mdl)
                    append_log(f"Loaded {{repo_id}} via compatible config family '{{fam}}'")
                    return tok, mdl
                except Exception:
                    pass
        raise ValueError(
            f"Model '{{repo_id}}' uses custom architecture '{{mtype}}' which is not supported by Transformers AutoModelForCausalLM. Select a standard Causal LM checkpoint (e.g. Qwen2.5, Dolphin-Llama-3.2, SmolLM2, Mistral)."
        ) from None

try:
    total_epochs = int(TRAIN_CFG["epochs"])
    batch_size = int(TRAIN_CFG["batchSize"])
    max_len = int(TRAIN_CFG["maxSeqLen"])
    lr = float(TRAIN_CFG["learningRate"])

    def load_texts(fname):
        samples = []
        p = os.path.join(DATASET_DIR, fname)
        if not os.path.exists(p):
            return samples
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj.get("prompt", "").strip()
                completion = obj.get("completion", "").strip()
                system = obj.get("system", "").strip()
                if not prompt or not completion:
                    continue
                if system:
                    txt = f"System: {{system}}\nUser: {{prompt}}\nAssistant: {{completion}}"
                else:
                    txt = f"User: {{prompt}}\nAssistant: {{completion}}"
                samples.append(txt)
        return samples

    train_texts = load_texts("train.jsonl")
    if len(train_texts) == 0:
        raise RuntimeError("No prompt/completion samples found for Causal LM training.")
    val_texts = load_texts("val.jsonl") or train_texts[:4]

    target_repo = resolve_hf_train_repo("{base_repo}")
    append_log(f"Loading causal LM base {{target_repo}}")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer, base_model = _load_hf_causal_lm_resilient(target_repo, {use_qlora}, dtype)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules="all-linear")
        model = get_peft_model(base_model, lora_cfg)
    except Exception:
        lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model = get_peft_model(base_model, lora_cfg)
    for _p in model.parameters():
        if _p.requires_grad:
            _p.data = _p.data.float()

    class CausalTextDataset(Dataset):
        def __init__(self, texts):
            self.enc = tokenizer([t + tokenizer.eos_token for t in texts], truncation=True, padding="max_length", max_length=max_len)
        def __len__(self):
            return len(self.enc["input_ids"])
        def __getitem__(self, idx):
            input_ids = torch.tensor(self.enc["input_ids"][idx], dtype=torch.long)
            attn = torch.tensor(self.enc["attention_mask"][idx], dtype=torch.long)
            labels = input_ids.clone()
            labels[attn == 0] = -100
            return {{"input_ids": input_ids, "attention_mask": attn, "labels": labels}}

    _qw_state = {{"ran": False, "t0": time.time(), "max_secs": 0, "target_ep": total_epochs}}

    class DriveCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or ("loss" not in logs and "eval_loss" not in logs):
                return
            ep = float(state.epoch or 0.0)
            pct = 25.0 + min(1.0, ep / max(1.0, float(total_epochs))) * 63.0
            tloss = _safe_float(logs.get("loss"))
            vloss = _safe_float(logs.get("eval_loss"))
            METRICS_HISTORY.append({{
                "step": int(state.global_step),
                "epoch": ep,
                "trainLoss": tloss,
                "valLoss": vloss,
                "score": None,
                "scoreName": None,
                "lr": _safe_float(logs.get("learning_rate"))
            }})
            parts = []
            if tloss is not None:
                parts.append(f"loss={{tloss:.4f}}")
            if vloss is not None:
                parts.append(f"eval_loss={{vloss:.4f}}")
            append_log(f"Step {{state.global_step}} epoch={{ep:.2f}} " + " ".join(parts))
            write_status("training", f"Step {{state.global_step}} (Epoch {{ep:.2f}} / {{total_epochs}})", current_epoch=ep, total_epochs=total_epochs, progress_pct=pct)
        def on_epoch_end(self, args, state, control, **kwargs):
            ep_done = round(float(state.epoch or 0.0))
            if ep_done >= _qw_state["target_ep"] or (_qw_state["max_secs"] > 0 and (time.time() - _qw_state["t0"]) >= _qw_state["max_secs"]):
                control.should_save = True
                control.should_training_stop = True

    args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        num_train_epochs=total_epochs,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=1,
        report_to=[]
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=CausalTextDataset(train_texts),
        eval_dataset=CausalTextDataset(val_texts),
        data_collator=default_data_collator,
        callbacks=[DriveCallback()]
    )

    def damnisium_train_chunk(start_ep, end_ep, max_secs=450):
        try:
            start_ep = int(start_ep)
            target_ep = min(total_epochs, int(end_ep))
            if start_ep >= total_epochs:
                emit_chunk_done(total_epochs, total_epochs)
                return total_epochs
            _qw_state["t0"] = time.time()
            _qw_state["max_secs"] = float(max_secs)
            _qw_state["target_ep"] = target_ep
            if not _qw_state["ran"]:
                write_status("training", "Fine-tuning Causal LM with LoRA", total_epochs=total_epochs, progress_pct=25.0)
                resume_ckpt = find_latest_ckpt_dir(RESUME_CKPT_DIR)
                _qw_state["ran"] = True
            else:
                resume_ckpt = find_latest_ckpt_dir(CHECKPOINT_DIR)
            trainer.train(resume_from_checkpoint=resume_ckpt or None)
            completed_ep = min(total_epochs, max(start_ep + 1, int(round(float(getattr(trainer.state, "epoch", target_ep) or target_ep)))))
            emit_chunk_done(completed_ep, total_epochs)
            return completed_ep
        except Exception as exc:
            clean_err = format_clean_exc(exc)
            append_log(f"Error: {{clean_err}}")
            write_status("failed", "Qwen LoRA training failed", total_epochs=total_epochs, error=clean_err)
            raise RuntimeError(clean_err) from None

    def damnisium_export_outputs():
        global model, base_model, trainer
        try:
            save_dir = os.path.join(WORK_DIR, "qwen_lora_saved")
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            shutil.make_archive(os.path.join(OUTPUT_DIR, "qwen_lora_adapter"), "zip", save_dir)
            artifacts = ["qwen_lora_adapter.zip"]

            if TRAIN_CFG.get("exportOnnx", True):
                append_log("Merging LoRA weights and exporting Qwen ONNX bundle")
                write_status("training", "Exporting Qwen ONNX runtime bundle", current_epoch=float(total_epochs), total_epochs=total_epochs, progress_pct=92.0)
                if {use_qlora} and torch.cuda.is_available():
                    del trainer
                    del model
                    del base_model
                    torch.cuda.empty_cache()
                    fp_base = AutoModelForCausalLM.from_pretrained(target_repo, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True)
                    peft_m = PeftModel.from_pretrained(fp_base, save_dir)
                    merged = peft_m.merge_and_unload().cpu()
                else:
                    merged = model.merge_and_unload().cpu()
                merged_dir = os.path.join(WORK_DIR, "qwen_merged")
                merged.save_pretrained(merged_dir)
                tokenizer.save_pretrained(merged_dir)
                onnx_out_dir = os.path.join(OUTPUT_DIR, "onnx_model")
                _opt_cli = shutil.which("optimum-cli")
                if _opt_cli:
                    run_logged_cmd([_opt_cli, "export", "onnx", "--model", merged_dir, "--task", "text-generation-with-past", onnx_out_dir])
                else:
                    run_logged_cmd([sys.executable, "-m", "optimum.exporters.onnx", "--model", merged_dir, "--task", "text-generation-with-past", onnx_out_dir])
                artifacts.append("onnx_model")

            last_val_loss = next((m["valLoss"] for m in reversed(METRICS_HISTORY) if m.get("valLoss") is not None), None)
            write_eval_bundle_meta(
                {{
                    "runtimeKind": "hf_text_gen",
                    "classes": []
                }},
                {{
                    "primaryName": "eval_loss",
                    "primaryValue": last_val_loss,
                    "valAccuracy": None,
                    "valLoss": last_val_loss,
                    "map50": None,
                    "map50_95": None,
                    "precision": None,
                    "recall": None,
                    "lossCurve": METRICS_HISTORY,
                    "confusion": None
                }}
            )
            artifacts.extend(["labels.json", "metrics.json"])
            finalize_completed_run("Qwen LoRA training and ONNX export completed", total_epochs, artifacts)
        except Exception as exc:
            clean_err = format_clean_exc(exc)
            append_log(f"Error: {{clean_err}}")
            write_status("failed", "Qwen LoRA export failed", total_epochs=total_epochs, error=clean_err)
            raise RuntimeError(clean_err) from None
except Exception as exc:
    clean_err = format_clean_exc(exc)
    append_log(f"Error: {{clean_err}}")
    write_status("failed", "Qwen LoRA setup failed", total_epochs=int(TRAIN_CFG["epochs"]), error=clean_err)
    raise RuntimeError(clean_err) from None
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{AudioAugConfig, ImageAugConfig, SplitRatio, TextAugConfig};

    fn dummy_project(module: ProjectModule, task: ProjectTask) -> Project {
        Project {
            id: "prj_test".into(),
            name: "Test".into(),
            module,
            task,
            app_version: APP_VERSION.into(),
            created_at: "2026-01-01T00:00:00Z".into(),
            updated_at: "2026-01-01T00:00:00Z".into(),
            classes: vec![],
            image_items: vec![],
            audio_items: vec![],
            text_items: vec![],
            image_aug: ImageAugConfig::default(),
            audio_aug: AudioAugConfig::default(),
            text_aug: TextAugConfig::default(),
            split_ratio: SplitRatio::default(),
        }
    }

    fn dummy_cfg(model_id: &str) -> TrainingConfig {
        TrainingConfig {
            project_id: "prj_test".into(),
            model_id: model_id.into(),
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
        }
    }

    #[test]
    fn test_all_colab_notebooks_valid_json_onnx_and_zero_python_comments() {
        let cases = vec![
            (ProjectModule::Image, ProjectTask::Detection, "yolov8n"),
            (ProjectModule::Image, ProjectTask::Segmentation, "yolo11n"),
            (ProjectModule::Image, ProjectTask::Classification, "mobilenet_v3_large"),
            (ProjectModule::Image, ProjectTask::Classification, "efficientnet_b0"),
            (ProjectModule::Audio, ProjectTask::Classification, "yamnet"),
            (ProjectModule::Audio, ProjectTask::Transcription, "whisper-tiny"),
            (ProjectModule::Text, ProjectTask::Intent, "distilbert-base-uncased"),
            (ProjectModule::Text, ProjectTask::Qa, "qwen2.5-0.5b"),
        ];
        let sync_ctx = NotebookDriveSyncContext {
            run_folder_id: "fld_run".into(),
            output_folder_id: "fld_out".into(),
            dataset_file_id: "fil_ds".into(),
            custom_weight_file_id: String::new(),
            status_file_id: "fil_status".into(),
            log_file_id: "fil_log".into(),
            output_bundle_id: "fil_bundle".into(),
        };
        for (module, task, model_id) in cases {
            let proj = dummy_project(module, task);
            let cfg = dummy_cfg(model_id);
            let bytes = build_colab_notebook("run_123", &proj, &cfg, false, &sync_ctx).unwrap();
            let parsed: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(parsed["nbformat"], 4);
            let cells = parsed["cells"].as_array().unwrap();
            let full_py: String = cells
                .iter()
                .filter(|c| c["cell_type"] == "code")
                .flat_map(|c| c["source"].as_array().unwrap().iter())
                .map(|v| v.as_str().unwrap())
                .collect();
            assert!(full_py.contains("write_eval_bundle_meta"));
            assert!(full_py.contains("finalize_completed_run"));
            assert!(full_py.contains("TeeStream"));
            assert!(full_py.contains("run_logged_cmd"));
            assert!(full_py.contains("CHECKPOINT_DIR"));
            assert!(full_py.contains("onnx"));
            assert!(full_py.contains("base64.b64decode"));
            assert!(full_py.contains("_sanitize_json_val"));
            assert!(!full_py.contains("os.replace(tmp_path, STATUS_PATH)"));
            for cell in cells {
                if cell["cell_type"] == "code" {
                    let src = cell["source"].as_array().unwrap();
                    for line in src {
                        let s = line.as_str().unwrap();
                        assert!(!s.trim_start().starts_with('#'), "Found Python comment: {s}");
                        assert!(!s.contains("\"\"\""), "Found docstring: {s}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_qlora_and_uncensored_hf_injection_with_resume() {
        let proj = dummy_project(ProjectModule::Text, ProjectTask::Qa);
        let mut cfg = dummy_cfg("custom-hf");
        cfg.is_custom_model = true;
        cfg.custom_hf_repo = Some("dphn/Dolphin3.0-Llama3.2-1B".into());
        cfg.resume_from_run_id = Some("run_prev_999".into());
        cfg.use_qlora = true;
        let bytes = build_colab_notebook(
            "run_456",
            &proj,
            &cfg,
            false,
            &NotebookDriveSyncContext::default(),
        )
        .unwrap();
        let raw = String::from_utf8(bytes).unwrap();
        assert!(raw.contains("dphn/Dolphin3.0-Llama3.2-1B"));
        assert!(raw.contains("BitsAndBytesConfig"));
        assert!(raw.contains("RESUME_FROM_RUN_ID"));
        assert!(raw.contains("default_data_collator"));
        assert!(raw.contains("PeftModel.from_pretrained"));
        let yolo_plan = build_headless_stages(
            "run_y1",
            &dummy_project(ProjectModule::Image, ProjectTask::Detection),
            &dummy_cfg("yolov8n"),
            false,
            "",
        )
        .unwrap();
        assert!(yolo_plan.setup_script.contains("_live_stdout()"));
        assert!(yolo_plan.prepare_script.contains("seen_local"));
        assert!(yolo_plan.prepare_script.contains("os.path.abspath(str(onnx_path)) != os.path.abspath(dst_onnx)"));

        let qwen_plan = build_headless_stages(
            "run_qw1",
            &proj,
            &cfg,
            false,
            "",
        )
        .unwrap();
        assert!(qwen_plan.prepare_script.contains("_p.data = _p.data.float()"));
        assert!(qwen_plan.prepare_script.contains("trust_remote_code=True"));
        assert!(qwen_plan.prepare_script.contains("_load_hf_causal_lm_resilient"));

        let cls_plan = build_headless_stages(
            "run_cls1",
            &dummy_project(ProjectModule::Image, ProjectTask::Classification),
            &dummy_cfg("mobilenet_v3_large"),
            false,
            "",
        )
        .unwrap();
        assert!(cls_plan.prepare_script.contains("drop_last=(len(train_ds) > batch_size"));

        let ym_plan = build_headless_stages(
            "run_ym1",
            &dummy_project(ProjectModule::Audio, ProjectTask::Classification),
            &dummy_cfg("yamnet"),
            false,
            "",
        )
        .unwrap();
        assert!(ym_plan.prepare_script.contains("if M_t.shape[0] == 1:"));

        let db_plan = build_headless_stages(
            "run_db1",
            &dummy_project(ProjectModule::Text, ProjectTask::Intent),
            &dummy_cfg("distilbert-base-uncased"),
            false,
            "",
        )
        .unwrap();
        assert!(db_plan.prepare_script.contains("ignore_mismatched_sizes=True"));
        assert!(db_plan.prepare_script.contains("isinstance(pred_out.predictions, tuple)"));
    }

    #[test]
    fn test_extended_module_models_map_correctly() {
        let checks = vec![
            (ProjectModule::Image, ProjectTask::Classification, "resnet18", "models.resnet18"),
            (ProjectModule::Audio, ProjectTask::Transcription, "whisper-base", "openai/whisper-base"),
            (ProjectModule::Audio, ProjectTask::Transcription, "whisper-small", "openai/whisper-small"),
            (ProjectModule::Text, ProjectTask::Intent, "roberta-base", "FacebookAI/roberta-base"),
            (ProjectModule::Text, ProjectTask::Intent, "minilm-l12", "microsoft/MiniLM-L12-H384-uncased"),
            (ProjectModule::Text, ProjectTask::Qa, "smollm2-360m", "HuggingFaceTB/SmolLM2-360M-Instruct"),
            (ProjectModule::Text, ProjectTask::Completion, "dolphin3-llama3.2-1b", "dphn/Dolphin3.0-Llama3.2-1B"),
            (ProjectModule::Text, ProjectTask::Qa, "qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct"),
        ];
        for (module, task, model_id, expected_marker) in checks {
            let proj = dummy_project(module, task);
            let cfg = dummy_cfg(model_id);
            let bytes = build_colab_notebook(
                "run_ext",
                &proj,
                &cfg,
                false,
                &NotebookDriveSyncContext::default(),
            )
            .unwrap();
            let raw = String::from_utf8(bytes).unwrap();
            assert!(
                raw.contains(expected_marker),
                "Expected {expected_marker} for {model_id}"
            );
            let parsed: Value = serde_json::from_str(&raw).unwrap();
            for cell in parsed["cells"].as_array().unwrap() {
                if cell["cell_type"] == "code" {
                    for line in cell["source"].as_array().unwrap() {
                        let s = line.as_str().unwrap();
                        assert!(!s.trim_start().starts_with('#'), "Found Python comment: {s}");
                        assert!(!s.contains("\"\"\""), "Found docstring: {s}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_generated_colab_python_ast_parses_cleanly() {
        let sync_ctx = NotebookDriveSyncContext {
            run_folder_id: "fld_run".into(),
            output_folder_id: "fld_out".into(),
            dataset_file_id: "fil_ds".into(),
            custom_weight_file_id: "fil_wt".into(),
            status_file_id: "fil_status".into(),
            log_file_id: "fil_log".into(),
            output_bundle_id: "fil_bundle".into(),
        };
        let combos = vec![
            (ProjectModule::Image, ProjectTask::Detection, "yolov8n", true, false),
            (ProjectModule::Image, ProjectTask::Segmentation, "yolo11n", false, false),
            (ProjectModule::Image, ProjectTask::Classification, "mobilenet_v3_large", true, false),
            (ProjectModule::Audio, ProjectTask::Classification, "yamnet", false, false),
            (ProjectModule::Audio, ProjectTask::Transcription, "whisper-tiny", false, false),
            (ProjectModule::Text, ProjectTask::Intent, "distilbert-base-uncased", false, false),
            (ProjectModule::Text, ProjectTask::Qa, "qwen2.5-0.5b", false, true),
        ];
        for (idx, (module, task, model_id, custom_w, qlora)) in combos.into_iter().enumerate() {
            let proj = dummy_project(module, task);
            let mut cfg = dummy_cfg(model_id);
            cfg.use_qlora = qlora;
            let bytes = build_colab_notebook(&format!("run_ast_{idx}"), &proj, &cfg, custom_w, &sync_ctx).unwrap();
            let parsed: Value = serde_json::from_slice(&bytes).unwrap();
            for (cidx, cell) in parsed["cells"].as_array().unwrap().iter().enumerate() {
                if cell["cell_type"] == "code" {
                    let src: String = cell["source"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_str().unwrap())
                        .collect();
                    let tmp_py = std::env::temp_dir().join(format!("damnisium_ast_{idx}_{cidx}.py"));
                    std::fs::write(&tmp_py, src.as_bytes()).unwrap();
                    let out = std::process::Command::new("python")
                        .args(["-c", "import ast, sys, pathlib; ast.parse(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))"])
                        .arg(&tmp_py)
                        .output()
                        .unwrap();
                    let _ = std::fs::remove_file(&tmp_py);
                    assert!(
                        out.status.success(),
                        "Python AST parse failed for {model_id} cell {cidx}: {}",
                        String::from_utf8_lossy(&out.stderr)
                    );
                }
            }
            let plan = build_headless_stages(&format!("run_hl_{idx}"), &proj, &cfg, custom_w, "").unwrap();
            let chunk_call = HeadlessStagePlan::chunk_script(0, 5, 450);
            for (sname, src) in [
                ("setup", plan.setup_script),
                ("prepare", plan.prepare_script),
                ("chunk", chunk_call),
                ("export", plan.export_script),
            ] {
                for line in src.lines() {
                    assert!(!line.trim_start().starts_with('#'), "Found headless comment in {sname}: {line}");
                    assert!(!line.contains("\"\"\""), "Found headless docstring in {sname}: {line}");
                }
                let tmp_py = std::env::temp_dir().join(format!("damnisium_hl_ast_{idx}_{sname}.py"));
                std::fs::write(&tmp_py, src.as_bytes()).unwrap();
                let out = std::process::Command::new("python")
                    .args(["-c", "import ast, sys, pathlib; ast.parse(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))"])
                    .arg(&tmp_py)
                    .output()
                    .unwrap();
                let _ = std::fs::remove_file(&tmp_py);
                assert!(
                    out.status.success(),
                    "Headless Python AST parse failed for {model_id} {sname}: {}",
                    String::from_utf8_lossy(&out.stderr)
                );
            }
        }
    }
}
