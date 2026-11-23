use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProjectModule {
    Image,
    Audio,
    Text,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProjectTask {
    Detection,
    Segmentation,
    Classification,
    Transcription,
    Qa,
    Completion,
    Intent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SplitType {
    #[default]
    Train,
    Val,
    Test,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LabelClass {
    pub id: String,
    pub name: String,
    pub color: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BoundingBox {
    pub id: String,
    pub class_id: String,
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
    pub confidence: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PolygonPoint {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PolygonShape {
    pub id: String,
    pub class_id: String,
    pub points: Vec<PolygonPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageItem {
    pub id: String,
    pub filename: String,
    pub rel_path: String,
    pub thumb_rel_path: String,
    pub width: u32,
    pub height: u32,
    #[serde(default)]
    pub split: SplitType,
    #[serde(default)]
    pub boxes: Vec<BoundingBox>,
    #[serde(default)]
    pub polygons: Vec<PolygonShape>,
    #[serde(default)]
    pub image_labels: Vec<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioSlice {
    pub id: String,
    pub start_sec: f64,
    pub end_sec: f64,
    pub class_id: Option<String>,
    pub transcript: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioItem {
    pub id: String,
    pub filename: String,
    pub rel_path: String,
    pub duration_sec: f64,
    pub sample_rate: u32,
    pub channels: u16,
    #[serde(default)]
    pub split: SplitType,
    #[serde(default)]
    pub slices: Vec<AudioSlice>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextItem {
    pub id: String,
    pub prompt: String,
    pub completion: String,
    pub system: String,
    pub intent_class_id: Option<String>,
    #[serde(default)]
    pub split: SplitType,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageAugConfig {
    pub flip_horizontal: bool,
    pub flip_vertical: bool,
    pub rotate_deg: f64,
    pub brightness_pct: f64,
    pub contrast_pct: f64,
    pub blur_px: f64,
    pub mosaic: bool,
    pub mixup: bool,
}

impl Default for ImageAugConfig {
    fn default() -> Self {
        Self {
            flip_horizontal: true,
            flip_vertical: false,
            rotate_deg: 10.0,
            brightness_pct: 15.0,
            contrast_pct: 10.0,
            blur_px: 0.0,
            mosaic: true,
            mixup: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioAugConfig {
    pub noise_level: f64,
    pub pitch_semitones: f64,
    pub speed_factor: f64,
    pub time_shift_pct: f64,
    pub gain_db: f64,
    pub spec_augment: bool,
}

impl Default for AudioAugConfig {
    fn default() -> Self {
        Self {
            noise_level: 0.02,
            pitch_semitones: 2.0,
            speed_factor: 1.0,
            time_shift_pct: 10.0,
            gain_db: 3.0,
            spec_augment: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextAugConfig {
    pub shuffle_prompt_order: bool,
    pub token_dropout_pct: f64,
    pub case_jitter: bool,
}

impl Default for TextAugConfig {
    fn default() -> Self {
        Self {
            shuffle_prompt_order: false,
            token_dropout_pct: 0.0,
            case_jitter: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SplitRatio {
    pub train: u8,
    pub val: u8,
    pub test: u8,
}

impl Default for SplitRatio {
    fn default() -> Self {
        Self {
            train: 80,
            val: 15,
            test: 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Project {
    pub id: String,
    pub name: String,
    pub module: ProjectModule,
    pub task: ProjectTask,
    pub app_version: String,
    pub created_at: String,
    pub updated_at: String,
    #[serde(default)]
    pub classes: Vec<LabelClass>,
    #[serde(default)]
    pub image_items: Vec<ImageItem>,
    #[serde(default)]
    pub audio_items: Vec<AudioItem>,
    #[serde(default)]
    pub text_items: Vec<TextItem>,
    #[serde(default)]
    pub image_aug: ImageAugConfig,
    #[serde(default)]
    pub audio_aug: AudioAugConfig,
    #[serde(default)]
    pub text_aug: TextAugConfig,
    #[serde(default)]
    pub split_ratio: SplitRatio,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectSummary {
    pub id: String,
    pub name: String,
    pub module: ProjectModule,
    pub task: ProjectTask,
    pub item_count: usize,
    pub labeled_count: usize,
    pub class_count: usize,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct AppConfig {
    pub oauth_client_id: String,
    pub oauth_client_secret: String,
    pub data_dir: String,
    #[serde(default)]
    pub hf_token: String,
    #[serde(default)]
    pub gcp_project_id: String,
    #[serde(default)]
    pub enable_drive_backup: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleUser {
    pub email: String,
    pub name: String,
    pub picture: String,
    #[serde(default)]
    pub sub: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct GoogleAuthStored {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_at_unix: i64,
    pub user: Option<GoogleUser>,
    pub drive_root_folder_id: Option<String>,
    #[serde(default)]
    pub scopes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleAuthStatus {
    pub connected: bool,
    pub user: Option<GoogleUser>,
    pub has_credentials: bool,
    pub drive_folder_id: Option<String>,
    #[serde(default)]
    pub scopes: String,
    #[serde(default)]
    pub colab_authorized: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PretrainedModelSpec {
    pub id: String,
    pub name: String,
    pub module: ProjectModule,
    pub tasks: Vec<ProjectTask>,
    pub architecture: String,
    pub param_count: String,
    pub size_label: String,
    pub speed_tier: String,
    pub supports_onnx_export: bool,
    #[serde(default)]
    pub hf_repo_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AutoLabelModelSpec {
    pub id: String,
    pub name: String,
    pub filename: String,
    pub download_url: String,
    pub input_size: u32,
    pub approx_mb: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AutoLabelModelInfo {
    pub id: String,
    pub name: String,
    pub filename: String,
    pub local_path: String,
    pub downloaded: bool,
    pub size_bytes: u64,
    pub input_size: u32,
    pub approx_mb: u32,
    pub is_custom: bool,
    #[serde(default)]
    pub classes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AutoLabelProgressPayload {
    pub model_id: String,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub done: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomModel {
    pub id: String,
    pub name: String,
    pub module: ProjectModule,
    pub task: ProjectTask,
    pub source_kind: String,
    pub local_rel_path: Option<String>,
    pub hf_repo_id: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingConfig {
    pub project_id: String,
    pub model_id: String,
    pub is_custom_model: bool,
    pub custom_hf_repo: Option<String>,
    pub custom_weight_rel_path: Option<String>,
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f64,
    pub img_size: u32,
    pub max_seq_len: u32,
    pub export_onnx: bool,
    pub auto_apply_split: bool,
    #[serde(default)]
    pub resume_from_run_id: Option<String>,
    #[serde(default)]
    pub use_qlora: bool,
    #[serde(default)]
    pub preferred_accelerator: Option<String>,
    #[serde(default)]
    pub runtime_variant: Option<String>,
    #[serde(default)]
    pub custom_model_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunState {
    Uploading,
    WaitingForColab,
    Installing,
    Training,
    DownloadingWeights,
    Completed,
    Failed,
    ExportedOffline,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingMetricPoint {
    pub step: u32,
    pub epoch: f64,
    pub train_loss: Option<f64>,
    pub val_loss: Option<f64>,
    pub score: Option<f64>,
    pub score_name: Option<String>,
    pub lr: Option<f64>,
}

fn default_spec_eligible() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ColabRuntimeSpec {
    #[serde(default)]
    pub variant: String,
    #[serde(default)]
    pub accelerator: String,
    #[serde(default)]
    pub shape: String,
    #[serde(default = "default_spec_eligible")]
    pub eligible: bool,
}

impl Default for ColabRuntimeSpec {
    fn default() -> Self {
        Self {
            variant: String::new(),
            accelerator: String::new(),
            shape: String::new(),
            eligible: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingRun {
    pub id: String,
    pub project_id: String,
    pub project_name: String,
    pub module: ProjectModule,
    pub task: ProjectTask,
    pub model_id: String,
    pub model_label: String,
    pub state: RunState,
    pub gpu_name: Option<String>,
    pub current_epoch: f64,
    pub total_epochs: u32,
    pub progress_pct: f64,
    pub eta_seconds: Option<u64>,
    pub message: String,
    pub error: Option<String>,
    pub metrics: Vec<TrainingMetricPoint>,
    pub log_tail: Vec<String>,
    pub colab_url: Option<String>,
    pub drive_run_folder_id: Option<String>,
    pub drive_output_folder_id: Option<String>,
    pub drive_notebook_id: Option<String>,
    #[serde(default)]
    pub drive_status_file_id: Option<String>,
    #[serde(default)]
    pub drive_log_file_id: Option<String>,
    #[serde(default)]
    pub drive_output_bundle_id: Option<String>,
    #[serde(default)]
    pub colab_runtime_name: Option<String>,
    #[serde(default)]
    pub colab_session_name: Option<String>,
    #[serde(default)]
    pub colab_accelerator: Option<String>,
    #[serde(default)]
    pub colab_last_execution_id: Option<String>,
    pub local_bundle_dir: String,
    pub weights_dir: String,
    pub downloaded_weights: Vec<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ColabRemoteStatus {
    pub state: String,
    pub gpu: Option<String>,
    pub current_epoch: Option<f64>,
    pub total_epochs: Option<u32>,
    pub progress_pct: Option<f64>,
    pub eta_seconds: Option<u64>,
    pub message: Option<String>,
    pub error: Option<String>,
    #[serde(default)]
    pub metrics: Vec<TrainingMetricPoint>,
    #[serde(default)]
    pub artifacts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetExportResult {
    pub zip_path: String,
    pub item_count: usize,
    pub class_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkspaceInfo {
    pub app_name: String,
    pub app_version: String,
    pub data_dir: String,
    pub config: AppConfig,
    pub auth: GoogleAuthStatus,
    pub pretrained_models: Vec<PretrainedModelSpec>,
    pub custom_models: Vec<CustomModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvalConfusionMatrix {
    pub labels: Vec<String>,
    pub matrix: Vec<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct EvalLabelsMeta {
    #[serde(default)]
    pub runtime_kind: String,
    #[serde(default)]
    pub classes: Vec<String>,
    pub img_size: Option<u32>,
    pub sample_rate: Option<u32>,
    pub n_mels: Option<u32>,
    pub patch_frames: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct EvalMetricsMeta {
    pub primary_name: Option<String>,
    pub primary_value: Option<f64>,
    pub val_accuracy: Option<f64>,
    pub val_loss: Option<f64>,
    pub map50: Option<f64>,
    pub map50_95: Option<f64>,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    #[serde(default)]
    pub loss_curve: Vec<TrainingMetricPoint>,
    pub confusion: Option<EvalConfusionMatrix>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvalBundleInfo {
    pub id: String,
    pub label: String,
    pub module: ProjectModule,
    pub task: ProjectTask,
    pub runtime_kind: String,
    pub onnx_path: String,
    pub onnx_dir: Option<String>,
    pub labels_path: Option<String>,
    pub metrics_path: Option<String>,
    pub classes: Vec<String>,
    pub input_size: u32,
    pub metrics: Option<EvalMetricsMeta>,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextEvalRow {
    pub prompt: String,
    pub expected: String,
    pub system: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConnectivityStatus {
    pub online: bool,
    pub latency_ms: Option<u64>,
    pub message: String,
    pub checked_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SplitCounts {
    pub train: usize,
    pub val: usize,
    pub test: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreflightIssue {
    pub severity: String,
    pub code: String,
    pub message: String,
    pub item_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreflightReport {
    pub ok: bool,
    pub total_items: usize,
    pub labeled_items: usize,
    pub unlabeled_items: usize,
    pub missing_files: usize,
    pub empty_classes: Vec<String>,
    pub split_counts: SplitCounts,
    pub issues: Vec<PreflightIssue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HfModelFile {
    pub rfilename: String,
    pub size_bytes: Option<u64>,
    pub format: String,
    pub is_mmproj: bool,
    pub download_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HfHubModel {
    pub repo_id: String,
    pub name: String,
    pub author: String,
    pub module_hint: String,
    pub format_tags: Vec<String>,
    pub downloads: u64,
    pub likes: u64,
    pub trending_score: f64,
    pub last_modified: String,
    pub param_label: Option<String>,
    pub estimated_gb: Option<f64>,
    #[serde(default)]
    pub total_weight_bytes: Option<u64>,
    #[serde(default)]
    pub size_label: Option<String>,
    pub over_4gb: bool,
    pub is_pinned: bool,
    pub summary: String,
    pub files: Vec<HfModelFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HfHubResponse {
    pub online: bool,
    pub models: Vec<HfHubModel>,
    pub warning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HfDownloadProgressPayload {
    pub repo_id: String,
    pub filename: String,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub done: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LocalGgufModel {
    pub id: String,
    pub name: String,
    pub repo_id: String,
    pub filename: String,
    pub local_path: String,
    pub size_bytes: u64,
    pub mmproj_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlamaRuntimeStatus {
    pub installed: bool,
    pub running: bool,
    pub binary_path: String,
    pub active_model_path: Option<String>,
    pub release_tag: Option<String>,
    pub download_size_mb: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GgufGenerationResult {
    pub output: String,
    pub tokens_per_sec: f64,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportBundleResult {
    pub output_path: String,
    pub is_zip: bool,
    pub files: Vec<String>,
    pub python_snippet: String,
    pub rust_snippet: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreparedOAuthSession {
    pub auth_url: String,
    pub redirect_uri: String,
    pub expires_in_sec: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OAuthProgressPayload {
    pub phase: String,
    pub message: String,
    pub error: Option<String>,
    pub auth: Option<GoogleAuthStatus>,
}
