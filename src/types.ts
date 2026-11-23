export type ProjectModule = "image" | "audio" | "text";

export type ProjectTask =
  | "detection"
  | "segmentation"
  | "classification"
  | "transcription"
  | "qa"
  | "completion"
  | "intent";

export type SplitType = "train" | "val" | "test";

export type RunState =
  | "uploading"
  | "waiting_for_colab"
  | "installing"
  | "training"
  | "downloading_weights"
  | "completed"
  | "failed"
  | "exported_offline";

export type AppView = "dashboard" | "workspace" | "train" | "evaluate" | "hub" | "settings";

export interface LabelClass {
  id: string;
  name: string;
  color: string;
}

export interface BoundingBox {
  id: string;
  classId: string;
  x: number;
  y: number;
  w: number;
  h: number;
  confidence?: number | null;
}

export interface PolygonPoint {
  x: number;
  y: number;
}

export interface PolygonShape {
  id: string;
  classId: string;
  points: PolygonPoint[];
}

export interface ImageItem {
  id: string;
  filename: string;
  relPath: string;
  thumbRelPath: string;
  width: number;
  height: number;
  split: SplitType;
  boxes: BoundingBox[];
  polygons: PolygonShape[];
  imageLabels: string[];
  createdAt: string;
}

export interface AudioSlice {
  id: string;
  startSec: number;
  endSec: number;
  classId: string | null;
  transcript: string;
}

export interface AudioItem {
  id: string;
  filename: string;
  relPath: string;
  durationSec: number;
  sampleRate: number;
  channels: number;
  split: SplitType;
  slices: AudioSlice[];
  createdAt: string;
}

export interface TextItem {
  id: string;
  prompt: string;
  completion: string;
  system: string;
  intentClassId: string | null;
  split: SplitType;
  createdAt: string;
}

export interface ImageAugConfig {
  flipHorizontal: boolean;
  flipVertical: boolean;
  rotateDeg: number;
  brightnessPct: number;
  contrastPct: number;
  blurPx: number;
  mosaic: boolean;
  mixup: boolean;
}

export interface AudioAugConfig {
  noiseLevel: number;
  pitchSemitones: number;
  speedFactor: number;
  timeShiftPct: number;
  gainDb: number;
  specAugment: boolean;
}

export interface TextAugConfig {
  shufflePromptOrder: boolean;
  tokenDropoutPct: number;
  caseJitter: boolean;
}

export interface SplitRatio {
  train: number;
  val: number;
  test: number;
}

export interface Project {
  id: string;
  name: string;
  module: ProjectModule;
  task: ProjectTask;
  appVersion: string;
  createdAt: string;
  updatedAt: string;
  classes: LabelClass[];
  imageItems: ImageItem[];
  audioItems: AudioItem[];
  textItems: TextItem[];
  imageAug: ImageAugConfig;
  audioAug: AudioAugConfig;
  textAug: TextAugConfig;
  splitRatio: SplitRatio;
}

export interface ProjectSummary {
  id: string;
  name: string;
  module: ProjectModule;
  task: ProjectTask;
  itemCount: number;
  labeledCount: number;
  classCount: number;
  updatedAt: string;
}

export interface AppConfig {
  oauthClientId: string;
  oauthClientSecret: string;
  dataDir: string;
  hfToken: string;
  gcpProjectId: string;
  enableDriveBackup: boolean;
}

export interface GoogleUser {
  email: string;
  name: string;
  picture: string;
  sub?: string;
}

export interface PreparedOAuthSession {
  authUrl: string;
  redirectUri: string;
  expiresInSec: number;
}

export interface OAuthProgressPayload {
  phase: "verifying" | "verified" | "error" | string;
  message: string;
  error: string | null;
  auth: GoogleAuthStatus | null;
}

export interface GoogleAuthStatus {
  connected: boolean;
  user: GoogleUser | null;
  hasCredentials: boolean;
  driveFolderId: string | null;
  scopes: string;
  colabAuthorized: boolean;
}

export interface ColabRuntimeSpec {
  variant: string;
  accelerator: string;
  shape: string;
  eligible?: boolean;
}

export interface PretrainedModelSpec {
  id: string;
  name: string;
  module: ProjectModule;
  tasks: ProjectTask[];
  architecture: string;
  paramCount: string;
  sizeLabel: string;
  speedTier: string;
  supportsOnnxExport: boolean;
  hfRepoId?: string | null;
}

export interface AutoLabelModelInfo {
  id: string;
  name: string;
  filename: string;
  localPath: string;
  downloaded: boolean;
  sizeBytes: number;
  inputSize: number;
  approxMb: number;
  isCustom: boolean;
  classes?: string[];
}

export interface AutoLabelProgressPayload {
  modelId: string;
  downloadedBytes: number;
  totalBytes: number | null;
  done: boolean;
}

export interface CustomModel {
  id: string;
  name: string;
  module: ProjectModule;
  task: ProjectTask;
  sourceKind: string;
  localRelPath: string | null;
  hfRepoId: string | null;
  createdAt: string;
}

export interface TrainingConfig {
  projectId: string;
  modelId: string;
  isCustomModel: boolean;
  customHfRepo: string | null;
  customWeightRelPath: string | null;
  epochs: number;
  batchSize: number;
  learningRate: number;
  imgSize: number;
  maxSeqLen: number;
  exportOnnx: boolean;
  autoApplySplit: boolean;
  resumeFromRunId: string | null;
  useQlora: boolean;
  preferredAccelerator?: string | null;
  runtimeVariant?: string | null;
  customModelName?: string | null;
}

export interface TrainingMetricPoint {
  step: number;
  epoch: number;
  trainLoss: number | null;
  valLoss: number | null;
  score: number | null;
  scoreName: string | null;
  lr: number | null;
}

export interface TrainingRun {
  id: string;
  projectId: string;
  projectName: string;
  module: ProjectModule;
  task: ProjectTask;
  modelId: string;
  modelLabel: string;
  state: RunState;
  gpuName: string | null;
  currentEpoch: number;
  totalEpochs: number;
  progressPct: number;
  etaSeconds: number | null;
  message: string;
  error: string | null;
  metrics: TrainingMetricPoint[];
  logTail: string[];
  colabUrl: string | null;
  driveRunFolderId: string | null;
  driveOutputFolderId: string | null;
  driveNotebookId: string | null;
  driveStatusFileId?: string | null;
  driveLogFileId?: string | null;
  driveOutputBundleId?: string | null;
  colabRuntimeName?: string | null;
  colabSessionName?: string | null;
  colabAccelerator?: string | null;
  colabLastExecutionId?: string | null;
  localBundleDir: string;
  weightsDir: string;
  downloadedWeights: string[];
  createdAt: string;
  updatedAt: string;
}

export interface DatasetExportResult {
  zipPath: string;
  itemCount: number;
  classCount: number;
}

export interface WorkspaceInfo {
  appName: string;
  appVersion: string;
  dataDir: string;
  config: AppConfig;
  auth: GoogleAuthStatus;
  pretrainedModels: PretrainedModelSpec[];
  customModels: CustomModel[];
}

export interface EvalConfusionMatrix {
  labels: string[];
  matrix: number[][];
}

export interface EvalMetricsMeta {
  primaryName: string | null;
  primaryValue: number | null;
  valAccuracy: number | null;
  valLoss: number | null;
  map50: number | null;
  map50_95: number | null;
  precision: number | null;
  recall: number | null;
  lossCurve: TrainingMetricPoint[];
  confusion: EvalConfusionMatrix | null;
}

export interface EvalBundleInfo {
  id: string;
  label: string;
  module: ProjectModule;
  task: ProjectTask;
  runtimeKind: string;
  onnxPath: string;
  onnxDir: string | null;
  labelsPath: string | null;
  metricsPath: string | null;
  classes: string[];
  inputSize: number;
  metrics: EvalMetricsMeta | null;
  source: string;
}

export interface TextEvalRow {
  prompt: string;
  expected: string;
  system: string;
}

export interface ConnectivityStatus {
  online: boolean;
  latencyMs: number | null;
  message: string;
  checkedAt: string;
}

export interface SplitCounts {
  train: number;
  val: number;
  test: number;
}

export interface PreflightIssue {
  severity: "error" | "warning" | string;
  code: string;
  message: string;
  itemIds: string[];
}

export interface PreflightReport {
  ok: boolean;
  totalItems: number;
  labeledItems: number;
  unlabeledItems: number;
  missingFiles: number;
  emptyClasses: string[];
  splitCounts: SplitCounts;
  issues: PreflightIssue[];
}

export interface HfModelFile {
  rfilename: string;
  sizeBytes: number | null;
  format: string;
  isMmproj: boolean;
  downloadUrl: string;
}

export interface HfHubModel {
  repoId: string;
  name: string;
  author: string;
  moduleHint: string;
  formatTags: string[];
  downloads: number;
  likes: number;
  trendingScore: number;
  lastModified: string;
  paramLabel: string | null;
  estimatedGb: number | null;
  totalWeightBytes?: number | null;
  sizeLabel?: string | null;
  over4gb: boolean;
  isPinned: boolean;
  summary: string;
  files: HfModelFile[];
}

export interface HfHubResponse {
  online: boolean;
  models: HfHubModel[];
  warning: string | null;
}

export interface HfDownloadProgressPayload {
  repoId: string;
  filename: string;
  downloadedBytes: number;
  totalBytes: number | null;
  done: boolean;
}

export interface LocalGgufModel {
  id: string;
  name: string;
  repoId: string;
  filename: string;
  localPath: string;
  sizeBytes: number;
  mmprojPath: string | null;
}

export interface LlamaRuntimeStatus {
  installed: boolean;
  running: boolean;
  binaryPath: string;
  activeModelPath: string | null;
  releaseTag: string | null;
  downloadSizeMb: number;
}

export interface GgufGenerationResult {
  output: string;
  tokensPerSec: number;
  durationMs: number;
}

export interface ExportBundleResult {
  outputPath: string;
  isZip: boolean;
  files: string[];
  pythonSnippet: string;
  rustSnippet: string;
}
