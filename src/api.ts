import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import type {
  AppConfig,
  AudioAugConfig,
  AudioSlice,
  AutoLabelModelInfo,
  AutoLabelProgressPayload,
  BoundingBox,
  ColabRuntimeSpec,
  ConnectivityStatus,
  CustomModel,
  DatasetExportResult,
  EvalBundleInfo,
  ExportBundleResult,
  GgufGenerationResult,
  GoogleAuthStatus,
  HfDownloadProgressPayload,
  HfHubModel,
  HfHubResponse,
  ImageAugConfig,
  LlamaRuntimeStatus,
  LocalGgufModel,
  OAuthProgressPayload,
  PolygonShape,
  PreflightReport,
  PreparedOAuthSession,
  Project,
  ProjectModule,
  ProjectSummary,
  ProjectTask,
  SplitRatio,
  SplitType,
  TextAugConfig,
  TextEvalRow,
  TextItem,
  TrainingConfig,
  TrainingRun,
  WorkspaceInfo,
} from "./types";

export const api = {
  getWorkspaceInfo: () => invoke<WorkspaceInfo>("get_workspace_info"),

  saveAppConfig: (
    oauthClientId: string,
    oauthClientSecret: string,
    hfToken?: string,
    gcpProjectId?: string,
    enableDriveBackup?: boolean
  ) =>
    invoke<AppConfig>("save_app_config", {
      oauthClientId,
      oauthClientSecret,
      hfToken: hfToken ?? null,
      gcpProjectId: gcpProjectId ?? null,
      enableDriveBackup: enableDriveBackup ?? null,
    }),

  changeDataDirectory: (newDir: string) =>
    invoke<WorkspaceInfo>("change_data_directory", { newDir }),

  prepareGoogleOAuth: () => invoke<PreparedOAuthSession>("prepare_google_oauth"),

  cancelGoogleOAuth: () => invoke<void>("cancel_google_oauth"),

  disconnectGoogleAccount: () => invoke<GoogleAuthStatus>("disconnect_google_account"),

  onGoogleOAuthState: (cb: (p: OAuthProgressPayload) => void): Promise<UnlistenFn> =>
    listen<OAuthProgressPayload>("google-oauth-state", (e) => cb(e.payload)),

  listProjects: () => invoke<ProjectSummary[]>("list_projects"),

  createProject: (name: string, module: ProjectModule, task: ProjectTask) =>
    invoke<Project>("create_project", { name, module, task }),

  loadProject: (projectId: string) => invoke<Project>("load_project", { projectId }),

  deleteProject: (projectId: string) => invoke<void>("delete_project", { projectId }),

  addLabelClass: (projectId: string, name: string, color: string | null = null) =>
    invoke<Project>("add_label_class", { projectId, name, color }),

  updateLabelClass: (projectId: string, classId: string, name: string, color: string) =>
    invoke<Project>("update_label_class", { projectId, classId, name, color }),

  deleteLabelClass: (projectId: string, classId: string) =>
    invoke<Project>("delete_label_class", { projectId, classId }),

  importImages: (projectId: string, filePaths: string[]) =>
    invoke<Project>("import_images", { projectId, filePaths }),

  importImageFolder: (projectId: string, folderPath: string) =>
    invoke<Project>("import_image_folder", { projectId, folderPath }),

  saveImageAnnotations: (
    projectId: string,
    itemId: string,
    split: SplitType,
    boxes: BoundingBox[],
    polygons: PolygonShape[],
    imageLabels: string[]
  ) =>
    invoke<Project>("save_image_annotations", {
      projectId,
      itemId,
      split,
      boxes,
      polygons,
      imageLabels,
    }),

  deleteImageItem: (projectId: string, itemId: string) =>
    invoke<Project>("delete_image_item", { projectId, itemId }),

  importAudioFiles: (projectId: string, filePaths: string[]) =>
    invoke<Project>("import_audio_files", { projectId, filePaths }),

  updateAudioMetadata: (
    projectId: string,
    itemId: string,
    durationSec: number,
    sampleRate: number,
    channels: number
  ) =>
    invoke<void>("update_audio_metadata", {
      projectId,
      itemId,
      durationSec,
      sampleRate,
      channels,
    }),

  saveAudioSlices: (
    projectId: string,
    itemId: string,
    split: SplitType,
    slices: AudioSlice[]
  ) => invoke<Project>("save_audio_slices", { projectId, itemId, split, slices }),

  deleteAudioItem: (projectId: string, itemId: string) =>
    invoke<Project>("delete_audio_item", { projectId, itemId }),

  upsertTextItem: (projectId: string, item: TextItem) =>
    invoke<Project>("upsert_text_item", { projectId, item }),

  deleteTextItem: (projectId: string, itemId: string) =>
    invoke<Project>("delete_text_item", { projectId, itemId }),

  previewCsvHeaders: (filePath: string) =>
    invoke<string[]>("preview_csv_headers", { filePath }),

  importTextJsonl: (projectId: string, filePath: string) =>
    invoke<Project>("import_text_jsonl", { projectId, filePath }),

  importTextCsv: (
    projectId: string,
    filePath: string,
    promptCol: string,
    completionCol: string | null,
    systemCol: string | null,
    labelCol: string | null
  ) =>
    invoke<Project>("import_text_csv", {
      projectId,
      filePath,
      promptCol,
      completionCol,
      systemCol,
      labelCol,
    }),

  exportTextDataset: (projectId: string, format: string, destPath: string) =>
    invoke<number>("export_text_dataset", { projectId, format, destPath }),

  updateProjectSettings: (
    projectId: string,
    imageAug: ImageAugConfig,
    audioAug: AudioAugConfig,
    textAug: TextAugConfig,
    splitRatio: SplitRatio,
    applySplitsNow: boolean
  ) =>
    invoke<Project>("update_project_settings", {
      projectId,
      imageAug,
      audioAug,
      textAug,
      splitRatio,
      applySplitsNow,
    }),

  exportDatasetZip: (projectId: string, destPath: string) =>
    invoke<DatasetExportResult>("export_dataset_zip", { projectId, destPath }),

  listAutolabelModels: () => invoke<AutoLabelModelInfo[]>("list_autolabel_models"),

  downloadAutolabelModel: (modelId: string) =>
    invoke<AutoLabelModelInfo>("download_autolabel_model", { modelId }),

  importCustomAutolabelOnnx: (filePath: string) =>
    invoke<AutoLabelModelInfo>("import_custom_autolabel_onnx", { filePath }),

  readBinaryFile: async (path: string): Promise<Uint8Array> => {
    const buf = await invoke<ArrayBuffer>("read_binary_file", { path });
    return new Uint8Array(buf);
  },

  registerCustomModel: (
    name: string,
    module: ProjectModule,
    task: ProjectTask,
    sourceKind: string,
    localWeightPath: string | null,
    hfRepoId: string | null
  ) =>
    invoke<CustomModel>("register_custom_model", {
      name,
      module,
      task,
      sourceKind,
      localWeightPath,
      hfRepoId,
    }),

  deleteCustomModel: (modelId: string) =>
    invoke<CustomModel[]>("delete_custom_model", { modelId }),

  startTrainingRun: (config: TrainingConfig, offlineOnly: boolean) =>
    invoke<TrainingRun>("start_training_run", { config, offlineOnly }),

  listTrainingRuns: () => invoke<TrainingRun[]>("list_training_runs"),

  syncTrainingRun: (runId: string) =>
    invoke<TrainingRun>("sync_training_run", { runId }),

  cancelTrainingPoll: (runId: string) =>
    invoke<TrainingRun>("cancel_training_poll", { runId }),

  renameTrainingRun: (runId: string, newName: string) =>
    invoke<TrainingRun>("rename_training_run", { runId, newName }),

  deleteTrainingRun: (runId: string) =>
    invoke<void>("delete_training_run", { runId }),

  listColabRuntimeSpecs: () => invoke<ColabRuntimeSpec[]>("list_colab_runtime_specs"),

  openLocalPath: (path: string) => invoke<void>("open_local_path", { path }),

  openExternalUrl: (url: string) => invoke<void>("open_external_url", { url }),

  resolveMediaAbsPath: (relPath: string) =>
    invoke<string>("resolve_media_abs_path", { relPath }),

  listEvalBundles: () => invoke<EvalBundleInfo[]>("list_eval_bundles"),

  inspectLocalEvalBundle: (path: string, module: ProjectModule, task: ProjectTask) =>
    invoke<EvalBundleInfo>("inspect_local_eval_bundle", { path, module, task }),

  listFolderMedia: (folderPath: string, kind: "image" | "audio") =>
    invoke<string[]>("list_folder_media", { folderPath, kind }),

  readTextEvalFile: (filePath: string) =>
    invoke<TextEvalRow[]>("read_text_eval_file", { filePath }),

  checkConnectivity: () => invoke<ConnectivityStatus>("check_connectivity"),

  runPreflightCheck: (projectId: string) =>
    invoke<PreflightReport>("run_preflight_check", { projectId }),

  fetchUncensoredHub: (
    moduleFilter: string,
    formatFilter: string,
    sortBy: string,
    searchQuery: string,
    hideOver4gb: boolean
  ) =>
    invoke<HfHubResponse>("fetch_uncensored_hub", {
      moduleFilter,
      formatFilter,
      sortBy,
      searchQuery,
      hideOver4gb,
    }),

  inspectHfRepo: (repoInput: string, forceRefresh?: boolean) =>
    invoke<HfHubModel>("inspect_hf_repo", {
      repoInput,
      forceRefresh: forceRefresh ?? false,
    }),

  downloadHfFile: (repoInput: string, rfilename: string) =>
    invoke<string>("download_hf_file", { repoInput, rfilename }),

  listLocalGgufModels: () => invoke<LocalGgufModel[]>("list_local_gguf_models"),

  getLlamaRuntimeStatus: () => invoke<LlamaRuntimeStatus>("get_llama_runtime_status"),

  installLlamaRuntime: () => invoke<LlamaRuntimeStatus>("install_llama_runtime"),

  runGgufChat: (
    modelPath: string,
    mmprojPath: string | null,
    system: string,
    prompt: string,
    maxTokens: number,
    temperature: number
  ) =>
    invoke<GgufGenerationResult>("run_gguf_chat", {
      modelPath,
      mmprojPath,
      system,
      prompt,
      maxTokens,
      temperature,
    }),

  unloadGgufRuntime: () => invoke<void>("unload_gguf_runtime"),

  exportRunNotebook: (runId: string, destPath: string) =>
    invoke<string>("export_run_notebook", { runId, destPath }),

  exportModelBundle: (
    onnxPath: string,
    labelsPath: string | null,
    metricsPath: string | null,
    destDir: string,
    runtimeKind: string,
    asZip: boolean
  ) =>
    invoke<ExportBundleResult>("export_model_bundle", {
      onnxPath,
      labelsPath,
      metricsPath,
      destDir,
      runtimeKind,
      asZip,
    }),

  fileSrc: (absPath: string): string => convertFileSrc(absPath),

  mediaSrc: (dataDir: string, relPath: string): string => {
    const cleanBase = dataDir.replace(/[\\/]+$/, "");
    const cleanRel = relPath.replace(/^[\\/]+/, "").replace(/\//g, "\\");
    return convertFileSrc(`${cleanBase}\\${cleanRel}`);
  },

  pickFiles: (filters: { name: string; extensions: string[] }[], multiple = true) =>
    open({ multiple, directory: false, filters }),

  pickFolder: () => open({ multiple: false, directory: true }),

  saveFile: (defaultPath: string, filters: { name: string; extensions: string[] }[]) =>
    save({ defaultPath, filters }),

  onTrainingUpdate: (cb: (run: TrainingRun) => void): Promise<UnlistenFn> =>
    listen<TrainingRun>("training-update", (e) => cb(e.payload)),

  onAutoLabelProgress: (cb: (p: AutoLabelProgressPayload) => void): Promise<UnlistenFn> =>
    listen<AutoLabelProgressPayload>("autolabel-download-progress", (e) => cb(e.payload)),

  onHfDownloadProgress: (cb: (p: HfDownloadProgressPayload) => void): Promise<UnlistenFn> =>
    listen<HfDownloadProgressPayload>("hf-download-progress", (e) => cb(e.payload)),
};
