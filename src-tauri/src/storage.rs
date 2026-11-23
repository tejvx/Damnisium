use crate::constants::APP_VERSION;
use crate::media::{create_thumbnail, inspect_audio_header, inspect_image_dimensions};
use crate::models::{
    AppConfig, AudioItem, CustomModel, DatasetExportResult, GoogleAuthStored, HfHubModel,
    ImageItem, LabelClass, LocalGgufModel, PreflightIssue, PreflightReport, Project, ProjectModule,
    ProjectSummary, ProjectTask, SplitCounts, SplitType, TextItem, TrainingRun,
};
use chrono::Utc;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use uuid::Uuid;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

const DEFAULT_CLASS_COLORS: &[&str] = &[
    "#38bdf8", "#a855f7", "#22c55e", "#f59e0b", "#ef4444", "#ec4899", "#14b8a6", "#6366f1",
    "#eab308", "#06b6d4",
];

#[derive(Clone)]
pub struct StorageManager {
    exe_dir: PathBuf,
    data_dir: PathBuf,
}

impl StorageManager {
    pub fn init() -> Result<Self, String> {
        let exe_dir = Self::resolve_base_dir()?;
        let pointer_file = exe_dir.join("damnisium-path.txt");
        let data_dir = if pointer_file.exists() {
            let raw = fs::read_to_string(&pointer_file).unwrap_or_default();
            let trimmed = raw.trim();
            if !trimmed.is_empty() {
                PathBuf::from(trimmed)
            } else {
                exe_dir.join("damnisium-data")
            }
        } else {
            exe_dir.join("damnisium-data")
        };

        let mgr = Self { exe_dir, data_dir };
        mgr.ensure_layout()?;
        Ok(mgr)
    }

    #[cfg(test)]
    pub fn from_data_dir(data_dir: PathBuf) -> Result<Self, String> {
        let exe_dir = data_dir.parent().unwrap_or(&data_dir).to_path_buf();
        let mgr = Self { exe_dir, data_dir };
        mgr.ensure_layout()?;
        Ok(mgr)
    }

    fn resolve_base_dir() -> Result<PathBuf, String> {
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        let parent = exe
            .parent()
            .ok_or_else(|| "Unable to determine executable directory.".to_string())?
            .to_path_buf();

        let is_cargo_target = parent
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n == "debug" || n == "release")
            .unwrap_or(false)
            && parent
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                .map(|n| n == "target")
                .unwrap_or(false);

        if is_cargo_target {
            if let Some(repo_root) = parent
                .parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
            {
                return Ok(repo_root.to_path_buf());
            }
        }
        Ok(parent)
    }

    pub fn ensure_layout(&self) -> Result<(), String> {
        for rel in [
            "",
            "projects",
            "runs",
            "models",
            "models/autolabel",
            "models/custom",
            "models/gguf",
            "runtime",
            "runtime/llama",
            "exports",
        ] {
            let p = if rel.is_empty() {
                self.data_dir.clone()
            } else {
                self.data_dir.join(rel)
            };
            fs::create_dir_all(&p).map_err(|e| format!("Failed creating {}: {e}", p.display()))?;
        }
        Ok(())
    }

    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    pub fn set_data_dir(&mut self, new_dir: PathBuf) -> Result<(), String> {
        fs::create_dir_all(&new_dir).map_err(|e| e.to_string())?;
        let pointer_file = self.exe_dir.join("damnisium-path.txt");
        fs::write(&pointer_file, new_dir.to_string_lossy().as_bytes())
            .map_err(|e| e.to_string())?;
        self.data_dir = new_dir;
        self.ensure_layout()?;
        Ok(())
    }

    fn write_json_atomic<T: serde::Serialize>(&self, path: &Path, val: &T) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let tmp = path.with_extension(format!(
            "json.{}.tmp",
            &Uuid::new_v4().to_string().replace('-', "")[..8]
        ));
        let bytes = serde_json::to_vec_pretty(val).map_err(|e| e.to_string())?;
        fs::write(&tmp, bytes).map_err(|e| e.to_string())?;
        if path.exists() {
            let _ = fs::remove_file(path);
        }
        if let Err(e) = fs::rename(&tmp, path) {
            let _ = fs::remove_file(&tmp);
            return Err(e.to_string());
        }
        Ok(())
    }

    fn read_json_or_default<T: serde::de::DeserializeOwned + Default>(&self, path: &Path) -> T {
        if !path.exists() {
            return T::default();
        }
        fs::read(path)
            .ok()
            .and_then(|b| {
                let clean = b.strip_prefix(b"\xef\xbb\xbf").unwrap_or(&b);
                serde_json::from_slice(clean).ok()
            })
            .unwrap_or_default()
    }

    pub fn load_config(&self) -> AppConfig {
        let mut cfg: AppConfig = self.read_json_or_default(&self.data_dir.join("config.json"));
        cfg.data_dir = self.data_dir.to_string_lossy().to_string();
        cfg
    }

    pub fn save_config(&self, mut cfg: AppConfig) -> Result<AppConfig, String> {
        cfg.data_dir = self.data_dir.to_string_lossy().to_string();
        self.write_json_atomic(&self.data_dir.join("config.json"), &cfg)?;
        Ok(cfg)
    }

    pub fn load_auth(&self) -> Result<GoogleAuthStored, String> {
        let path = self.data_dir.join("auth.json");
        if !path.exists() {
            return Ok(GoogleAuthStored::default());
        }
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        let clean = bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(&bytes);
        serde_json::from_slice(clean).map_err(|e| e.to_string())
    }

    pub fn save_auth(&self, auth: &GoogleAuthStored) -> Result<(), String> {
        self.write_json_atomic(&self.data_dir.join("auth.json"), auth)
    }

    pub fn clear_auth(&self) -> Result<(), String> {
        let path = self.data_dir.join("auth.json");
        if path.exists() {
            fs::remove_file(path).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn project_dir(&self, project_id: &str) -> PathBuf {
        self.data_dir.join("projects").join(project_id)
    }

    pub fn create_project(
        &self,
        name: String,
        module: ProjectModule,
        task: ProjectTask,
    ) -> Result<Project, String> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err("Project name cannot be empty.".into());
        }
        Self::validate_module_task(&module, &task)?;
        let now = Utc::now().to_rfc3339();
        let id = format!("prj_{}", &Uuid::new_v4().to_string().replace('-', "")[..12]);
        let project = Project {
            id: id.clone(),
            name: trimmed.to_string(),
            module,
            task,
            app_version: APP_VERSION.to_string(),
            created_at: now.clone(),
            updated_at: now,
            classes: vec![],
            image_items: vec![],
            audio_items: vec![],
            text_items: vec![],
            image_aug: Default::default(),
            audio_aug: Default::default(),
            text_aug: Default::default(),
            split_ratio: Default::default(),
        };
        let pdir = self.project_dir(&id);
        fs::create_dir_all(pdir.join("media")).map_err(|e| e.to_string())?;
        fs::create_dir_all(pdir.join("thumbs")).map_err(|e| e.to_string())?;
        self.save_project(&project)?;
        Ok(project)
    }

    pub fn validate_module_task(module: &ProjectModule, task: &ProjectTask) -> Result<(), String> {
        let valid = match module {
            ProjectModule::Image => matches!(
                task,
                ProjectTask::Detection | ProjectTask::Segmentation | ProjectTask::Classification
            ),
            ProjectModule::Audio => matches!(
                task,
                ProjectTask::Classification | ProjectTask::Transcription
            ),
            ProjectModule::Text => matches!(
                task,
                ProjectTask::Qa | ProjectTask::Completion | ProjectTask::Intent
            ),
        };
        if !valid {
            return Err("Selected task is not supported for this module.".into());
        }
        Ok(())
    }

    pub fn load_project(&self, project_id: &str) -> Result<Project, String> {
        let path = self.project_dir(project_id).join("project.json");
        if !path.exists() {
            return Err(format!("Project {project_id} not found."));
        }
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        serde_json::from_slice(&bytes).map_err(|e| e.to_string())
    }

    pub fn save_project(&self, project: &Project) -> Result<(), String> {
        let mut updated = project.clone();
        updated.updated_at = Utc::now().to_rfc3339();
        let path = self.project_dir(&updated.id).join("project.json");
        self.write_json_atomic(&path, &updated)
    }

    pub fn delete_project(&self, project_id: &str) -> Result<(), String> {
        let dir = self.project_dir(project_id);
        if dir.exists() {
            fs::remove_dir_all(dir).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn list_projects(&self) -> Result<Vec<ProjectSummary>, String> {
        let root = self.data_dir.join("projects");
        if !root.exists() {
            return Ok(vec![]);
        }
        let mut summaries = Vec::new();
        for entry in fs::read_dir(root).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let pfile = entry.path().join("project.json");
            if pfile.exists() {
                if let Ok(bytes) = fs::read(&pfile) {
                    if let Ok(proj) = serde_json::from_slice::<Project>(&bytes) {
                        let (item_count, labeled_count) = match proj.module {
                            ProjectModule::Image => {
                                let labeled = proj
                                    .image_items
                                    .iter()
                                    .filter(|i| {
                                        !i.boxes.is_empty()
                                            || !i.polygons.is_empty()
                                            || !i.image_labels.is_empty()
                                    })
                                    .count();
                                (proj.image_items.len(), labeled)
                            }
                            ProjectModule::Audio => {
                                let labeled = proj
                                    .audio_items
                                    .iter()
                                    .filter(|a| !a.slices.is_empty())
                                    .count();
                                (proj.audio_items.len(), labeled)
                            }
                            ProjectModule::Text => {
                                let labeled = proj
                                    .text_items
                                    .iter()
                                    .filter(|t| {
                                        !t.prompt.trim().is_empty()
                                            && (!t.completion.trim().is_empty()
                                                || t.intent_class_id.is_some())
                                    })
                                    .count();
                                (proj.text_items.len(), labeled)
                            }
                        };
                        summaries.push(ProjectSummary {
                            id: proj.id,
                            name: proj.name,
                            module: proj.module,
                            task: proj.task,
                            item_count,
                            labeled_count,
                            class_count: proj.classes.len(),
                            updated_at: proj.updated_at,
                        });
                    }
                }
            }
        }
        summaries.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(summaries)
    }

    pub fn add_class(
        &self,
        project_id: &str,
        name: String,
        color: Option<String>,
    ) -> Result<Project, String> {
        let mut project = self.load_project(project_id)?;
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err("Class name cannot be empty.".into());
        }
        if project
            .classes
            .iter()
            .any(|c| c.name.eq_ignore_ascii_case(trimmed))
        {
            return Err(format!("Class '{trimmed}' already exists."));
        }
        let chosen_color = color.unwrap_or_else(|| {
            DEFAULT_CLASS_COLORS[project.classes.len() % DEFAULT_CLASS_COLORS.len()].to_string()
        });
        project.classes.push(LabelClass {
            id: format!("cls_{}", &Uuid::new_v4().to_string().replace('-', "")[..10]),
            name: trimmed.to_string(),
            color: chosen_color,
        });
        self.save_project(&project)?;
        self.load_project(project_id)
    }

    pub fn ensure_class_by_name(&self, project: &mut Project, name: &str) -> String {
        let trimmed = name.trim();
        if let Some(existing) = project
            .classes
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case(trimmed))
        {
            return existing.id.clone();
        }
        let id = format!("cls_{}", &Uuid::new_v4().to_string().replace('-', "")[..10]);
        let color =
            DEFAULT_CLASS_COLORS[project.classes.len() % DEFAULT_CLASS_COLORS.len()].to_string();
        project.classes.push(LabelClass {
            id: id.clone(),
            name: trimmed.to_string(),
            color,
        });
        id
    }

    pub fn import_images(
        &self,
        project_id: &str,
        file_paths: Vec<String>,
    ) -> Result<Project, String> {
        let mut project = self.load_project(project_id)?;
        if project.module != ProjectModule::Image {
            return Err("Images can only be imported into Image ML projects.".into());
        }
        let pdir = self.project_dir(project_id);
        let media_dir = pdir.join("media");
        let thumbs_dir = pdir.join("thumbs");
        fs::create_dir_all(&media_dir).map_err(|e| e.to_string())?;
        fs::create_dir_all(&thumbs_dir).map_err(|e| e.to_string())?;

        for src_str in file_paths {
            let src = PathBuf::from(&src_str);
            if !src.is_file() {
                continue;
            }
            let ext = src
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("png")
                .to_lowercase();
            if !matches!(
                ext.as_str(),
                "png" | "jpg" | "jpeg" | "webp" | "bmp" | "gif"
            ) {
                continue;
            }
            let (width, height) = match inspect_image_dimensions(&src) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let item_id = format!("img_{}", &Uuid::new_v4().to_string().replace('-', "")[..12]);
            let orig_name = src
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("image.png")
                .to_string();
            let stored_name = format!("{item_id}.{ext}");
            let rel_path = format!("projects/{project_id}/media/{stored_name}");
            let dest = media_dir.join(&stored_name);
            fs::copy(&src, &dest).map_err(|e| e.to_string())?;

            let thumb_name = format!("{item_id}.jpg");
            let thumb_rel = format!("projects/{project_id}/thumbs/{thumb_name}");
            let thumb_dest = thumbs_dir.join(&thumb_name);
            let _ = create_thumbnail(&dest, &thumb_dest);

            let split = Self::pick_split_by_index(project.image_items.len(), &project);
            project.image_items.push(ImageItem {
                id: item_id,
                filename: orig_name,
                rel_path,
                thumb_rel_path: if thumb_dest.exists() {
                    thumb_rel
                } else {
                    format!("projects/{project_id}/media/{stored_name}")
                },
                width,
                height,
                split,
                boxes: vec![],
                polygons: vec![],
                image_labels: vec![],
                created_at: Utc::now().to_rfc3339(),
            });
        }
        self.save_project(&project)?;
        self.load_project(project_id)
    }

    pub fn import_audio_files(
        &self,
        project_id: &str,
        file_paths: Vec<String>,
    ) -> Result<Project, String> {
        let mut project = self.load_project(project_id)?;
        if project.module != ProjectModule::Audio {
            return Err("Audio files can only be imported into Audio projects.".into());
        }
        let media_dir = self.project_dir(project_id).join("media");
        fs::create_dir_all(&media_dir).map_err(|e| e.to_string())?;

        for src_str in file_paths {
            let src = PathBuf::from(&src_str);
            if !src.is_file() {
                continue;
            }
            let ext = src
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("wav")
                .to_lowercase();
            if !matches!(ext.as_str(), "wav" | "mp3" | "ogg" | "flac" | "m4a" | "aac") {
                continue;
            }
            let item_id = format!("aud_{}", &Uuid::new_v4().to_string().replace('-', "")[..12]);
            let orig_name = src
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("audio.wav")
                .to_string();
            let stored_name = format!("{item_id}.{ext}");
            let rel_path = format!("projects/{project_id}/media/{stored_name}");
            let dest = media_dir.join(&stored_name);
            fs::copy(&src, &dest).map_err(|e| e.to_string())?;
            let (duration_sec, sample_rate, channels) = inspect_audio_header(&dest);
            let split = Self::pick_split_by_index(project.audio_items.len(), &project);
            project.audio_items.push(AudioItem {
                id: item_id,
                filename: orig_name,
                rel_path,
                duration_sec,
                sample_rate,
                channels,
                split,
                slices: vec![],
                created_at: Utc::now().to_rfc3339(),
            });
        }
        self.save_project(&project)?;
        self.load_project(project_id)
    }

    fn pick_split_by_index(idx: usize, project: &Project) -> SplitType {
        let total = (project.split_ratio.train as usize)
            + (project.split_ratio.val as usize)
            + (project.split_ratio.test as usize);
        if total == 0 {
            return SplitType::Train;
        }
        let slot = (idx * 37) % 100;
        let train_cut = ((project.split_ratio.train as usize) * 100) / total;
        let val_cut = (((project.split_ratio.train as usize) + (project.split_ratio.val as usize))
            * 100)
            / total;
        if slot < train_cut {
            SplitType::Train
        } else if slot < val_cut {
            SplitType::Val
        } else {
            SplitType::Test
        }
    }

    pub fn import_text_jsonl(&self, project_id: &str, file_path: &str) -> Result<Project, String> {
        let mut project = self.load_project(project_id)?;
        if project.module != ProjectModule::Text {
            return Err("Text data can only be imported into Text projects.".into());
        }
        let raw = fs::read_to_string(file_path).map_err(|e| e.to_string())?;
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
                .or_else(|| val.get("input"))
                .or_else(|| val.get("text"))
                .or_else(|| val.get("instruction"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let completion = val
                .get("completion")
                .or_else(|| val.get("answer"))
                .or_else(|| val.get("output"))
                .or_else(|| val.get("response"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let system = val
                .get("system")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let label_name = val
                .get("label")
                .or_else(|| val.get("intent"))
                .or_else(|| val.get("category"))
                .and_then(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());

            let intent_class_id = label_name.map(|n| self.ensure_class_by_name(&mut project, &n));
            let split = match val.get("split").and_then(|v| v.as_str()) {
                Some("val") => SplitType::Val,
                Some("test") => SplitType::Test,
                Some("train") => SplitType::Train,
                _ => Self::pick_split_by_index(project.text_items.len(), &project),
            };
            project.text_items.push(TextItem {
                id: format!("txt_{}", &Uuid::new_v4().to_string().replace('-', "")[..12]),
                prompt,
                completion,
                system,
                intent_class_id,
                split,
                created_at: Utc::now().to_rfc3339(),
            });
        }
        self.save_project(&project)?;
        self.load_project(project_id)
    }

    pub fn import_text_csv(
        &self,
        project_id: &str,
        file_path: &str,
        prompt_col: &str,
        completion_col: Option<&str>,
        system_col: Option<&str>,
        label_col: Option<&str>,
    ) -> Result<Project, String> {
        let mut project = self.load_project(project_id)?;
        if project.module != ProjectModule::Text {
            return Err("CSV text data can only be imported into Text projects.".into());
        }
        let mut reader = csv::ReaderBuilder::new()
            .flexible(true)
            .from_path(file_path)
            .map_err(|e| format!("Failed opening CSV: {e}"))?;
        let headers = reader
            .headers()
            .map_err(|e| format!("Failed reading CSV headers: {e}"))?
            .clone();

        let idx_of = |name: &str| -> Option<usize> {
            headers
                .iter()
                .position(|h| h.trim().eq_ignore_ascii_case(name.trim()))
        };
        let prompt_idx = idx_of(prompt_col)
            .ok_or_else(|| format!("Column '{prompt_col}' not found in CSV headers."))?;
        let completion_idx = completion_col.and_then(idx_of);
        let system_idx = system_col.and_then(idx_of);
        let label_idx = label_col.and_then(idx_of);

        for rec in reader.records() {
            let record = rec.map_err(|e| format!("CSV row error: {e}"))?;
            let prompt = record.get(prompt_idx).unwrap_or("").trim().to_string();
            if prompt.is_empty() {
                continue;
            }
            let completion = completion_idx
                .and_then(|i| record.get(i))
                .unwrap_or("")
                .trim()
                .to_string();
            let system = system_idx
                .and_then(|i| record.get(i))
                .unwrap_or("")
                .trim()
                .to_string();
            let intent_class_id = label_idx
                .and_then(|i| record.get(i))
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|name| self.ensure_class_by_name(&mut project, name));

            let split = Self::pick_split_by_index(project.text_items.len(), &project);
            project.text_items.push(TextItem {
                id: format!("txt_{}", &Uuid::new_v4().to_string().replace('-', "")[..12]),
                prompt,
                completion,
                system,
                intent_class_id,
                split,
                created_at: Utc::now().to_rfc3339(),
            });
        }
        self.save_project(&project)?;
        self.load_project(project_id)
    }

    pub fn export_text_dataset(
        &self,
        project_id: &str,
        format: &str,
        dest_path: &str,
    ) -> Result<usize, String> {
        let project = self.load_project(project_id)?;
        let class_map: HashMap<&str, &str> = project
            .classes
            .iter()
            .map(|c| (c.id.as_str(), c.name.as_str()))
            .collect();

        if format.eq_ignore_ascii_case("csv") {
            let mut wtr = csv::Writer::from_path(dest_path).map_err(|e| e.to_string())?;
            wtr.write_record(["prompt", "completion", "system", "label", "split"])
                .map_err(|e| e.to_string())?;
            for item in &project.text_items {
                let label = item
                    .intent_class_id
                    .as_deref()
                    .and_then(|id| class_map.get(id).copied())
                    .unwrap_or("");
                let split = Self::split_str(&item.split);
                wtr.write_record([
                    item.prompt.as_str(),
                    item.completion.as_str(),
                    item.system.as_str(),
                    label,
                    split,
                ])
                .map_err(|e| e.to_string())?;
            }
            wtr.flush().map_err(|e| e.to_string())?;
            return Ok(project.text_items.len());
        }

        let mut out = File::create(dest_path).map_err(|e| e.to_string())?;
        for item in &project.text_items {
            let label = item
                .intent_class_id
                .as_deref()
                .and_then(|id| class_map.get(id).copied());
            let row = serde_json::json!({
                "prompt": item.prompt,
                "completion": item.completion,
                "system": item.system,
                "label": label,
                "split": Self::split_str(&item.split)
            });
            writeln!(
                out,
                "{}",
                serde_json::to_string(&row).map_err(|e| e.to_string())?
            )
            .map_err(|e| e.to_string())?;
        }
        Ok(project.text_items.len())
    }

    pub fn split_str(split: &SplitType) -> &'static str {
        match split {
            SplitType::Train => "train",
            SplitType::Val => "val",
            SplitType::Test => "test",
        }
    }

    pub fn apply_split_ratios(&self, project: &mut Project) {
        let mut indices: Vec<usize> = match project.module {
            ProjectModule::Image => (0..project.image_items.len()).collect(),
            ProjectModule::Audio => (0..project.audio_items.len()).collect(),
            ProjectModule::Text => (0..project.text_items.len()).collect(),
        };
        let n = indices.len();
        if n == 0 {
            return;
        }
        let total = (project.split_ratio.train as usize)
            + (project.split_ratio.val as usize)
            + (project.split_ratio.test as usize);
        let mut train_n = if total == 0 {
            n
        } else {
            ((n * project.split_ratio.train as usize) as f64 / total as f64).round() as usize
        }
        .max(1)
        .min(n);
        let mut val_n = if total == 0 {
            0
        } else {
            ((n * project.split_ratio.val as usize) as f64 / total as f64).round() as usize
        }
        .min(n.saturating_sub(train_n));
        if n >= 3 && project.split_ratio.val > 0 && val_n == 0 {
            val_n = 1;
            train_n = train_n.min(n - 1).max(1);
        }

        for (pos, idx) in indices.drain(..).enumerate() {
            let s = if pos < train_n {
                SplitType::Train
            } else if pos < train_n + val_n {
                SplitType::Val
            } else {
                SplitType::Test
            };
            match project.module {
                ProjectModule::Image => project.image_items[idx].split = s,
                ProjectModule::Audio => project.audio_items[idx].split = s,
                ProjectModule::Text => project.text_items[idx].split = s,
            }
        }
    }

    pub fn package_dataset_zip(
        &self,
        project: &Project,
        dest_zip: &Path,
    ) -> Result<DatasetExportResult, String> {
        if let Some(parent) = dest_zip.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let file = File::create(dest_zip).map_err(|e| e.to_string())?;
        let mut zip = ZipWriter::new(file);
        let opts = SimpleFileOptions::default()
            .compression_method(CompressionMethod::Deflated)
            .compression_level(Some(6));

        let class_index: HashMap<&str, usize> = project
            .classes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.id.as_str(), i))
            .collect();
        let class_names: HashMap<&str, &str> = project
            .classes
            .iter()
            .map(|c| (c.id.as_str(), c.name.as_str()))
            .collect();

        let item_count = match project.module {
            ProjectModule::Image => {
                let mut count = 0usize;
                let mut has_val = false;
                let mut has_test = false;
                for item in &project.image_items {
                    if item.split == SplitType::Val {
                        has_val = true;
                    } else if item.split == SplitType::Test {
                        has_test = true;
                    }
                    let split = Self::split_str(&item.split);
                    let src_path = self.data_dir.join(&item.rel_path);
                    if !src_path.is_file() {
                        continue;
                    }
                    let ext = src_path
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("png");

                    if project.task == ProjectTask::Classification {
                        let label_name = item
                            .image_labels
                            .first()
                            .and_then(|cid| class_names.get(cid.as_str()).copied())
                            .map(Self::sanitize_slug)
                            .unwrap_or_else(|| "unlabeled".to_string());
                        if label_name == "unlabeled" {
                            continue;
                        }
                        let archive_path = format!("{split}/{label_name}/{}.{}", item.id, ext);
                        let bytes = fs::read(&src_path).map_err(|e| e.to_string())?;
                        zip.start_file(&archive_path, opts)
                            .map_err(|e| e.to_string())?;
                        zip.write_all(&bytes).map_err(|e| e.to_string())?;
                        count += 1;
                    } else {
                        let img_archive = format!("images/{split}/{}.{}", item.id, ext);
                        let bytes = fs::read(&src_path).map_err(|e| e.to_string())?;
                        zip.start_file(&img_archive, opts)
                            .map_err(|e| e.to_string())?;
                        zip.write_all(&bytes).map_err(|e| e.to_string())?;

                        let mut label_lines = Vec::new();
                        if project.task == ProjectTask::Detection {
                            for b in &item.boxes {
                                if let Some(&cls_idx) = class_index.get(b.class_id.as_str()) {
                                    let cx = (b.x + b.w * 0.5).clamp(0.0, 1.0);
                                    let cy = (b.y + b.h * 0.5).clamp(0.0, 1.0);
                                    let w = b.w.clamp(0.001, 1.0);
                                    let h = b.h.clamp(0.001, 1.0);
                                    label_lines
                                        .push(format!("{cls_idx} {cx:.6} {cy:.6} {w:.6} {h:.6}"));
                                }
                            }
                        } else if project.task == ProjectTask::Segmentation {
                            for poly in &item.polygons {
                                if poly.points.len() < 3 {
                                    continue;
                                }
                                if let Some(&cls_idx) = class_index.get(poly.class_id.as_str()) {
                                    let mut parts = vec![cls_idx.to_string()];
                                    for pt in &poly.points {
                                        parts.push(format!(
                                            "{:.6} {:.6}",
                                            pt.x.clamp(0.0, 1.0),
                                            pt.y.clamp(0.0, 1.0)
                                        ));
                                    }
                                    label_lines.push(parts.join(" "));
                                }
                            }
                        }
                        let lbl_archive = format!("labels/{split}/{}.txt", item.id);
                        zip.start_file(&lbl_archive, opts)
                            .map_err(|e| e.to_string())?;
                        zip.write_all(label_lines.join("\n").as_bytes())
                            .map_err(|e| e.to_string())?;
                        count += 1;
                    }
                }

                if project.task != ProjectTask::Classification {
                    let val_path = if has_val {
                        "images/val"
                    } else {
                        "images/train"
                    };
                    let mut yaml = String::from("path: .\ntrain: images/train\n");
                    yaml.push_str(&format!("val: {val_path}\n"));
                    if has_test {
                        yaml.push_str("test: images/test\n");
                    }
                    yaml.push_str(&format!("nc: {}\nnames:\n", project.classes.len()));
                    for (idx, cls) in project.classes.iter().enumerate() {
                        let safe = cls.name.replace('"', "\\\"");
                        yaml.push_str(&format!("  {idx}: \"{safe}\"\n"));
                    }
                    zip.start_file("data.yaml", opts)
                        .map_err(|e| e.to_string())?;
                    zip.write_all(yaml.as_bytes()).map_err(|e| e.to_string())?;
                }
                let classes_json = serde_json::to_vec_pretty(
                    &project
                        .classes
                        .iter()
                        .map(|c| c.name.clone())
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| e.to_string())?;
                zip.start_file("classes.json", opts)
                    .map_err(|e| e.to_string())?;
                zip.write_all(&classes_json).map_err(|e| e.to_string())?;
                count
            }
            ProjectModule::Audio => {
                let mut manifest = Vec::new();
                let mut seen_files: HashSet<String> = HashSet::new();
                for item in &project.audio_items {
                    let src_path = self.data_dir.join(&item.rel_path);
                    if !src_path.is_file() {
                        continue;
                    }
                    let ext = src_path
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("wav");
                    let rel_audio = format!("audio/{}.{}", item.id, ext);
                    if seen_files.insert(rel_audio.clone()) {
                        let bytes = fs::read(&src_path).map_err(|e| e.to_string())?;
                        zip.start_file(&rel_audio, opts)
                            .map_err(|e| e.to_string())?;
                        zip.write_all(&bytes).map_err(|e| e.to_string())?;
                    }
                    let split = Self::split_str(&item.split);
                    for s in &item.slices {
                        let label = s
                            .class_id
                            .as_deref()
                            .and_then(|id| class_names.get(id).copied())
                            .unwrap_or("");
                        manifest.push(serde_json::json!({
                            "audio_path": rel_audio,
                            "start_sec": s.start_sec,
                            "end_sec": s.end_sec,
                            "label": label,
                            "transcript": s.transcript,
                            "split": split
                        }));
                    }
                }
                let manifest_bytes =
                    serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
                zip.start_file("manifest.json", opts)
                    .map_err(|e| e.to_string())?;
                zip.write_all(&manifest_bytes).map_err(|e| e.to_string())?;

                let classes_json = serde_json::to_vec_pretty(
                    &project
                        .classes
                        .iter()
                        .map(|c| c.name.clone())
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| e.to_string())?;
                zip.start_file("classes.json", opts)
                    .map_err(|e| e.to_string())?;
                zip.write_all(&classes_json).map_err(|e| e.to_string())?;
                manifest.len()
            }
            ProjectModule::Text => {
                let mut train_rows = Vec::new();
                let mut val_rows = Vec::new();
                let mut test_rows = Vec::new();
                for item in &project.text_items {
                    let label = item
                        .intent_class_id
                        .as_deref()
                        .and_then(|id| class_names.get(id).copied());
                    let line = serde_json::to_string(&serde_json::json!({
                        "prompt": item.prompt,
                        "completion": item.completion,
                        "system": item.system,
                        "label": label
                    }))
                    .map_err(|e| e.to_string())?;
                    match item.split {
                        SplitType::Train => train_rows.push(line),
                        SplitType::Val => val_rows.push(line),
                        SplitType::Test => test_rows.push(line),
                    }
                }
                let total = train_rows.len() + val_rows.len() + test_rows.len();
                zip.start_file("train.jsonl", opts)
                    .map_err(|e| e.to_string())?;
                zip.write_all(train_rows.join("\n").as_bytes())
                    .map_err(|e| e.to_string())?;

                let val_content = if val_rows.is_empty() {
                    train_rows
                        .iter()
                        .take(5)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("\n")
                } else {
                    val_rows.join("\n")
                };
                zip.start_file("val.jsonl", opts)
                    .map_err(|e| e.to_string())?;
                zip.write_all(val_content.as_bytes())
                    .map_err(|e| e.to_string())?;

                zip.start_file("test.jsonl", opts)
                    .map_err(|e| e.to_string())?;
                zip.write_all(test_rows.join("\n").as_bytes())
                    .map_err(|e| e.to_string())?;

                let classes_json = serde_json::to_vec_pretty(
                    &project
                        .classes
                        .iter()
                        .map(|c| c.name.clone())
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| e.to_string())?;
                zip.start_file("classes.json", opts)
                    .map_err(|e| e.to_string())?;
                zip.write_all(&classes_json).map_err(|e| e.to_string())?;
                total
            }
        };

        zip.finish().map_err(|e| e.to_string())?;
        Ok(DatasetExportResult {
            zip_path: dest_zip.to_string_lossy().to_string(),
            item_count,
            class_count: project.classes.len(),
        })
    }

    fn sanitize_slug(input: &str) -> String {
        let s: String = input
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        if s.is_empty() {
            "class".into()
        } else {
            s
        }
    }

    pub fn load_custom_models(&self) -> Vec<CustomModel> {
        self.read_json_or_default(&self.data_dir.join("models").join("custom_registry.json"))
    }

    pub fn save_custom_models(&self, list: &[CustomModel]) -> Result<(), String> {
        self.write_json_atomic(
            &self.data_dir.join("models").join("custom_registry.json"),
            &list,
        )
    }

    pub fn register_custom_model(
        &self,
        name: String,
        module: ProjectModule,
        task: ProjectTask,
        source_kind: String,
        local_weight_src: Option<String>,
        hf_repo_id: Option<String>,
    ) -> Result<CustomModel, String> {
        Self::validate_module_task(&module, &task)?;
        let id = format!("cm_{}", &Uuid::new_v4().to_string().replace('-', "")[..10]);
        let local_rel_path = if let Some(src_path) = local_weight_src {
            let src = PathBuf::from(&src_path);
            if !src.is_file() {
                return Err("Selected custom weight file does not exist.".into());
            }
            let filename = src
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("weights.bin");
            let dest_name = format!("{id}_{filename}");
            let dest = self.data_dir.join("models").join("custom").join(&dest_name);
            fs::copy(&src, &dest).map_err(|e| e.to_string())?;
            Some(format!("models/custom/{dest_name}"))
        } else {
            None
        };

        let item = CustomModel {
            id,
            name: name.trim().to_string(),
            module,
            task,
            source_kind,
            local_rel_path,
            hf_repo_id: hf_repo_id
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
            created_at: Utc::now().to_rfc3339(),
        };
        let mut all = self.load_custom_models();
        all.push(item.clone());
        self.save_custom_models(&all)?;
        Ok(item)
    }

    pub fn delete_custom_model(&self, model_id: &str) -> Result<Vec<CustomModel>, String> {
        let mut all = self.load_custom_models();
        if let Some(pos) = all.iter().position(|m| m.id == model_id) {
            let removed = all.remove(pos);
            if let Some(rel) = removed.local_rel_path {
                let p = self.data_dir.join(rel);
                if p.exists() {
                    let _ = fs::remove_file(p);
                }
            }
        }
        self.save_custom_models(&all)?;
        Ok(all)
    }

    pub fn run_dir(&self, run_id: &str) -> PathBuf {
        self.data_dir.join("runs").join(run_id)
    }

    pub fn delete_run(&self, run_id: &str) -> Result<(), String> {
        let trimmed = run_id.trim();
        if trimmed.is_empty()
            || trimmed.contains("..")
            || trimmed.contains('/')
            || trimmed.contains('\\')
        {
            return Err("Invalid training run identifier.".into());
        }
        let dir = self.run_dir(trimmed);
        if dir.exists() {
            fs::remove_dir_all(dir).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn load_hf_telemetry_cache(&self) -> HashMap<String, HfHubModel> {
        self.read_json_or_default(&self.data_dir.join("hf_telemetry_cache.json"))
    }

    pub fn save_hf_telemetry_cache(
        &self,
        cache: &HashMap<String, HfHubModel>,
    ) -> Result<(), String> {
        self.write_json_atomic(&self.data_dir.join("hf_telemetry_cache.json"), cache)
    }

    pub fn save_run(&self, run: &TrainingRun) -> Result<(), String> {
        let path = self.run_dir(&run.id).join("run.json");
        self.write_json_atomic(&path, run)
    }

    pub fn load_run(&self, run_id: &str) -> Result<TrainingRun, String> {
        let path = self.run_dir(run_id).join("run.json");
        if !path.exists() {
            return Err(format!("Training run {run_id} not found."));
        }
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        serde_json::from_slice(&bytes).map_err(|e| e.to_string())
    }

    pub fn list_runs(&self) -> Result<Vec<TrainingRun>, String> {
        let root = self.data_dir.join("runs");
        if !root.exists() {
            return Ok(vec![]);
        }
        let mut runs = Vec::new();
        for entry in fs::read_dir(root).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let rfile = entry.path().join("run.json");
            if rfile.exists() {
                if let Ok(bytes) = fs::read(&rfile) {
                    if let Ok(run) = serde_json::from_slice::<TrainingRun>(&bytes) {
                        runs.push(run);
                    }
                }
            }
        }
        runs.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(runs)
    }

    pub fn preflight_check(&self, project: &Project) -> PreflightReport {
        let mut split_counts = SplitCounts {
            train: 0,
            val: 0,
            test: 0,
        };
        let mut used_classes: HashSet<String> = HashSet::new();
        let mut missing_ids: Vec<String> = Vec::new();
        let mut unlabeled_ids: Vec<String> = Vec::new();
        let mut labeled_items = 0usize;
        let total_items;

        match project.module {
            ProjectModule::Image => {
                total_items = project.image_items.len();
                for item in &project.image_items {
                    match item.split {
                        SplitType::Train => split_counts.train += 1,
                        SplitType::Val => split_counts.val += 1,
                        SplitType::Test => split_counts.test += 1,
                    }
                    let abs = self.data_dir.join(&item.rel_path);
                    let file_ok =
                        abs.is_file() && fs::metadata(&abs).map(|m| m.len() > 0).unwrap_or(false);
                    if !file_ok {
                        missing_ids.push(item.filename.clone());
                    }
                    let has_label = match project.task {
                        ProjectTask::Detection => {
                            for b in &item.boxes {
                                used_classes.insert(b.class_id.clone());
                            }
                            !item.boxes.is_empty()
                        }
                        ProjectTask::Segmentation => {
                            for p in &item.polygons {
                                used_classes.insert(p.class_id.clone());
                            }
                            !item.polygons.is_empty()
                        }
                        ProjectTask::Classification => {
                            for cid in &item.image_labels {
                                used_classes.insert(cid.clone());
                            }
                            !item.image_labels.is_empty()
                        }
                        _ => false,
                    };
                    if has_label {
                        labeled_items += 1;
                    } else {
                        unlabeled_ids.push(item.filename.clone());
                    }
                }
            }
            ProjectModule::Audio => {
                total_items = project.audio_items.len();
                for item in &project.audio_items {
                    match item.split {
                        SplitType::Train => split_counts.train += 1,
                        SplitType::Val => split_counts.val += 1,
                        SplitType::Test => split_counts.test += 1,
                    }
                    let abs = self.data_dir.join(&item.rel_path);
                    let file_ok =
                        abs.is_file() && fs::metadata(&abs).map(|m| m.len() > 0).unwrap_or(false);
                    if !file_ok {
                        missing_ids.push(item.filename.clone());
                    }
                    let has_label = match project.task {
                        ProjectTask::Classification => {
                            let mut any_cls = false;
                            for s in &item.slices {
                                if let Some(cid) = &s.class_id {
                                    used_classes.insert(cid.clone());
                                    any_cls = true;
                                }
                            }
                            any_cls
                        }
                        ProjectTask::Transcription => {
                            item.slices.iter().any(|s| !s.transcript.trim().is_empty())
                        }
                        _ => !item.slices.is_empty(),
                    };
                    if has_label {
                        labeled_items += 1;
                    } else {
                        unlabeled_ids.push(item.filename.clone());
                    }
                }
            }
            ProjectModule::Text => {
                total_items = project.text_items.len();
                for item in &project.text_items {
                    match item.split {
                        SplitType::Train => split_counts.train += 1,
                        SplitType::Val => split_counts.val += 1,
                        SplitType::Test => split_counts.test += 1,
                    }
                    let has_label = match project.task {
                        ProjectTask::Intent => {
                            if let Some(cid) = &item.intent_class_id {
                                if !cid.is_empty() {
                                    used_classes.insert(cid.clone());
                                }
                            }
                            !item.prompt.trim().is_empty()
                                && item
                                    .intent_class_id
                                    .as_deref()
                                    .map(|c| !c.is_empty())
                                    .unwrap_or(false)
                        }
                        _ => !item.prompt.trim().is_empty() && !item.completion.trim().is_empty(),
                    };
                    if has_label {
                        labeled_items += 1;
                    } else {
                        unlabeled_ids.push(item.id.clone());
                    }
                }
            }
        }

        let needs_classes = matches!(
            project.task,
            ProjectTask::Detection
                | ProjectTask::Segmentation
                | ProjectTask::Classification
                | ProjectTask::Intent
        );
        let empty_classes: Vec<String> = if needs_classes {
            project
                .classes
                .iter()
                .filter(|c| !used_classes.contains(&c.id))
                .map(|c| c.name.clone())
                .collect()
        } else {
            vec![]
        };

        let missing_files = missing_ids.len();
        let unlabeled_items = unlabeled_ids.len();
        let mut issues = Vec::new();

        if total_items == 0 {
            issues.push(PreflightIssue {
                severity: "error".into(),
                code: "NO_ITEMS".into(),
                message: "Project has no dataset items.".into(),
                item_ids: vec![],
            });
        }
        if needs_classes && project.classes.is_empty() {
            issues.push(PreflightIssue {
                severity: "error".into(),
                code: "NO_CLASSES".into(),
                message: "Define at least one class label before training.".into(),
                item_ids: vec![],
            });
        }
        if missing_files > 0 {
            issues.push(PreflightIssue {
                severity: "error".into(),
                code: "MISSING_FILES".into(),
                message: format!("{missing_files} dataset file(s) are missing or empty on disk."),
                item_ids: missing_ids.into_iter().take(20).collect(),
            });
        }
        if total_items > 0 && labeled_items == 0 {
            issues.push(PreflightIssue {
                severity: "error".into(),
                code: "NO_LABELED_ITEMS".into(),
                message: "No labeled items found to train on.".into(),
                item_ids: vec![],
            });
        }
        if unlabeled_items > 0 {
            issues.push(PreflightIssue {
                severity: "warning".into(),
                code: "UNLABELED_ITEMS".into(),
                message: format!("{unlabeled_items} item(s) have no labels or incomplete fields."),
                item_ids: unlabeled_ids.into_iter().take(20).collect(),
            });
        }
        if !empty_classes.is_empty() {
            issues.push(PreflightIssue {
                severity: "warning".into(),
                code: "EMPTY_CLASSES".into(),
                message: format!(
                    "{} class(es) have 0 labeled samples: {}",
                    empty_classes.len(),
                    empty_classes.join(", ")
                ),
                item_ids: vec![],
            });
        }
        if total_items >= 4 && split_counts.val == 0 {
            issues.push(PreflightIssue {
                severity: "warning".into(),
                code: "NO_VAL_SPLIT".into(),
                message: "Validation split currently has 0 items; apply split before training."
                    .into(),
                item_ids: vec![],
            });
        }

        let ok = !issues.iter().any(|i| i.severity == "error");
        PreflightReport {
            ok,
            total_items,
            labeled_items,
            unlabeled_items,
            missing_files,
            empty_classes,
            split_counts,
            issues,
        }
    }

    pub fn list_local_gguf_models(&self) -> Result<Vec<LocalGgufModel>, String> {
        let root = self.data_dir.join("models").join("gguf");
        if !root.is_dir() {
            return Ok(vec![]);
        }
        let mut out = Vec::new();
        for entry in walkdir::WalkDir::new(&root)
            .max_depth(3)
            .into_iter()
            .flatten()
        {
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();
            if ext != "gguf" {
                continue;
            }
            let filename = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            if filename.to_lowercase().contains("mmproj") {
                continue;
            }
            let parent = path.parent().unwrap_or(&root);
            let mmproj_path = fs::read_dir(parent).ok().and_then(|rd| {
                rd.flatten()
                    .map(|e| e.path())
                    .find(|p| {
                        p.is_file()
                            && p.extension().and_then(|e| e.to_str()) == Some("gguf")
                            && p.file_name()
                                .and_then(|n| n.to_str())
                                .map(|n| n.to_lowercase().contains("mmproj"))
                                .unwrap_or(false)
                    })
                    .map(|p| p.to_string_lossy().to_string())
            });
            let repo_id = parent
                .strip_prefix(&root)
                .ok()
                .and_then(|rel| rel.components().next())
                .and_then(|c| c.as_os_str().to_str())
                .map(|s| s.replace("--", "/"))
                .unwrap_or_else(|| "local".to_string());
            let size_bytes = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            let local_str = path.to_string_lossy().to_string();
            out.push(LocalGgufModel {
                id: format!("gguf_{filename}"),
                name: format!("{repo_id} · {filename}"),
                repo_id,
                filename,
                local_path: local_str,
                size_bytes,
                mmproj_path,
            });
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{BoundingBox, TextItem};

    #[test]
    fn test_project_crud_and_text_export() {
        let tmp = std::env::temp_dir().join(format!("damnisium_test_{}", Uuid::new_v4()));
        let mgr = StorageManager::from_data_dir(tmp.clone()).unwrap();

        let p = mgr
            .create_project("QA Test".into(), ProjectModule::Text, ProjectTask::Qa)
            .unwrap();
        assert_eq!(p.app_version, APP_VERSION);

        let mut loaded = mgr.load_project(&p.id).unwrap();
        loaded.text_items.push(TextItem {
            id: "txt_1".into(),
            prompt: "What is Damnisium?".into(),
            completion: "An ML training workstation.".into(),
            system: "Be concise.".into(),
            intent_class_id: None,
            split: SplitType::Train,
            created_at: Utc::now().to_rfc3339(),
        });
        mgr.save_project(&loaded).unwrap();

        let zip_dest = tmp.join("test_dataset.zip");
        let res = mgr
            .package_dataset_zip(&mgr.load_project(&p.id).unwrap(), &zip_dest)
            .unwrap();
        assert_eq!(res.item_count, 1);
        assert!(zip_dest.exists());

        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn test_image_detection_yolo_export() {
        let tmp = std::env::temp_dir().join(format!("damnisium_img_{}", Uuid::new_v4()));
        let mgr = StorageManager::from_data_dir(tmp.clone()).unwrap();

        let mut p = mgr
            .create_project("Det".into(), ProjectModule::Image, ProjectTask::Detection)
            .unwrap();
        let cls_id = mgr.ensure_class_by_name(&mut p, "drone");
        let media_path = tmp
            .join("projects")
            .join(&p.id)
            .join("media")
            .join("img1.png");
        fs::write(&media_path, b"fake_png_bytes").unwrap();
        p.image_items.push(ImageItem {
            id: "img_1".into(),
            filename: "img1.png".into(),
            rel_path: format!("projects/{}/media/img1.png", p.id),
            thumb_rel_path: format!("projects/{}/media/img1.png", p.id),
            width: 640,
            height: 480,
            split: SplitType::Train,
            boxes: vec![BoundingBox {
                id: "b1".into(),
                class_id: cls_id,
                x: 0.1,
                y: 0.2,
                w: 0.4,
                h: 0.3,
                confidence: None,
            }],
            polygons: vec![],
            image_labels: vec![],
            created_at: Utc::now().to_rfc3339(),
        });
        mgr.save_project(&p).unwrap();

        let zip_dest = tmp.join("yolo.zip");
        let res = mgr.package_dataset_zip(&p, &zip_dest).unwrap();
        assert_eq!(res.item_count, 1);
        assert_eq!(res.class_count, 1);

        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn test_preflight_check_reports_issues_accurately() {
        let tmp = std::env::temp_dir().join(format!("damnisium_pf_{}", Uuid::new_v4()));
        let mgr = StorageManager::from_data_dir(tmp.clone()).unwrap();

        let mut p = mgr
            .create_project("PF".into(), ProjectModule::Image, ProjectTask::Detection)
            .unwrap();
        let used_cls = mgr.ensure_class_by_name(&mut p, "car");
        let _unused_cls = mgr.ensure_class_by_name(&mut p, "bus");

        let ok_path = tmp
            .join("projects")
            .join(&p.id)
            .join("media")
            .join("ok.png");
        fs::write(&ok_path, b"pngdata").unwrap();
        p.image_items.push(ImageItem {
            id: "i1".into(),
            filename: "ok.png".into(),
            rel_path: format!("projects/{}/media/ok.png", p.id),
            thumb_rel_path: format!("projects/{}/media/ok.png", p.id),
            width: 640,
            height: 480,
            split: SplitType::Train,
            boxes: vec![BoundingBox {
                id: "b1".into(),
                class_id: used_cls,
                x: 0.1,
                y: 0.1,
                w: 0.2,
                h: 0.2,
                confidence: None,
            }],
            polygons: vec![],
            image_labels: vec![],
            created_at: Utc::now().to_rfc3339(),
        });
        p.image_items.push(ImageItem {
            id: "i2".into(),
            filename: "missing.png".into(),
            rel_path: format!("projects/{}/media/missing.png", p.id),
            thumb_rel_path: format!("projects/{}/media/missing.png", p.id),
            width: 640,
            height: 480,
            split: SplitType::Val,
            boxes: vec![],
            polygons: vec![],
            image_labels: vec![],
            created_at: Utc::now().to_rfc3339(),
        });

        let report = mgr.preflight_check(&p);
        assert!(!report.ok);
        assert_eq!(report.total_items, 2);
        assert_eq!(report.labeled_items, 1);
        assert_eq!(report.unlabeled_items, 1);
        assert_eq!(report.missing_files, 1);
        assert_eq!(report.empty_classes, vec!["bus".to_string()]);
        assert_eq!(report.split_counts.train, 1);
        assert_eq!(report.split_counts.val, 1);

        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn test_delete_run_and_project_cleanup() {
        let tmp = std::env::temp_dir().join(format!("damnisium_del_{}", Uuid::new_v4()));
        let mgr = StorageManager::from_data_dir(tmp.clone()).unwrap();
        let proj = mgr
            .create_project("ToDelete".into(), ProjectModule::Text, ProjectTask::Qa)
            .unwrap();
        let run_dir = mgr.run_dir("run_del_1");
        fs::create_dir_all(&run_dir).unwrap();
        fs::write(run_dir.join("run.json"), br#"{"id":"run_del_1"}"#).unwrap();
        assert!(run_dir.exists());
        mgr.delete_run("run_del_1").unwrap();
        assert!(!run_dir.exists());
        assert!(mgr.delete_run("../bad").is_err());

        mgr.delete_project(&proj.id).unwrap();
        assert!(!mgr.project_dir(&proj.id).exists());
        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn test_small_dataset_split_preserves_val_item() {
        let tmp = std::env::temp_dir().join(format!("damnisium_sp_{}", Uuid::new_v4()));
        let mgr = StorageManager::from_data_dir(tmp.clone()).unwrap();
        let mut p = mgr
            .create_project("Small".into(), ProjectModule::Text, ProjectTask::Qa)
            .unwrap();
        for idx in 0..3 {
            p.text_items.push(TextItem {
                id: format!("t_{idx}"),
                prompt: format!("q{idx}"),
                completion: format!("a{idx}"),
                system: String::new(),
                intent_class_id: None,
                split: SplitType::Train,
                created_at: Utc::now().to_rfc3339(),
            });
        }
        p.split_ratio = crate::models::SplitRatio {
            train: 80,
            val: 15,
            test: 5,
        };
        mgr.apply_split_ratios(&mut p);
        assert_eq!(
            p.text_items
                .iter()
                .filter(|t| t.split == SplitType::Train)
                .count(),
            2
        );
        assert_eq!(
            p.text_items
                .iter()
                .filter(|t| t.split == SplitType::Val)
                .count(),
            1
        );
        assert_eq!(
            p.text_items
                .iter()
                .filter(|t| t.split == SplitType::Test)
                .count(),
            0
        );

        p.split_ratio = crate::models::SplitRatio {
            train: 67,
            val: 33,
            test: 0,
        };
        mgr.apply_split_ratios(&mut p);
        assert_eq!(
            p.text_items
                .iter()
                .filter(|t| t.split == SplitType::Train)
                .count(),
            2
        );
        assert_eq!(
            p.text_items
                .iter()
                .filter(|t| t.split == SplitType::Val)
                .count(),
            1
        );
        assert_eq!(
            p.text_items
                .iter()
                .filter(|t| t.split == SplitType::Test)
                .count(),
            0
        );

        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn test_pick_split_by_index_distributes_splits() {
        let tmp = std::env::temp_dir().join(format!("damnisium_psi_{}", Uuid::new_v4()));
        let mgr = StorageManager::from_data_dir(tmp.clone()).unwrap();
        let mut p = mgr
            .create_project("IdxSplit".into(), ProjectModule::Text, ProjectTask::Qa)
            .unwrap();
        let mut t100 = 0usize;
        let mut v100 = 0usize;
        let mut te100 = 0usize;
        for idx in 0..100 {
            match StorageManager::pick_split_by_index(idx, &p) {
                SplitType::Train => t100 += 1,
                SplitType::Val => v100 += 1,
                SplitType::Test => te100 += 1,
            }
        }
        assert_eq!(t100, 80);
        assert_eq!(v100, 15);
        assert_eq!(te100, 5);

        let mut t20 = 0usize;
        let mut v20 = 0usize;
        let mut te20 = 0usize;
        for idx in 0..20 {
            match StorageManager::pick_split_by_index(idx, &p) {
                SplitType::Train => t20 += 1,
                SplitType::Val => v20 += 1,
                SplitType::Test => te20 += 1,
            }
        }
        assert_eq!(t20, 16);
        assert_eq!(v20, 3);
        assert_eq!(te20, 1);

        p.split_ratio = crate::models::SplitRatio {
            train: 0,
            val: 0,
            test: 0,
        };
        assert_eq!(StorageManager::pick_split_by_index(5, &p), SplitType::Train);

        let _ = fs::remove_dir_all(tmp);
    }
}
