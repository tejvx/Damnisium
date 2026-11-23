mod colab_api;
mod colab_notebook;
mod commands;
mod constants;
mod gguf_runtime;
mod google_drive;
mod google_oauth;
mod hf_hub;
mod media;
mod models;
mod storage;
mod training;

use commands::AppState;
use gguf_runtime::GgufServerState;
use google_oauth::OAuthManager;
use std::sync::{Arc, RwLock};
use storage::StorageManager;
use tauri::{Manager, RunEvent};
use training::PollerRegistry;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let storage = StorageManager::init().expect("failed initializing Damnisium storage");
    let data_dir = storage.data_dir().to_path_buf();
    let pollers = Arc::new(PollerRegistry::default());
    let gguf_server = Arc::new(GgufServerState::default());
    let oauth_manager = Arc::new(OAuthManager::default());

    let state = AppState {
        storage: RwLock::new(storage.clone()),
        pollers: pollers.clone(),
        gguf_server: gguf_server.clone(),
        oauth: oauth_manager.clone(),
    };

    let gguf_exit = gguf_server.clone();
    let oauth_exit = oauth_manager.clone();

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(state)
        .setup(move |app| {
            let _ = app.asset_protocol_scope().allow_directory(&data_dir, true);
            let handle = app.handle().clone();
            let st = storage.clone();
            let reg = pollers.clone();
            tauri::async_runtime::spawn(async move {
                training::resume_active_pollers(handle, st, reg).await;
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_workspace_info,
            commands::save_app_config,
            commands::change_data_directory,
            commands::prepare_google_oauth,
            commands::cancel_google_oauth,
            commands::disconnect_google_account,
            commands::list_projects,
            commands::create_project,
            commands::load_project,
            commands::delete_project,
            commands::add_label_class,
            commands::update_label_class,
            commands::delete_label_class,
            commands::import_images,
            commands::import_image_folder,
            commands::save_image_annotations,
            commands::delete_image_item,
            commands::import_audio_files,
            commands::update_audio_metadata,
            commands::save_audio_slices,
            commands::delete_audio_item,
            commands::upsert_text_item,
            commands::delete_text_item,
            commands::preview_csv_headers,
            commands::import_text_jsonl,
            commands::import_text_csv,
            commands::export_text_dataset,
            commands::update_project_settings,
            commands::export_dataset_zip,
            commands::list_autolabel_models,
            commands::download_autolabel_model,
            commands::import_custom_autolabel_onnx,
            commands::read_binary_file,
            commands::register_custom_model,
            commands::delete_custom_model,
            commands::start_training_run,
            commands::list_training_runs,
            commands::sync_training_run,
            commands::cancel_training_poll,
            commands::rename_training_run,
            commands::delete_training_run,
            commands::export_run_notebook,
            commands::open_local_path,
            commands::open_external_url,
            commands::list_colab_runtime_specs,
            commands::resolve_media_abs_path,
            commands::list_eval_bundles,
            commands::inspect_local_eval_bundle,
            commands::list_folder_media,
            commands::read_text_eval_file,
            commands::check_connectivity,
            commands::run_preflight_check,
            commands::fetch_uncensored_hub,
            commands::inspect_hf_repo,
            commands::download_hf_file,
            commands::list_local_gguf_models,
            commands::get_llama_runtime_status,
            commands::install_llama_runtime,
            commands::run_gguf_chat,
            commands::unload_gguf_runtime,
            commands::export_model_bundle,
        ])
        .build(tauri::generate_context!())
        .expect("error while building Damnisium");

    app.run(move |_app_handle, event| {
        if let RunEvent::Exit = event {
            oauth_exit.cancel();
            gguf_exit.unload();
        }
    });
}
