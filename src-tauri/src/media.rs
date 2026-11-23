use crate::models::AutoLabelProgressPayload;
use reqwest::Client;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use tauri::{AppHandle, Emitter};

pub fn inspect_image_dimensions(path: &Path) -> Result<(u32, u32), String> {
    image::image_dimensions(path).map_err(|e| format!("Failed reading image dimensions: {e}"))
}

pub fn create_thumbnail(src: &Path, dst: &Path) -> Result<(), String> {
    let img = image::open(src).map_err(|e| format!("Failed opening image for thumbnail: {e}"))?;
    let thumb = img.thumbnail(192, 192);
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    thumb
        .to_rgb8()
        .save_with_format(dst, image::ImageFormat::Jpeg)
        .map_err(|e| format!("Failed saving thumbnail: {e}"))
}

pub fn inspect_audio_header(path: &Path) -> (f64, u32, u16) {
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return (0.0, 0, 0),
    };
    let mut buf = [0u8; 4096];
    let n = match file.read(&mut buf) {
        Ok(n) if n >= 44 => n,
        _ => return (0.0, 0, 0),
    };
    if &buf[0..4] != b"RIFF" || &buf[8..12] != b"WAVE" {
        return (0.0, 0, 0);
    }
    let mut pos = 12usize;
    let mut channels: u16 = 0;
    let mut sample_rate: u32 = 0;
    let mut byte_rate: u32 = 0;
    let mut data_size: u32 = 0;
    while pos + 8 <= n {
        let chunk_id = &buf[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            buf[pos + 4],
            buf[pos + 5],
            buf[pos + 6],
            buf[pos + 7],
        ]);
        if chunk_id == b"fmt " && pos + 24 <= n {
            channels = u16::from_le_bytes([buf[pos + 10], buf[pos + 11]]);
            sample_rate = u32::from_le_bytes([
                buf[pos + 12],
                buf[pos + 13],
                buf[pos + 14],
                buf[pos + 15],
            ]);
            byte_rate = u32::from_le_bytes([
                buf[pos + 16],
                buf[pos + 17],
                buf[pos + 18],
                buf[pos + 19],
            ]);
        } else if chunk_id == b"data" {
            data_size = chunk_size;
            break;
        }
        let step = 8usize.saturating_add(chunk_size as usize).saturating_add((chunk_size & 1) as usize);
        if step == 0 {
            break;
        }
        pos = pos.saturating_add(step);
    }
    if data_size == 0 {
        if let Ok(meta) = std::fs::metadata(path) {
            let len = meta.len();
            if len > 44 {
                data_size = (len - 44).min(u32::MAX as u64) as u32;
            }
        }
    }
    let duration = if byte_rate > 0 {
        data_size as f64 / byte_rate as f64
    } else {
        0.0
    };
    (duration, sample_rate, channels)
}

pub async fn download_file_stream(
    app: &AppHandle,
    url: &str,
    dest: &Path,
    model_id: &str,
) -> Result<u64, String> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let client = Client::builder()
        .user_agent("Damnisium/v1-beta")
        .build()
        .map_err(|e| e.to_string())?;
    let mut resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("Network request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("Download failed with HTTP {}", resp.status()));
    }
    let total_bytes = resp.content_length();
    let tmp_path = dest.with_extension("part");
    let mut out = File::create(&tmp_path).map_err(|e| e.to_string())?;
    let mut downloaded: u64 = 0;
    let mut last_emit: u64 = 0;
    while let Some(chunk) = resp.chunk().await.map_err(|e| e.to_string())? {
        if let Err(e) = out.write_all(&chunk) {
            drop(out);
            let _ = std::fs::remove_file(&tmp_path);
            return Err(e.to_string());
        }
        downloaded += chunk.len() as u64;
        if downloaded - last_emit >= 131_072 {
            last_emit = downloaded;
            let _ = app.emit(
                "autolabel-download-progress",
                AutoLabelProgressPayload {
                    model_id: model_id.to_string(),
                    downloaded_bytes: downloaded,
                    total_bytes,
                    done: false,
                },
            );
        }
    }
    if let Err(e) = out.flush() {
        drop(out);
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e.to_string());
    }
    drop(out);
    if downloaded == 0 {
        let _ = std::fs::remove_file(&tmp_path);
        return Err("Downloaded model file is empty.".into());
    }
    if let Some(expected) = total_bytes {
        if downloaded != expected {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(format!(
                "Incomplete model download: got {downloaded} bytes, expected {expected}."
            ));
        }
    }
    let disk_size = std::fs::metadata(&tmp_path).map(|m| m.len()).unwrap_or(0);
    if disk_size != downloaded
        || (dest.extension().and_then(|e| e.to_str()) == Some("onnx") && disk_size < 1024)
    {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(format!("Downloaded file failed integrity check ({disk_size} bytes)."));
    }
    if dest.exists() {
        let _ = std::fs::remove_file(dest);
    }
    std::fs::rename(&tmp_path, dest).map_err(|e| e.to_string())?;
    let _ = app.emit(
        "autolabel-download-progress",
        AutoLabelProgressPayload {
            model_id: model_id.to_string(),
            downloaded_bytes: downloaded,
            total_bytes,
            done: true,
        },
    );
    Ok(downloaded)
}
