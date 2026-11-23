use reqwest::multipart::{Form, Part};
use reqwest::{Client, Response};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::File;
use std::io::Read;
use std::path::Path;

const DRIVE_FILES_API: &str = "https://www.googleapis.com/drive/v3/files";
const DRIVE_UPLOAD_API: &str = "https://www.googleapis.com/upload/drive/v3/files";
const DRIVE_CHUNK_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Deserialize)]
pub struct DriveFileItem {
    pub id: String,
    #[serde(default)]
    pub name: String,
    pub size: Option<String>,
    pub trashed: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct DriveFileList {
    #[serde(default)]
    files: Vec<DriveFileItem>,
}

#[derive(Debug, Serialize)]
struct CreateFileMeta<'a> {
    name: &'a str,
    #[serde(rename = "mimeType")]
    mime_type: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    parents: Option<Vec<&'a str>>,
}

fn escape_drive_query(input: &str) -> String {
    input.replace('\\', "\\\\").replace('\'', "\\'")
}

fn format_drive_error(context: &str, status: u16, body: &str) -> String {
    if let Ok(val) = serde_json::from_str::<Value>(body) {
        if let Some(err) = val.get("error") {
            let code = err
                .get("code")
                .and_then(|c| c.as_u64())
                .unwrap_or(status as u64);
            let msg = err
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Google Drive request rejected");
            let reason = err
                .get("errors")
                .and_then(|arr| arr.as_array())
                .and_then(|arr| arr.first())
                .and_then(|first| first.get("reason"))
                .and_then(|r| r.as_str());
            if let Some(r) = reason {
                return format!("{context} ({code} {r}): {msg}");
            }
            return format!("{context} ({code}): {msg}");
        }
    }
    let trimmed = body.trim();
    if trimmed.is_empty() {
        format!("{context} (HTTP {status})")
    } else {
        format!("{context} (HTTP {status}): {trimmed}")
    }
}

async fn decode_drive_json<T: DeserializeOwned>(
    resp: Response,
    context: &str,
) -> Result<T, String> {
    let status = resp.status();
    let body = resp
        .text()
        .await
        .map_err(|e| format!("{context} response read failed: {e}"))?;
    if !status.is_success() {
        return Err(format_drive_error(context, status.as_u16(), &body));
    }
    serde_json::from_str::<T>(&body).map_err(|e| {
        let snippet: String = body.chars().take(240).collect();
        format!("{context} JSON parse failed ({e}): {snippet}")
    })
}

pub async fn ensure_folder(
    client: &Client,
    token: &str,
    name: &str,
    parent_id: Option<&str>,
) -> Result<String, String> {
    let escaped = escape_drive_query(name);
    let query = match parent_id {
        Some(pid) => format!(
            "name = '{escaped}' and mimeType = 'application/vnd.google-apps.folder' and '{pid}' in parents and trashed = false"
        ),
        None => format!(
            "name = '{escaped}' and mimeType = 'application/vnd.google-apps.folder' and 'root' in parents and trashed = false"
        ),
    };
    let resp = client
        .get(DRIVE_FILES_API)
        .bearer_auth(token)
        .query(&[
            ("q", query.as_str()),
            ("fields", "files(id, name, mimeType)"),
            ("pageSize", "5"),
        ])
        .send()
        .await
        .map_err(|e| format!("Drive folder lookup failed: {e}"))?;
    let list: DriveFileList = decode_drive_json(resp, "Drive folder lookup error").await?;
    if let Some(first) = list.files.into_iter().next() {
        return Ok(first.id);
    }

    let meta = CreateFileMeta {
        name,
        mime_type: "application/vnd.google-apps.folder",
        parents: parent_id.map(|p| vec![p]),
    };
    let created = client
        .post(DRIVE_FILES_API)
        .bearer_auth(token)
        .json(&meta)
        .send()
        .await
        .map_err(|e| format!("Drive folder creation failed: {e}"))?;
    let item: DriveFileItem = decode_drive_json(created, "Drive folder create error").await?;
    Ok(item.id)
}

pub async fn upload_bytes_multipart(
    client: &Client,
    token: &str,
    parent_id: &str,
    filename: &str,
    mime_type: &str,
    bytes: Vec<u8>,
) -> Result<String, String> {
    let file_len_expected = bytes.len() as u64;
    let meta = CreateFileMeta {
        name: filename,
        mime_type,
        parents: Some(vec![parent_id]),
    };
    let meta_json = serde_json::to_string(&meta).map_err(|e| e.to_string())?;
    let meta_part = Part::text(meta_json)
        .mime_str("application/json; charset=UTF-8")
        .map_err(|e| e.to_string())?;
    let file_part = Part::bytes(bytes)
        .file_name(filename.to_string())
        .mime_str(mime_type)
        .map_err(|e| e.to_string())?;
    let form = Form::new()
        .part("metadata", meta_part)
        .part("file", file_part);

    let resp = client
        .post(format!(
            "{DRIVE_UPLOAD_API}?uploadType=multipart&fields=id,name"
        ))
        .bearer_auth(token)
        .multipart(form)
        .send()
        .await
        .map_err(|e| format!("Drive upload failed: {e}"))?;
    let item: DriveFileItem = decode_drive_json(resp, "Drive multipart upload error").await?;
    verify_uploaded_file(client, token, &item.id, file_len_expected).await?;
    Ok(item.id)
}

pub async fn verify_uploaded_file(
    client: &Client,
    token: &str,
    file_id: &str,
    expected_bytes: u64,
) -> Result<DriveFileItem, String> {
    if expected_bytes == 0 {
        return Err("Cannot upload an empty file to Google Drive.".into());
    }
    let resp = client
        .get(format!("{DRIVE_FILES_API}/{file_id}"))
        .bearer_auth(token)
        .query(&[("fields", "id,name,mimeType,size,trashed")])
        .send()
        .await
        .map_err(|e| format!("Drive post-upload verification request failed: {e}"))?;
    let item: DriveFileItem = decode_drive_json(resp, "Drive upload verification failed").await?;
    if item.trashed.unwrap_or(false) {
        return Err(format!("Uploaded Drive file {file_id} is marked trashed."));
    }
    let remote_size = item
        .size
        .as_deref()
        .and_then(|s| s.parse::<u64>().ok())
        .ok_or_else(|| format!("Drive did not report byte size for uploaded file {file_id}."))?;
    if remote_size != expected_bytes {
        return Err(format!(
            "Drive upload size mismatch for {}: local {expected_bytes} bytes, remote {remote_size} bytes.",
            item.name
        ));
    }
    Ok(item)
}

pub async fn upload_file_resumable(
    client: &Client,
    token: &str,
    parent_id: &str,
    filename: &str,
    mime_type: &str,
    file_path: &Path,
) -> Result<String, String> {
    let mut file =
        File::open(file_path).map_err(|e| format!("Failed reading file to upload: {e}"))?;
    let total_len = file.metadata().map(|m| m.len()).unwrap_or(0);
    if total_len <= 4 * 1024 * 1024 {
        let mut data = Vec::with_capacity(total_len as usize);
        file.read_to_end(&mut data).map_err(|e| e.to_string())?;
        return upload_bytes_multipart(client, token, parent_id, filename, mime_type, data).await;
    }

    let meta = CreateFileMeta {
        name: filename,
        mime_type,
        parents: Some(vec![parent_id]),
    };
    let init_resp = client
        .post(format!(
            "{DRIVE_UPLOAD_API}?uploadType=resumable&fields=id,name"
        ))
        .bearer_auth(token)
        .header("X-Upload-Content-Type", mime_type)
        .header("X-Upload-Content-Length", total_len.to_string())
        .json(&meta)
        .send()
        .await
        .map_err(|e| format!("Resumable upload init failed: {e}"))?;
    if !init_resp.status().is_success() {
        let status = init_resp.status().as_u16();
        let body = init_resp.text().await.unwrap_or_default();
        return Err(format_drive_error("Resumable init error", status, &body));
    }
    let upload_url = init_resp
        .headers()
        .get("location")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| "Missing upload session location from Google Drive.".to_string())?
        .to_string();

    let mut buf = vec![0u8; DRIVE_CHUNK_BYTES];
    let mut offset = 0u64;
    let mut uploaded_item: Option<DriveFileItem> = None;
    while offset < total_len {
        let mut filled = 0usize;
        while filled < buf.len() {
            let n = file.read(&mut buf[filled..]).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            filled += n;
        }
        if filled == 0 {
            break;
        }
        let end = offset + filled as u64 - 1;
        let range = format!("bytes {offset}-{end}/{total_len}");
        let resp = client
            .put(&upload_url)
            .header("Content-Type", mime_type)
            .header("Content-Range", range)
            .body(buf[..filled].to_vec())
            .send()
            .await
            .map_err(|e| format!("Resumable chunk PUT failed: {e}"))?;
        let status = resp.status().as_u16();
        offset += filled as u64;
        if status == 308 {
            continue;
        }
        let item: DriveFileItem = decode_drive_json(resp, "Resumable upload error").await?;
        uploaded_item = Some(item);
        break;
    }
    let item = uploaded_item
        .ok_or_else(|| "Google Drive resumable upload did not finalize.".to_string())?;
    verify_uploaded_file(client, token, &item.id, total_len).await?;
    Ok(item.id)
}

pub async fn download_text(client: &Client, token: &str, file_id: &str) -> Result<String, String> {
    let resp = client
        .get(format!("{DRIVE_FILES_API}/{file_id}?alt=media"))
        .bearer_auth(token)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let status = resp.status();
    let body = resp.text().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format_drive_error(
            "Drive text download error",
            status.as_u16(),
            &body,
        ));
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drive_item_id_only_and_error_formatting() {
        let parsed: DriveFileItem = serde_json::from_str(r#"{"id":"1AbCdEf"}"#).unwrap();
        assert_eq!(parsed.id, "1AbCdEf");
        assert_eq!(parsed.name, "");

        let raw = r#"{"error":{"code":403,"message":"Google Drive API disabled.","errors":[{"reason":"accessNotConfigured"}]}}"#;
        let msg = format_drive_error("Drive folder lookup error", 403, raw);
        assert!(msg.contains("403 accessNotConfigured"));
        assert!(msg.contains("Google Drive API disabled."));
    }
}
