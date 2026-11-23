use crate::models::{
    AppConfig, GoogleAuthStatus, GoogleAuthStored, GoogleUser, OAuthProgressPayload,
    PreparedOAuthSession,
};
use crate::storage::StorageManager;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use chrono::Utc;
use rand::distributions::Alphanumeric;
use rand::Rng;
use reqwest::Client;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;
use url::Url;

const OBF_MASK: [u8; 16] = [
    0x9A, 0x4F, 0xC3, 0x71, 0x5E, 0xB2, 0x29, 0xE7, 0x3D, 0x84, 0x6A, 0xD5, 0x18, 0xF1, 0xA6,
    0x53,
];
const OBF_CID: &[u8] = &[
    0xAE, 0x7A, 0xEE, 0x61, 0x5D, 0xC3, 0x5F, 0x8B, 0x6D, 0xC8, 0xDE, 0x68, 0xA9, 0x2B, 0x29,
    0xFD, 0x3B, 0xE7, 0x58, 0xE3, 0x68, 0x90, 0x52, 0xA5, 0x61, 0xB0, 0x5E, 0xE9, 0x46, 0xFD,
    0x43, 0xAB, 0x0C, 0x8C, 0x0B, 0xC5, 0xEC, 0x3A, 0xA8, 0x73, 0x47, 0xF5, 0x3B, 0x8E, 0x40,
    0x96, 0x91, 0x40, 0x9A, 0x41, 0x67, 0x81, 0x95, 0x6C, 0xF0, 0x40, 0x80, 0x14, 0xEB, 0x4F,
    0x66, 0x8B, 0xEF, 0x0E, 0xAE, 0x67, 0xF7, 0x62, 0x04, 0x50, 0xC8, 0x11,
];
const OBF_CSEC: &[u8] = &[
    0xDD, 0x0D, 0x9A, 0x05, 0x3A, 0xAB, 0x4A, 0xF4, 0x00, 0xB6, 0xD0, 0x2A, 0xCE, 0x35, 0x4F,
    0xC7, 0x30, 0xC4, 0x19, 0xD5, 0x2D, 0xCB, 0x60, 0x88, 0x35, 0x8A, 0x49, 0xEF, 0x13, 0xCB,
    0x19, 0x85, 0x76, 0x87, 0x0B,
];
const OBF_PID: &[u8] = &[
    0xAE, 0x7A, 0xEE, 0x61, 0x5D, 0xC3, 0x5F, 0x8B, 0x6D, 0xC8, 0xDE, 0x68,
];

fn decode_obf(data: &[u8]) -> String {
    let bytes: Vec<u8> = data
        .iter()
        .enumerate()
        .map(|(i, b)| b ^ OBF_MASK[i % OBF_MASK.len()] ^ ((i.wrapping_mul(13) & 0xFF) as u8))
        .collect();
    String::from_utf8(bytes).unwrap_or_default()
}

pub fn default_oauth_client_id() -> String {
    decode_obf(OBF_CID)
}

pub fn default_oauth_client_secret() -> String {
    decode_obf(OBF_CSEC)
}

pub fn default_gcp_project_id() -> String {
    decode_obf(OBF_PID)
}

pub const SCOPE_COLAB: &str = "https://www.googleapis.com/auth/colaboratory";
pub const SCOPE_USERINFO_EMAIL: &str = "https://www.googleapis.com/auth/userinfo.email";
pub const SCOPE_DRIVE_FILE: &str = "https://www.googleapis.com/auth/drive.file";

const GOOGLE_AUTH_ENDPOINT: &str = "https://accounts.google.com/o/oauth2/v2/auth";
const GOOGLE_TOKEN_ENDPOINT: &str = "https://oauth2.googleapis.com/token";
const GOOGLE_USERINFO_ENDPOINT: &str = "https://www.googleapis.com/oauth2/v3/userinfo";

#[derive(Default)]
pub struct OAuthManager {
    pending: Mutex<Option<JoinHandle<()>>>,
}

impl Drop for OAuthManager {
    fn drop(&mut self) {
        self.cancel();
    }
}

impl OAuthManager {
    pub fn cancel(&self) {
        if let Ok(mut guard) = self.pending.lock() {
            if let Some(handle) = guard.take() {
                handle.abort();
            }
        }
    }

    pub fn replace(&self, handle: JoinHandle<()>) {
        if let Ok(mut guard) = self.pending.lock() {
            if let Some(prev) = guard.take() {
                prev.abort();
            }
            *guard = Some(handle);
        }
    }
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    expires_in: Option<i64>,
    refresh_token: Option<String>,
    scope: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UserInfoResponse {
    sub: Option<String>,
    email: Option<String>,
    name: Option<String>,
    picture: Option<String>,
}

pub fn resolve_oauth_credentials(config: &AppConfig) -> Option<(String, String)> {
    let cid = config.oauth_client_id.trim();
    if !cid.is_empty() {
        return Some((
            cid.to_string(),
            config.oauth_client_secret.trim().to_string(),
        ));
    }
    let def_id = default_oauth_client_id();
    let def_trimmed = def_id.trim();
    if !def_trimmed.is_empty() {
        return Some((
            def_trimmed.to_string(),
            default_oauth_client_secret().trim().to_string(),
        ));
    }
    None
}

pub fn has_oauth_credentials(config: &AppConfig) -> bool {
    resolve_oauth_credentials(config).is_some()
}

pub fn resolve_gcp_project_id(config: &AppConfig) -> String {
    let trimmed = config.gcp_project_id.trim();
    if !trimmed.is_empty() {
        return trimmed.to_string();
    }
    let cid = config.oauth_client_id.trim();
    if !cid.is_empty() {
        if let Some(prefix) = cid.split('-').next() {
            if !prefix.is_empty() && prefix.bytes().all(|b| b.is_ascii_digit()) {
                return prefix.to_string();
            }
        }
    }
    default_gcp_project_id()
}

pub fn resolve_oauth_scopes(config: &AppConfig) -> String {
    if config.enable_drive_backup {
        format!("{SCOPE_COLAB} {SCOPE_USERINFO_EMAIL} {SCOPE_DRIVE_FILE}")
    } else {
        format!("{SCOPE_COLAB} {SCOPE_USERINFO_EMAIL}")
    }
}

pub fn has_colab_scope(scopes: &str) -> bool {
    scopes.split_whitespace().any(|s| s == SCOPE_COLAB)
}

pub fn generate_pkce_pair() -> (String, String) {
    let verifier: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(64)
        .map(char::from)
        .collect();
    let digest = Sha256::digest(verifier.as_bytes());
    let challenge = URL_SAFE_NO_PAD.encode(digest);
    (verifier, challenge)
}

pub fn generate_state_token() -> String {
    rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(32)
        .map(char::from)
        .collect()
}

pub fn build_google_auth_url(
    client_id: &str,
    redirect_uri: &str,
    state: &str,
    code_challenge: &str,
    scopes: &str,
) -> Result<String, String> {
    let mut parsed = Url::parse(GOOGLE_AUTH_ENDPOINT).map_err(|e| e.to_string())?;
    parsed
        .query_pairs_mut()
        .append_pair("client_id", client_id.trim())
        .append_pair("redirect_uri", redirect_uri)
        .append_pair("response_type", "code")
        .append_pair("scope", scopes)
        .append_pair("access_type", "offline")
        .append_pair("prompt", "consent")
        .append_pair("state", state)
        .append_pair("code_challenge", code_challenge)
        .append_pair("code_challenge_method", "S256");
    Ok(parsed.to_string())
}

fn render_callback_html(success: bool, title: &str, subtitle: &str) -> String {
    let accent = if success { "#10b981" } else { "#ef4444" };
    let safe_title = title
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    let safe_sub = subtitle
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Damnisium</title></head><body style=\"background:#09090b;color:#ffffff;font-family:Inter,Segoe UI,system-ui,sans-serif;display:grid;place-items:center;height:100vh;margin:0\"><div style=\"border:1px solid {accent};background:#17151a;padding:28px 36px;border-radius:10px;max-width:440px;text-align:center\"><h2 style=\"margin:0 0 8px 0;font-size:18px\">{safe_title}</h2><p style=\"margin:0;color:#a1a1aa;font-size:13px;line-height:1.5\">{safe_sub}</p></div></body></html>"
    )
}

async fn write_http_response(socket: &mut TcpStream, status_line: &str, html: &str) {
    let resp = format!(
        "HTTP/1.1 {status_line}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n{}",
        html.len(),
        html
    );
    let _ = socket.write_all(resp.as_bytes()).await;
    let _ = socket.flush().await;
    let _ = socket.shutdown().await;
}

pub async fn prepare_oauth_session(
    app: AppHandle,
    storage: StorageManager,
    config: &AppConfig,
    manager: &Arc<OAuthManager>,
) -> Result<PreparedOAuthSession, String> {
    manager.cancel();
    let (client_id, client_secret) = resolve_oauth_credentials(config).ok_or_else(|| {
        "Google Desktop OAuth credentials are not configured yet. Configure Client ID in Settings > Advanced OAuth Override.".to_string()
    })?;

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| format!("Failed binding localhost OAuth callback listener: {e}"))?;
    let port = listener.local_addr().map_err(|e| e.to_string())?.port();
    let redirect_uri = format!("http://127.0.0.1:{port}/callback");
    let (verifier, challenge) = generate_pkce_pair();
    let state = generate_state_token();
    let requested_scopes = resolve_oauth_scopes(config);
    let auth_url = build_google_auth_url(
        &client_id,
        &redirect_uri,
        &state,
        &challenge,
        &requested_scopes,
    )?;

    let app_handle = app.clone();
    let st_clone = storage.clone();
    let rd_clone = redirect_uri.clone();
    let st_expected = state.clone();

    let handle = tokio::spawn(async move {
        let emit_state = |payload: OAuthProgressPayload| {
            let _ = app_handle.emit("google-oauth-state", payload);
        };
        let outcome = tokio::time::timeout(
            Duration::from_secs(300),
            run_oauth_callback_once(
                listener,
                &st_clone,
                &client_id,
                &client_secret,
                &rd_clone,
                &verifier,
                &st_expected,
                &requested_scopes,
                GOOGLE_TOKEN_ENDPOINT,
                GOOGLE_USERINFO_ENDPOINT,
                |msg| {
                    emit_state(OAuthProgressPayload {
                        phase: "verifying".into(),
                        message: msg,
                        error: None,
                        auth: None,
                    });
                },
            ),
        )
        .await;

        match outcome {
            Ok(Ok(status)) => {
                emit_state(OAuthProgressPayload {
                    phase: "verified".into(),
                    message: format!(
                        "Verified {}",
                        status
                            .user
                            .as_ref()
                            .map(|u| u.email.as_str())
                            .unwrap_or("Google Account")
                    ),
                    error: None,
                    auth: Some(status),
                });
            }
            Ok(Err(err_msg)) => {
                emit_state(OAuthProgressPayload {
                    phase: "error".into(),
                    message: "Authentication failed".into(),
                    error: Some(err_msg),
                    auth: None,
                });
            }
            Err(_) => {
                emit_state(OAuthProgressPayload {
                    phase: "error".into(),
                    message: "Authentication timed out".into(),
                    error: Some("Google OAuth sign-in timed out after 300 seconds.".into()),
                    auth: None,
                });
            }
        }
    });

    manager.replace(handle);

    Ok(PreparedOAuthSession {
        auth_url,
        redirect_uri,
        expires_in_sec: 300,
    })
}

#[allow(clippy::too_many_arguments)]
async fn run_oauth_callback_once<F>(
    listener: TcpListener,
    storage: &StorageManager,
    client_id: &str,
    client_secret: &str,
    redirect_uri: &str,
    code_verifier: &str,
    expected_state: &str,
    requested_scopes: &str,
    token_endpoint: &str,
    userinfo_endpoint: &str,
    on_verifying: F,
) -> Result<GoogleAuthStatus, String>
where
    F: Fn(String),
{
    loop {
        let (mut socket, _) = listener.accept().await.map_err(|e| e.to_string())?;
        let mut buf = [0u8; 8192];
        let n = socket.read(&mut buf).await.map_err(|e| e.to_string())?;
        if n == 0 {
            continue;
        }
        let req = String::from_utf8_lossy(&buf[..n]);
        let first_line = req.lines().next().unwrap_or("");
        let path_part = first_line.split_whitespace().nth(1).unwrap_or("/");
        let full = format!("http://127.0.0.1{path_part}");
        let Ok(parsed) = Url::parse(&full) else {
            let html = render_callback_html(
                false,
                "Authentication failed",
                "Malformed callback request.",
            );
            write_http_response(&mut socket, "400 Bad Request", &html).await;
            return Err("Malformed OAuth callback URL.".into());
        };

        if parsed.path() != "/callback" {
            let _ = socket
                .write_all(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .await;
            let _ = socket.shutdown().await;
            continue;
        }

        let mut code = None;
        let mut returned_state = None;
        let mut oauth_error = None;
        let mut oauth_error_desc = None;
        for (k, v) in parsed.query_pairs() {
            if k == "code" {
                code = Some(v.into_owned());
            } else if k == "state" {
                returned_state = Some(v.into_owned());
            } else if k == "error" {
                oauth_error = Some(v.into_owned());
            } else if k == "error_description" {
                oauth_error_desc = Some(v.into_owned());
            }
        }

        if let Some(err) = oauth_error {
            let detail = oauth_error_desc.unwrap_or(err);
            let html = render_callback_html(
                false,
                "Authentication failed",
                "Authentication failed — check Damnisium for details.",
            );
            write_http_response(&mut socket, "400 Bad Request", &html).await;
            return Err(format!("Google sign-in declined: {detail}"));
        }

        if returned_state.as_deref() != Some(expected_state) {
            let html = render_callback_html(
                false,
                "Authentication failed",
                "Authentication failed — check Damnisium for details.",
            );
            write_http_response(&mut socket, "400 Bad Request", &html).await;
            return Err("OAuth CSRF state validation failed.".into());
        }

        let Some(auth_code) = code.filter(|c| !c.trim().is_empty()) else {
            let html = render_callback_html(
                false,
                "Authentication failed",
                "Authentication failed — check Damnisium for details.",
            );
            write_http_response(&mut socket, "400 Bad Request", &html).await;
            return Err("Missing authorization code in callback.".into());
        };

        on_verifying("Verifying account...".into());

        let result = verify_and_store_tokens(
            storage,
            client_id,
            client_secret,
            redirect_uri,
            code_verifier,
            &auth_code,
            requested_scopes,
            token_endpoint,
            userinfo_endpoint,
        )
        .await;

        match result {
            Ok(status) => {
                let email = status
                    .user
                    .as_ref()
                    .map(|u| u.email.as_str())
                    .unwrap_or("your Google account");
                let html = render_callback_html(
                    true,
                    "Authentication complete — you can close this tab",
                    &format!("Verified active account: {email}"),
                );
                write_http_response(&mut socket, "200 OK", &html).await;
                return Ok(status);
            }
            Err(err) => {
                let html = render_callback_html(
                    false,
                    "Authentication failed",
                    "Authentication failed — check Damnisium for details.",
                );
                write_http_response(&mut socket, "400 Bad Request", &html).await;
                return Err(err);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn verify_and_store_tokens(
    storage: &StorageManager,
    client_id: &str,
    client_secret: &str,
    redirect_uri: &str,
    code_verifier: &str,
    auth_code: &str,
    requested_scopes: &str,
    token_endpoint: &str,
    userinfo_endpoint: &str,
) -> Result<GoogleAuthStatus, String> {
    let client = Client::builder()
        .timeout(Duration::from_secs(20))
        .build()
        .map_err(|e| e.to_string())?;

    let mut form: HashMap<&str, String> = HashMap::new();
    form.insert("client_id", client_id.to_string());
    if !client_secret.is_empty() {
        form.insert("client_secret", client_secret.to_string());
    }
    form.insert("code", auth_code.to_string());
    form.insert("code_verifier", code_verifier.to_string());
    form.insert("redirect_uri", redirect_uri.to_string());
    form.insert("grant_type", "authorization_code".to_string());

    let token_resp = client
        .post(token_endpoint)
        .form(&form)
        .send()
        .await
        .map_err(|e| format!("Token exchange network error: {e}"))?;
    if !token_resp.status().is_success() {
        let status = token_resp.status();
        let body = token_resp.text().await.unwrap_or_default();
        return Err(format!(
            "Google token exchange returned HTTP {status}: {body}"
        ));
    }

    let token_data: TokenResponse = token_resp
        .json()
        .await
        .map_err(|e| format!("Invalid token JSON from Google: {e}"))?;
    if token_data.access_token.trim().is_empty() {
        return Err("Google token endpoint returned an empty access token.".into());
    }

    let user =
        fetch_verified_user_profile(&client, &token_data.access_token, userinfo_endpoint).await?;

    let granted_scopes = token_data
        .scope
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(requested_scopes)
        .to_string();
    let colab_authorized = has_colab_scope(&granted_scopes);
    let expires_at = Utc::now().timestamp() + token_data.expires_in.unwrap_or(3600);
    let existing = storage.load_auth().unwrap_or_default();
    let stored = GoogleAuthStored {
        access_token: token_data.access_token,
        refresh_token: token_data.refresh_token.unwrap_or(existing.refresh_token),
        expires_at_unix: expires_at,
        user: Some(user.clone()),
        drive_root_folder_id: existing.drive_root_folder_id.clone(),
        scopes: granted_scopes.clone(),
    };
    storage.save_auth(&stored)?;

    let confirmed = storage.load_auth()?;
    if confirmed.access_token != stored.access_token
        || confirmed
            .user
            .as_ref()
            .map(|u| u.email.as_str())
            .unwrap_or("")
            != user.email
    {
        return Err("Failed confirming persisted credentials on disk.".into());
    }

    Ok(GoogleAuthStatus {
        connected: true,
        user: Some(user),
        has_credentials: true,
        drive_folder_id: confirmed.drive_root_folder_id,
        scopes: granted_scopes,
        colab_authorized,
    })
}

async fn fetch_verified_user_profile(
    client: &Client,
    access_token: &str,
    userinfo_endpoint: &str,
) -> Result<GoogleUser, String> {
    let resp = client
        .get(userinfo_endpoint)
        .bearer_auth(access_token)
        .send()
        .await
        .map_err(|e| format!("Account verification request failed: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        return Err(format!(
            "Google account verification failed with HTTP {status}."
        ));
    }
    let info: UserInfoResponse = resp
        .json()
        .await
        .map_err(|e| format!("Failed parsing userinfo response: {e}"))?;
    let email = info.email.unwrap_or_default().trim().to_string();
    let sub = info.sub.unwrap_or_default().trim().to_string();
    if !email.contains('@') || sub.is_empty() {
        return Err("Google userinfo did not return an active account ID and email.".into());
    }
    Ok(GoogleUser {
        email: email.clone(),
        name: info
            .name
            .map(|n| n.trim().to_string())
            .filter(|n| !n.is_empty())
            .unwrap_or(email),
        picture: info.picture.unwrap_or_default(),
        sub,
    })
}

async fn refresh_access_token_now(
    storage: &StorageManager,
    config: &AppConfig,
    stored: &mut GoogleAuthStored,
) -> Result<String, String> {
    if stored.refresh_token.is_empty() {
        return Err("Google session expired. Please reconnect in Settings.".into());
    }
    let (client_id, client_secret) = resolve_oauth_credentials(config)
        .ok_or_else(|| "Google OAuth credentials are missing.".to_string())?;
    let client = Client::new();
    let mut form: HashMap<&str, String> = HashMap::new();
    form.insert("client_id", client_id);
    if !client_secret.is_empty() {
        form.insert("client_secret", client_secret);
    }
    form.insert("refresh_token", stored.refresh_token.clone());
    form.insert("grant_type", "refresh_token".to_string());

    let resp = client
        .post(GOOGLE_TOKEN_ENDPOINT)
        .form(&form)
        .send()
        .await
        .map_err(|e| format!("Token refresh failed: {e}"))?;
    if !resp.status().is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Google token refresh failed: {body}"));
    }
    let token_data: TokenResponse = resp.json().await.map_err(|e| e.to_string())?;
    stored.access_token = token_data.access_token.clone();
    stored.expires_at_unix = Utc::now().timestamp() + token_data.expires_in.unwrap_or(3600);
    if let Some(rt) = token_data.refresh_token {
        stored.refresh_token = rt;
    }
    if let Some(sc) = token_data.scope.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        stored.scopes = sc.to_string();
    }
    storage.save_auth(stored)?;
    Ok(stored.access_token.clone())
}

pub async fn get_valid_access_token(
    storage: &StorageManager,
    config: &AppConfig,
) -> Result<String, String> {
    let mut stored = storage.load_auth().unwrap_or_default();
    if stored.access_token.is_empty() && stored.refresh_token.is_empty() {
        return Err("Google account is not linked.".into());
    }
    let now = Utc::now().timestamp();
    if !stored.access_token.is_empty() && stored.expires_at_unix > now + 300 {
        return Ok(stored.access_token);
    }
    refresh_access_token_now(storage, config, &mut stored).await
}

pub async fn force_refresh_colab_token(
    storage: &StorageManager,
    config: &AppConfig,
) -> Result<String, String> {
    let mut stored = storage.load_auth().unwrap_or_default();
    if stored.access_token.is_empty() && stored.refresh_token.is_empty() {
        return Err("Google account is not linked.".into());
    }
    let token = refresh_access_token_now(storage, config, &mut stored).await?;
    if !has_colab_scope(&stored.scopes) {
        return Err(
            "Google session lacks Colab API scope (colaboratory). Disconnect and sign in again in Settings."
                .into(),
        );
    }
    Ok(token)
}

pub async fn get_valid_colab_token(
    storage: &StorageManager,
    config: &AppConfig,
) -> Result<String, String> {
    let token = get_valid_access_token(storage, config).await?;
    let stored = storage.load_auth().unwrap_or_default();
    if !has_colab_scope(&stored.scopes) {
        return Err(
            "Google session lacks Colab API scope (colaboratory). Disconnect and sign in again in Settings."
                .into(),
        );
    }
    Ok(token)
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_pkce_and_auth_url() {
        let (verifier, challenge) = generate_pkce_pair();
        assert_eq!(verifier.len(), 64);
        assert!(!challenge.is_empty());
        assert!(!challenge.contains('='));

        let mut cfg = AppConfig::default();
        assert_eq!(resolve_gcp_project_id(&cfg), default_gcp_project_id());
        assert!(!default_oauth_client_id().is_empty());
        assert!(!default_oauth_client_secret().is_empty());
        let default_scopes = resolve_oauth_scopes(&cfg);
        assert!(default_scopes.contains(SCOPE_COLAB));
        assert!(default_scopes.contains(SCOPE_USERINFO_EMAIL));
        assert!(!default_scopes.contains(SCOPE_DRIVE_FILE));

        cfg.oauth_client_id = "1234567890-customclient.apps.googleusercontent.com".into();
        assert_eq!(resolve_gcp_project_id(&cfg), "1234567890");
        cfg.enable_drive_backup = true;
        cfg.gcp_project_id = "999888777".into();
        assert_eq!(resolve_gcp_project_id(&cfg), "999888777");
        let backup_scopes = resolve_oauth_scopes(&cfg);
        assert!(backup_scopes.contains(SCOPE_DRIVE_FILE));

        let url = build_google_auth_url(
            "12345.apps.googleusercontent.com",
            "http://127.0.0.1:54321/callback",
            "state_abc",
            &challenge,
            &default_scopes,
        )
        .unwrap();
        assert!(url.contains("code_challenge_method=S256"));
        assert!(url.contains("access_type=offline"));
        assert!(url.contains("12345.apps.googleusercontent.com"));
        assert!(url.contains("colaboratory"));
        assert!(url.contains("userinfo.email"));
        assert!(!url.contains("drive.file"));
        assert!(url.contains("%2Fcallback"));
    }

    async fn spawn_mock_http_endpoint(status: &str, json_body: String) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let status_owned = status.to_string();
        tokio::spawn(async move {
            if let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = [0u8; 4096];
                let _ = sock.read(&mut buf).await;
                let reply = format!(
                    "HTTP/1.1 {status_owned}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    json_body.len(),
                    json_body
                );
                let _ = sock.write_all(reply.as_bytes()).await;
                let _ = sock.shutdown().await;
            }
        });
        format!("http://127.0.0.1:{port}")
    }

    async fn hit_callback(port: u16, query: &str) -> String {
        let mut stream = TcpStream::connect(format!("127.0.0.1:{port}"))
            .await
            .unwrap();
        let req = format!(
            "GET /callback{query} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
        );
        stream.write_all(req.as_bytes()).await.unwrap();
        let mut resp = Vec::new();
        let _ = stream.read_to_end(&mut resp).await;
        String::from_utf8_lossy(&resp).to_string()
    }

    #[tokio::test]
    async fn test_verified_loopback_oauth_happy_and_error_paths() {
        let tmp_ok = std::env::temp_dir().join(format!("damnisium_oauth_ok_{}", Uuid::new_v4()));
        let mgr_ok = StorageManager::from_data_dir(tmp_ok.clone()).unwrap();
        let token_url = spawn_mock_http_endpoint(
            "200 OK",
            r#"{"access_token":"ya29.test_access","refresh_token":"1//test_refresh","expires_in":3600}"#.into(),
        )
        .await;
        let user_url = spawn_mock_http_endpoint(
            "200 OK",
            r#"{"sub":"100200300","email":"tej@example.com","name":"Tej","picture":""}"#.into(),
        )
        .await;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let rd = format!("http://127.0.0.1:{port}/callback");
        let st_clone = mgr_ok.clone();
        let server_task = tokio::spawn(async move {
            run_oauth_callback_once(
                listener,
                &st_clone,
                "cid",
                "csec",
                &rd,
                "verifier123",
                "state_ok",
                "https://www.googleapis.com/auth/colaboratory https://www.googleapis.com/auth/userinfo.email",
                &token_url,
                &user_url,
                |_| {},
            )
            .await
        });

        let browser_http = hit_callback(port, "?code=auth_code_1&state=state_ok").await;
        assert!(browser_http.starts_with("HTTP/1.1 200 OK"));
        assert!(browser_http.contains("Authentication complete"));
        let res = server_task.await.unwrap().unwrap();
        assert!(res.connected);
        assert!(res.colab_authorized);
        assert_eq!(res.user.unwrap().email, "tej@example.com");
        let saved = mgr_ok.load_auth().unwrap();
        assert_eq!(saved.access_token, "ya29.test_access");
        assert!(has_colab_scope(&saved.scopes));
        assert_eq!(saved.user.unwrap().sub, "100200300");
        let _ = std::fs::remove_dir_all(&tmp_ok);

        let tmp_bad_state =
            std::env::temp_dir().join(format!("damnisium_oauth_st_{}", Uuid::new_v4()));
        let mgr_bad = StorageManager::from_data_dir(tmp_bad_state.clone()).unwrap();
        let listener2 = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port2 = listener2.local_addr().unwrap().port();
        let rd2 = format!("http://127.0.0.1:{port2}/callback");
        let mgr_bad_c = mgr_bad.clone();
        let bad_task = tokio::spawn(async move {
            run_oauth_callback_once(
                listener2,
                &mgr_bad_c,
                "cid",
                "csec",
                &rd2,
                "verifier123",
                "expected_state",
                "https://www.googleapis.com/auth/colaboratory https://www.googleapis.com/auth/userinfo.email",
                "http://127.0.0.1:1",
                "http://127.0.0.1:1",
                |_| {},
            )
            .await
        });
        let bad_http = hit_callback(port2, "?code=abc&state=wrong_state").await;
        assert!(bad_http.starts_with("HTTP/1.1 400 Bad Request"));
        assert!(bad_http.contains("Authentication failed"));
        assert!(bad_task.await.unwrap().is_err());
        assert!(!tmp_bad_state.join("auth.json").exists());
        assert!(mgr_bad.load_auth().unwrap().access_token.is_empty());
        let _ = std::fs::remove_dir_all(&tmp_bad_state);

        let tmp_ui_err =
            std::env::temp_dir().join(format!("damnisium_oauth_ui_{}", Uuid::new_v4()));
        let mgr_ui = StorageManager::from_data_dir(tmp_ui_err.clone()).unwrap();
        let tok_ok = spawn_mock_http_endpoint(
            "200 OK",
            r#"{"access_token":"ya29.abc","expires_in":3600}"#.into(),
        )
        .await;
        let ui_401 =
            spawn_mock_http_endpoint("401 Unauthorized", r#"{"error":"invalid"}"#.into()).await;
        let listener3 = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port3 = listener3.local_addr().unwrap().port();
        let rd3 = format!("http://127.0.0.1:{port3}/callback");
        let mgr_ui_c = mgr_ui.clone();
        let ui_task = tokio::spawn(async move {
            run_oauth_callback_once(
                listener3,
                &mgr_ui_c,
                "cid",
                "csec",
                &rd3,
                "verifier123",
                "s3",
                "https://www.googleapis.com/auth/colaboratory https://www.googleapis.com/auth/userinfo.email",
                &tok_ok,
                &ui_401,
                |_| {},
            )
            .await
        });
        let ui_http = hit_callback(port3, "?code=abc&state=s3").await;
        assert!(ui_http.starts_with("HTTP/1.1 400 Bad Request"));
        assert!(ui_http.contains("Authentication failed"));
        assert!(ui_task.await.unwrap().is_err());
        assert!(!tmp_ui_err.join("auth.json").exists());
        assert!(mgr_ui.load_auth().unwrap().access_token.is_empty());
        let _ = std::fs::remove_dir_all(&tmp_ui_err);
    }
}
