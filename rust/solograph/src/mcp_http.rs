//! `solograph mcp-http`: the same MCP server as `serve`, over streamable HTTP, for an agent
//! that cannot start a local process: Claude Code or Codex on another machine, later the phone.
//!
//! It listens on 127.0.0.1 only. Reaching it from outside is the tunnel's job (Cloudflare
//! Access in front), and every request must also carry `Authorization: Bearer <token>`, so a
//! misconfigured tunnel still does not open the user's sessions to the internet. Without a
//! token it refuses to start.
//!
//!   SOLOGRAPH_MCP_TOKEN or SOLOGRAPH_MCP_TOKEN_FILE   the bearer token (32+ characters)
//!   SOLOGRAPH_MCP_HOSTS   public host names the tunnel forwards (the Host header rmcp checks)
//!   SOLOGRAPH_MCP_PORT    default 8014

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use axum::{
    Router,
    extract::Request,
    http::{StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
};
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};

pub fn port() -> u16 {
    std::env::var("SOLOGRAPH_MCP_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8014)
}

/// The token from the environment or a file; short or missing tokens are refused.
pub fn token() -> Result<String> {
    let raw = match std::env::var("SOLOGRAPH_MCP_TOKEN") {
        Ok(t) => t,
        Err(_) => {
            let path = std::env::var("SOLOGRAPH_MCP_TOKEN_FILE")
                .context("set SOLOGRAPH_MCP_TOKEN or SOLOGRAPH_MCP_TOKEN_FILE")?;
            std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?
        }
    };
    let t = raw.trim().to_string();
    if t.len() < 32 {
        bail!("the token must be at least 32 characters");
    }
    Ok(t)
}

/// Loopback names, plus the public ones the tunnel forwards.
pub fn hosts() -> Vec<String> {
    let mut h: Vec<String> = ["127.0.0.1", "localhost"].map(String::from).to_vec();
    if let Ok(extra) = std::env::var("SOLOGRAPH_MCP_HOSTS") {
        h.extend(
            extra
                .split([',', ' '])
                .filter(|s| !s.is_empty())
                .map(String::from),
        );
    }
    h
}

/// Same answer in the same time whether the first or the last byte differs.
fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn authorized(req: &Request, token: &str) -> bool {
    req.headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|got| same(got.trim().as_bytes(), token.as_bytes()))
}

pub async fn serve(port: u16) -> Result<()> {
    let token: Arc<str> = token()?.into();
    let hosts = hosts();
    let config = StreamableHttpServerConfig::default().with_allowed_hosts(hosts.clone());
    let handler = crate::server::Solograph::new();
    let mcp = StreamableHttpService::new(
        move || Ok(handler.clone()),
        LocalSessionManager::default().into(),
        config,
    );
    let gate = middleware::from_fn(move |req: Request, next: Next| {
        let token = token.clone();
        async move {
            if authorized(&req, &token) {
                next.run(req).await
            } else {
                unauthorized()
            }
        }
    });
    let app = Router::new().nest_service("/mcp", mcp).layer(gate);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
        .await
        .with_context(|| format!("bind 127.0.0.1:{port}"))?;
    eprintln!(
        "solograph mcp-http on http://127.0.0.1:{port}/mcp (hosts: {})",
        hosts.join(", ")
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        [(header::WWW_AUTHENTICATE, "Bearer")],
        "bearer token required\n",
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(auth: Option<&str>) -> Request {
        let mut b = Request::builder().uri("/mcp");
        if let Some(a) = auth {
            b = b.header(header::AUTHORIZATION, a);
        }
        b.body(axum::body::Body::empty()).unwrap()
    }

    #[test]
    fn only_the_exact_bearer_token_passes() {
        let t = "x".repeat(40);
        assert!(authorized(&req(Some(&format!("Bearer {t}"))), &t));
        assert!(!authorized(&req(None), &t));
        assert!(!authorized(&req(Some(&t)), &t), "no scheme");
        assert!(!authorized(&req(Some(&format!("Bearer {t}y"))), &t));
        assert!(!authorized(&req(Some("Bearer ")), &t));
        assert!(!authorized(&req(Some(&format!("Basic {t}"))), &t));
    }

    #[test]
    fn compare_is_exact() {
        assert!(same(b"abc", b"abc"));
        assert!(!same(b"abc", b"abd"));
        assert!(!same(b"abc", b"ab"));
    }
}
