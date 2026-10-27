//! `solograph http`: the sources index over HTTP, for a SearXNG engine
//! (`sources_local` in searxng-docker-tavily-adapter). Same endpoints, port and
//! JSON as the Python search_server.py it replaces:
//!
//!   GET /search?q=...&source=producthunt&n=5  — full-text search
//!   GET /sources                               — indexed sources with counts
//!   GET /health                                — status + total count

use anyhow::{Result, anyhow};
use serde_json::{Value, json};

use crate::sources::SourceIndex;

pub fn port() -> u16 {
    std::env::var("SOLOGRAPH_SEARCH_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8002)
}

/// Percent-decoding as Python's parse_qs does it: `+` is a space, `%XX` a byte.
fn decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                match u8::from_str_radix(hex, 16) {
                    Ok(b) => {
                        out.push(b);
                        i += 2;
                    }
                    Err(_) => out.push(b'%'),
                }
            }
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// First non-blank value of each key, like parse_qs()[k][0] (blank values dropped).
fn param(query: &str, key: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (decode(k) == key && !v.is_empty()).then(|| decode(v))
    })
}

/// One request: status and JSON body. Kept apart from the socket so it can be tested.
pub fn handle(idx: &SourceIndex, url: &str) -> (u16, Value) {
    let (path, query) = url.split_once('?').unwrap_or((url, ""));
    let result = match path {
        "/search" => search(idx, query),
        "/sources" => idx.list_sources().map(|s| (200, json!({"sources": s}))),
        "/health" => idx.count().and_then(|total| {
            Ok((
                200,
                json!({"status": "ok", "total": total, "sources": idx.list_sources()?.len()}),
            ))
        }),
        _ => Ok((404, json!({"error": "not found"}))),
    };
    result.unwrap_or_else(|e| (500, json!({"error": format!("{e:#}")})))
}

fn search(idx: &SourceIndex, query: &str) -> Result<(u16, Value)> {
    let Some(q) = param(query, "q") else {
        return Ok((400, json!({"error": "missing q param"})));
    };
    let n = match param(query, "n").map(|n| n.parse::<usize>()) {
        None => 5,
        Some(Ok(n)) => n,
        Some(Err(_)) => return Ok((400, json!({"error": "n must be a number"}))),
    };
    let source = param(query, "source");
    if let Some(src) = &source {
        let available: Vec<String> = idx
            .list_sources()?
            .iter()
            .filter_map(|s| s["source"].as_str().map(String::from))
            .collect();
        if !available.contains(src) {
            return Ok((
                400,
                json!({"error": format!("unknown source: {src}"), "available": available}),
            ));
        }
    }
    let results = idx.search(&q, source.as_deref(), n)?;
    Ok((
        200,
        json!({"results": results, "query": q, "source": source}),
    ))
}

pub fn serve(port: u16) -> Result<()> {
    let idx = SourceIndex::open(None)?;
    let sources = idx.list_sources()?;
    let names: Vec<&str> = sources
        .iter()
        .filter_map(|s| s["source"].as_str())
        .collect();
    println!(
        "SourceIndex ready: {} items across {} sources: {}",
        idx.count()?,
        sources.len(),
        names.join(", ")
    );
    let server = tiny_http::Server::http(("0.0.0.0", port)).map_err(|e| anyhow!("{e}"))?;
    println!("Solograph search API on :{port}");
    println!("  GET /search?q=AI+tool&source=producthunt&n=5");
    println!("  GET /sources");
    println!("  GET /health");
    let content_type =
        tiny_http::Header::from_bytes("Content-Type", "application/json").expect("static header");
    for req in server.incoming_requests() {
        let (status, body) = handle(&idx, req.url());
        let resp = tiny_http::Response::from_string(body.to_string())
            .with_status_code(status)
            .with_header(content_type.clone());
        let _ = req.respond(resp);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
