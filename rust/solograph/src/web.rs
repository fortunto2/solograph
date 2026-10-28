//! web_fetch and the YouTube id parser, as server.py had them.

use std::sync::LazyLock;
use std::time::Duration;

use regex::Regex;
use serde_json::{Value, json};

const USER_AGENTS: [&str; 5] = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:134.0) Gecko/20100101 Firefox/134.0",
];

static SCRIPT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?is)<script[^>]*>.*?</script>").expect("static regex"));
static STYLE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?is)<style[^>]*>.*?</style>").expect("static regex"));
static TAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]+>").expect("static regex"));
static SPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").expect("static regex"));
static YT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})").expect("static regex")
});

/// A video URL's 11-character id, or the input when it already is one.
pub fn youtube_id(url: &str) -> String {
    YT.captures(url)
        .map_or_else(|| url.to_string(), |c| c[1].to_string())
}

pub fn strip_html(html: &str) -> String {
    let text = SCRIPT.replace_all(html, " ");
    let text = STYLE.replace_all(&text, " ");
    let text = TAG.replace_all(&text, " ");
    SPACE.replace_all(&text, " ").trim().to_string()
}

pub async fn fetch(url: &str, max_length: usize, extract_text: bool, timeout: u64) -> Value {
    let agent = USER_AGENTS[std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos() as usize)
        % USER_AGENTS.len()];
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout))
        .redirect(reqwest::redirect::Policy::limited(10))
        // A login or consent page sets a cookie and redirects; without a jar the
        // redirect loops until the limit (review, 28 Oct 2026).
        .cookie_store(true)
        .build()
    {
        Ok(c) => c,
        Err(e) => return json!({"url": url, "error": e.to_string(), "status": 0}),
    };
    let req = client
        .get(url)
        .header("User-Agent", agent)
        .header(
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        )
        .header("Accept-Language", "en-US,en;q=0.9")
        .header("DNT", "1")
        .header("Upgrade-Insecure-Requests", "1")
        .header("Sec-Fetch-Dest", "document")
        .header("Sec-Fetch-Mode", "navigate")
        .header("Sec-Fetch-Site", "none")
        .header("Sec-Fetch-User", "?1")
        .header("Cache-Control", "max-age=0");
    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) if e.is_timeout() => return json!({"url": url, "error": "Timeout", "status": 0}),
        Err(e) if e.is_connect() => {
            return json!({"url": url, "error": format!("Connection failed: {e}"), "status": 0});
        }
        Err(e) => return json!({"url": url, "error": e.to_string(), "status": 0}),
    };
    let final_url = resp.url().to_string();
    let status = resp.status().as_u16();
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let body = match resp.text().await {
        Ok(t) => t,
        Err(e) => return json!({"url": url, "error": e.to_string(), "status": 0}),
    };
    if status >= 400 {
        let head: String = body.chars().take(500).collect();
        return json!({"url": final_url, "status": status, "error": format!("HTTP {status}"), "content": head});
    }
    let content = if extract_text {
        strip_html(&body)
    } else {
        body
    };
    let total = content.chars().count();
    json!({
        "url": final_url,
        "status": status,
        "content_type": content_type,
        "content": content.chars().take(max_length).collect::<String>(),
        "truncated": total > max_length,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn youtube_ids_and_html() {
        assert_eq!(
            youtube_id("https://www.youtube.com/watch?v=EsTrWCV0Ph4&t=796s"),
            "EsTrWCV0Ph4"
        );
        assert_eq!(youtube_id("https://youtu.be/EsTrWCV0Ph4"), "EsTrWCV0Ph4");
        assert_eq!(youtube_id("EsTrWCV0Ph4"), "EsTrWCV0Ph4");
        assert_eq!(strip_html("<p>a</p><script>x()</script>\n<b>b</b>"), "a b");
    }
}

#[cfg(test)]
mod fetch_tests {
    use super::fetch;

    /// A local server: `/1251` answers in windows-1251, `/login` sets a cookie and
    /// redirects to `/home`, which answers only with that cookie.
    fn serve() -> String {
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let addr = server.server_addr().to_ip().unwrap();
        std::thread::spawn(move || {
            for req in server.incoming_requests().take(4) {
                let header = |k: &str, v: &str| tiny_http::Header::from_bytes(k, v).unwrap();
                let cookie = req
                    .headers()
                    .iter()
                    .any(|h| h.field.equiv("Cookie") && h.value.as_str().contains("s=1"));
                let resp = match req.url() {
                    // "Привет" in windows-1251
                    "/1251" => {
                        tiny_http::Response::from_data(vec![0xcf, 0xf0, 0xe8, 0xe2, 0xe5, 0xf2])
                            .with_header(header("Content-Type", "text/plain; charset=windows-1251"))
                    }
                    "/login" => tiny_http::Response::from_data(Vec::new())
                        .with_status_code(302)
                        .with_header(header("Set-Cookie", "s=1; Path=/"))
                        .with_header(header("Location", "/home")),
                    _ if cookie => tiny_http::Response::from_data(b"welcome".to_vec()),
                    _ => tiny_http::Response::from_data(Vec::new())
                        .with_status_code(302)
                        .with_header(header("Location", "/login")),
                };
                let _ = req.respond(resp);
            }
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn a_page_in_another_charset_is_decoded_and_a_cookie_survives_a_redirect() {
        // Review 28 Oct: without reqwest's charset and cookies features the body was
        // always read as UTF-8 and Set-Cookie was dropped between redirects.
        let base = serve();
        let page = fetch(&format!("{base}/1251"), 1000, false, 5).await;
        assert_eq!(page["content"], "Привет", "{page}");
        let home = fetch(&format!("{base}/login"), 1000, false, 5).await;
        assert_eq!(home["content"], "welcome", "{home}");
    }
}
