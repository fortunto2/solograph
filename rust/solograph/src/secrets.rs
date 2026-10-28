//! Secrets never enter an index. Every text an index stores (title, summary,
//! body) passes through `redact` first, and the indexes written before this
//! existed are cleaned once on open.
//!
//! Measured on 28 Oct 2026: the session index held 12 AWS access keys, 10
//! `*_TOKEN=` / `*_SECRET=` lines, 5 passwords, 3 Telegram bot tokens, 3 OpenAI
//! keys and an Anthropic key, all searchable, and for deleted transcripts the
//! index is the only copy left anywhere.
//!
//! The rules are solo-factory's secret_guard hook (hooks/secret_guard.py), which
//! masks secrets in tool output and adapted them from ScreenShield's leak
//! detectors: the same rules, names and placeholder filter, so what the guard
//! would hide never sits in an index. One addition: `password: <value>`, under
//! the same filter (5 in the session index). A token is replaced whole; an
//! assignment keeps its name, so `OPENAI_API_KEY=` is still found and only the
//! value is gone. The marker is lowercase on purpose: the guard refuses to write
//! `[REDACTED:` markers into files, and code that handles them has to.

use std::borrow::Cow;
use std::sync::LazyLock;

use regex::Regex;

/// Bump when the rules change: indexes cleaned under an older version are
/// cleaned again on open.
pub const VERSION: i64 = 1;

const TOKEN_RULES: &[(&str, &str)] = &[
    (
        "private-key-block",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|\z)",
    ),
    ("anthropic-key", r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
    ("openai-key", r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}"),
    ("aws-access-key", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    (
        "github-token",
        r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})",
    ),
    ("stripe-live-key", r"\b[sr]k_live_[0-9a-zA-Z]{20,}"),
    ("slack-token", r"\bxox[baprs]-[0-9A-Za-z-]{10,}"),
    ("google-api-key", r"\bAIza[0-9A-Za-z_-]{35}"),
    ("huggingface-token", r"\bhf_[A-Za-z0-9]{34,}"),
    ("telegram-bot-token", r"\b\d{8,10}:AA[A-Za-z0-9_-]{33}\b"),
    (
        "jwt",
        r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
    ),
];

/// Assignments: (rule, pattern with the value as group `v`).
const VALUE_RULES: &[(&str, &str)] = &[
    (
        "env-secret",
        r#"\b[A-Z][A-Z0-9_]*(?:SECRET|TOKEN|API_KEY|APIKEY|PASSWORD|PASSWD|PRIVATE_KEY|ACCESS_KEY)[A-Z0-9_]*\s*[:=]\s*["']?(?P<v>[^\s"'`,;)\[]{12,})"#,
    ),
    (
        "dsn-password",
        r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s:/@]+:(?P<v>[^\s@/]{4,})@",
    ),
    (
        "password",
        r#"(?i)\b(?:password|passwd|pwd)\s*[:=]\s*["']?(?P<v>[^\s"'`,;)\[]{8,})"#,
    ),
];

static PLACEHOLDER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)x{3,}|\*{3,}|your|example|changeme|placeholder|dummy|redacted|<|>|\$|\{|\.\.\.",
    )
    .expect("static regex")
});
static DIGIT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d").expect("static regex"));

struct Rule {
    name: &'static str,
    re: Regex,
}

fn compile(src: &[(&'static str, &str)]) -> Vec<Rule> {
    src.iter()
        .map(|(name, p)| Rule {
            name,
            re: Regex::new(p).expect("static secret rule"),
        })
        .collect()
}

static TOKENS: LazyLock<Vec<Rule>> = LazyLock::new(|| compile(TOKEN_RULES));
static VALUES: LazyLock<Vec<Rule>> = LazyLock::new(|| compile(VALUE_RULES));

/// A value counts when it is not a placeholder and mixes character classes.
fn looks_secret(v: &str) -> bool {
    // A URL is an endpoint (OAUTH_TOKEN_URL), not a secret; credentials inside
    // one are dsn-password's.
    if PLACEHOLDER.is_match(v) || v.contains("EXAMPLE") || v.contains("://") {
        return false;
    }
    let classes = [
        v.chars().any(|c| c.is_ascii_lowercase()),
        v.chars().any(|c| c.is_ascii_uppercase()),
        v.chars().any(|c| c.is_ascii_digit()),
    ];
    classes.iter().filter(|x| **x).count() >= 2
}

/// Every (rule, secret) in `text`, as secret_guard's find() reports them.
fn find(text: &str) -> Vec<(&'static str, String)> {
    let mut out: Vec<(&'static str, String)> = Vec::new();
    for r in TOKENS.iter() {
        for m in r.re.find_iter(text) {
            let v = m.as_str();
            if r.name == "aws-access-key" && v.contains("EXAMPLE") {
                continue;
            }
            if matches!(r.name, "openai-key" | "jwt") && !DIGIT.is_match(v) {
                continue;
            }
            out.push((r.name, v.to_string()));
        }
    }
    for r in VALUES.iter() {
        for c in r.re.captures_iter(text) {
            let v = c.name("v").map_or("", |m| m.as_str());
            let keep = if r.name == "dsn-password" {
                !PLACEHOLDER.is_match(v)
            } else {
                looks_secret(v)
            };
            if keep && !out.iter().any(|(_, o)| o.contains(v)) {
                out.push((r.name, v.to_string()));
            }
        }
    }
    out
}

/// The text with every secret replaced by `[redacted:<rule>]`. Borrowed when
/// nothing matched, which is nearly always.
pub fn redact(text: &str) -> Cow<'_, str> {
    let mut hits = find(text);
    if hits.is_empty() {
        return Cow::Borrowed(text);
    }
    // Longest first, so a key inside a longer match is not half-replaced.
    hits.sort_by_key(|h| std::cmp::Reverse(h.1.len()));
    let mut out = text.to_string();
    for (rule, v) in hits {
        out = out.replace(&v, &format!("[redacted:{rule}]"));
    }
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::redact;

    #[test]
    fn every_token_format_is_redacted_whole() {
        for (text, rule) in [
            ("key AKIAZ7Q3LMN4PQR8STUV here", "aws-access-key"),
            ("ghp_abcdefghijklmnopqrstuvwxyz0123456789AB", "github-token"),
            ("sk_live_abcdefghijklmnopqrstuv", "stripe-live-key"),
            ("xoxb-1234567890-abcdefghij", "slack-token"),
            ("sk-ant-api03-abcdefghijklmnopqrstuvwxyz", "anthropic-key"),
            ("sk-proj-abcdefghijklmnopqrstuvwxyz0123456789", "openai-key"),
            ("AIzaSyA1234567890abcdefghijklmnopqrstuv", "google-api-key"),
            ("hf_abcdefghijklmnopqrstuvwxyz01234567", "huggingface-token"),
            (
                "bot 123456789:AAabcdefghijklmnopqrstuvwxyz0123456 ok",
                "telegram-bot-token",
            ),
            ("eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0.abcdefghijklmnop", "jwt"),
        ] {
            let out = redact(text);
            assert!(out.contains(&format!("[redacted:{rule}]")), "{rule}: {out}");
        }
    }

    #[test]
    fn an_assignment_keeps_its_name_and_loses_its_value() {
        assert_eq!(
            redact("export OPENAI_API_KEY=abcd1234EFGH5678"),
            "export OPENAI_API_KEY=[redacted:env-secret]"
        );
        assert_eq!(
            redact("password: Hunter2hunter"),
            "password: [redacted:password]"
        );
        assert_eq!(
            redact("postgres://admin:s3cretPass@db.example.com:5432/app"),
            "postgres://admin:[redacted:dsn-password]@db.example.com:5432/app"
        );
    }

    #[test]
    fn a_private_key_block_goes_whole() {
        let text = "before\n-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\n-----END OPENSSH PRIVATE KEY-----\nafter";
        assert_eq!(redact(text), "before\n[redacted:private-key-block]\nafter");
    }

    #[test]
    fn placeholders_documentation_keys_and_prose_are_left_alone() {
        for text in [
            "Run the agent in a sandbox; the token budget is 4096.",
            "Set OPENAI_API_KEY in your shell, never in the repo.",
            "OPENAI_API_KEY=your-key-goes-here and API_TOKEN=${{ secrets.TOKEN }}",
            "AWS docs use AKIAIOSFODNN7EXAMPLE as the sample key",
            "password: changeme123 and PASSWORD=xxxxxxxxxxxx",
            "postgres://user:<password>@localhost/db",
            "OAUTH_TOKEN_URL = \"https://api.producthunt.com/v2/oauth/token\"",
            "sk-proj-abcdefghijklmnopqrstuvwxyzabcdefgh has no digit, so it is not a key",
            "песочница для агента на маке",
        ] {
            assert_eq!(redact(text), text);
        }
    }

    #[test]
    fn redaction_is_idempotent() {
        let once =
            redact("API_TOKEN=abcdEFGH12345678 and sk-proj-abcdefghijklmnopqrstuvwxyz0123456789")
                .into_owned();
        assert!(
            once.contains("[redacted:env-secret]") && once.contains("[redacted:openai-key]"),
            "{once}"
        );
        assert_eq!(redact(&once), once);
    }
}
