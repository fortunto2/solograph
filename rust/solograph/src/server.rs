//! The MCP server: the 12 tools server.py exposes, same names, arguments and
//! result shapes. A list comes back as one JSON text block per item plus
//! `structuredContent: {"result": [...]}`, a dict as one JSON text block — the
//! way Python's MCP server serialises them.

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use anyhow::Result;
use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerConfig};
use rmcp::{ErrorData, ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::kb::KnowledgeBase;
use crate::paths::{env_path, home};
use crate::portfolio::Portfolio;
use crate::sessions::SessionSearch;
use crate::sources::SourceIndex;

fn five() -> usize {
    5
}
fn twenty() -> usize {
    20
}
fn youtube() -> String {
    "youtube".into()
}

#[derive(Deserialize, JsonSchema)]
pub struct KbSearchArgs {
    /// Search words (e.g. "privacy architecture", "агент память")
    query: String,
    /// Number of results (default 5)
    #[serde(default = "five")]
    n_results: usize,
    /// Filter by frontmatter `type` (substring match)
    #[serde(default)]
    doc_type: Option<String>,
    /// Add knowledge-graph neighbours of the top hits (1-hop)
    #[serde(default)]
    expand_graph: bool,
}

#[derive(Deserialize, JsonSchema)]
pub struct SessionSearchArgs {
    /// Search words (e.g. "hubspot webhook", "транскрипция midi")
    query: String,
    /// Number of results (default 5)
    #[serde(default = "five")]
    n_results: usize,
    /// Filter by project name, substring (e.g. "life2film")
    #[serde(default)]
    project: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
pub struct SymbolSearchArgs {
    /// Words or an identifier (e.g. "auth middleware", "MontageProfile")
    query: String,
    /// Limit to one project (exact registry name)
    #[serde(default)]
    project: Option<String>,
    /// "function", "class" or "protocol"
    #[serde(default)]
    kind: Option<String>,
    /// Number of results (default 20)
    #[serde(default = "twenty")]
    n_results: usize,
}

#[derive(Deserialize, JsonSchema)]
pub struct ProjectArgs {
    /// Project name (e.g. "my-app")
    project: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct SharedArgs {
    /// Optional package name or part of one (e.g. "react", "supabase")
    #[serde(default)]
    package: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
pub struct ProjectInfoArgs {
    /// Project name (e.g. "my-app"). Omit for full list.
    #[serde(default)]
    name: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
pub struct WebFetchArgs {
    /// URL to fetch
    url: String,
    /// Max content length in chars (default 20000)
    #[serde(default = "web_max")]
    max_length: usize,
    /// Strip HTML tags and return plain text (default true)
    #[serde(default = "yes")]
    extract_text: bool,
    /// Request timeout in seconds (default 30)
    #[serde(default = "web_timeout")]
    timeout: u64,
}
fn web_max() -> usize {
    20000
}
fn yes() -> bool {
    true
}
fn web_timeout() -> u64 {
    30
}

#[derive(Deserialize, JsonSchema)]
pub struct SourceSearchArgs {
    /// Search query (e.g. "startup idea", "revenue growth")
    query: String,
    /// Filter by source name (e.g. "telegram", "youtube"). Omit to search all.
    #[serde(default)]
    source: Option<String>,
    /// Number of results (default 5)
    #[serde(default = "five")]
    n_results: usize,
}

#[derive(Deserialize, JsonSchema)]
pub struct SourceArgs {
    /// Source name (default: "youtube")
    #[serde(default = "youtube")]
    source: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct RelatedArgs {
    /// YouTube video URL or video ID
    video_url: String,
    /// Source name (default: "youtube")
    #[serde(default = "youtube")]
    source: String,
}

#[derive(Default)]
struct Stores {
    kb: Option<KnowledgeBase>,
    sessions: Option<SessionSearch>,
    sources: Option<SourceIndex>,
    portfolio: Option<Portfolio>,
}

#[derive(Clone)]
pub struct Solograph {
    stores: std::sync::Arc<Mutex<Stores>>,
    tool_router: ToolRouter<Self>,
}

fn pretty(v: &Value) -> String {
    serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string())
}

fn list_result(items: Vec<Value>) -> CallToolResult {
    let mut r = CallToolResult::success(
        items
            .iter()
            .map(|v| ContentBlock::text(pretty(v)))
            .collect(),
    );
    r.structured_content = Some(json!({"result": items}));
    r
}

fn dict_result(v: Value) -> CallToolResult {
    CallToolResult::success(vec![ContentBlock::text(pretty(&v))])
}

fn failed(tool: &str, e: anyhow::Error) -> CallToolResult {
    CallToolResult::error(vec![ContentBlock::text(format!(
        "Error executing tool {tool}: {e:#}"
    ))])
}

/// KB_PATH, else the working directory when it looks like a knowledge base.
fn detect_kb_path() -> Option<PathBuf> {
    if let Some(p) = env_path("KB_PATH") {
        return Some(p);
    }
    let cwd = std::env::current_dir().ok()?;
    if cwd.join(".solo").join("kb").exists() {
        return Some(cwd);
    }
    let mut checked = 0;
    for dir in std::fs::read_dir(&cwd)
        .ok()?
        .flatten()
        .filter(|e| e.path().is_dir())
    {
        for f in std::fs::read_dir(dir.path())
            .into_iter()
            .flatten()
            .flatten()
        {
            if checked >= 5 {
                return None;
            }
            if f.path().extension().is_some_and(|x| x == "md") {
                checked += 1;
                let head = std::fs::read(f.path()).unwrap_or_default();
                if head.starts_with(b"---") {
                    return Some(cwd);
                }
            }
        }
    }
    None
}

fn registry_path() -> Option<PathBuf> {
    match env_path("CODEGRAPH_REGISTRY") {
        Some(p) => p.exists().then_some(p),
        None => Some(home().join(".solo").join("registry.yaml")).filter(|p| p.exists()),
    }
}

impl Solograph {
    pub fn new() -> Self {
        Self {
            stores: Default::default(),
            tool_router: Self::tool_router(),
        }
    }

    fn stores(&self) -> MutexGuard<'_, Stores> {
        self.stores.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn project_info_value(name: Option<&str>) -> Result<Value> {
        let Some(path) = registry_path() else {
            return Ok(
                json!({"error": "Registry not found. Set CODEGRAPH_REGISTRY env var or run: solograph scan"}),
            );
        };
        let data: serde_yaml::Value = serde_yaml::from_str(&std::fs::read_to_string(path)?)?;
        let data = crate::yamlval::to_json(&data);
        let projects = data["projects"].as_array().cloned().unwrap_or_default();
        let Some(name) = name else {
            let field = |p: &Value, k: &str, d: Value| p.get(k).cloned().unwrap_or(d);
            return Ok(Value::Array(
                projects
                    .iter()
                    .map(|p| {
                        json!({"name": p["name"], "stacks": field(p, "stacks", json!([])),
                               "status": field(p, "status", json!("active")), "last_commit": field(p, "last_commit", json!(""))})
                    })
                    .collect(),
            ));
        };
        let lower = name.to_lowercase();
        Ok(projects
            .into_iter()
            .find(|p| {
                p["name"]
                    .as_str()
                    .is_some_and(|n| n.to_lowercase() == lower)
            })
            .unwrap_or_else(|| json!({"error": format!("Project '{name}' not found in registry")})))
    }
}

impl Default for Solograph {
    fn default() -> Self {
        Self::new()
    }
}

#[tool_router]
impl Solograph {
    #[tool(
        description = "Full-text search over the knowledge base (markdown with YAML frontmatter).\n\nBM25 ranking with Russian and English stemming: \"графами\" finds \"граф\".\nMatching is lexical, so use the words the document would use, in its\nlanguage. New and edited files are picked up on every call. `score` is\nBM25: higher is better, comparable only within one answer."
    )]
    async fn kb_search(
        &self,
        Parameters(a): Parameters<KbSearchArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let mut st = self.stores();
        if st.kb.is_none() {
            let Some(root) = detect_kb_path() else {
                return Ok(list_result(vec![
                    json!({"error": "KB not configured. Set KB_PATH env var."}),
                ]));
            };
            match KnowledgeBase::open(&root, None) {
                Ok(kb) => st.kb = Some(kb),
                Err(e) => return Ok(failed("kb_search", e)),
            }
        }
        let kb = st.kb.as_mut().expect("opened above");
        Ok(
            match kb.search(&a.query, a.n_results, a.doc_type.as_deref(), a.expand_graph) {
                Ok(v) => list_result(v),
                Err(e) => failed("kb_search", e),
            },
        )
    }

    #[tool(
        description = "Full-text search over Claude Code session history — what was asked and answered.\n\nFor \"how did I solve X?\", \"when did I work on Y?\", \"what did we decide about Z?\".\nSearches every prompt and every answer, BM25-ranked, Russian and English\nstemming. Matching is lexical: use the words the conversation would have\nused, and try the other language if the first returns nothing — sessions\nare mostly in Russian. Includes sessions Claude Code has already deleted\nfrom disk (it keeps transcripts 30 days); those come back marked\n`transcript: deleted`. Live ones carry a `resume` command."
    )]
    async fn session_search(
        &self,
        Parameters(a): Parameters<SessionSearchArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let mut st = self.stores();
        if st.sessions.is_none() {
            match SessionSearch::open(None, None) {
                Ok(s) => st.sessions = Some(s),
                Err(e) => return Ok(failed("session_search", e)),
            }
        }
        let s = st.sessions.as_mut().expect("opened above");
        Ok(
            match s.search(&a.query, a.n_results, a.project.as_deref()) {
                Ok(v) => dict_result(v),
                Err(e) => failed("session_search", e),
            },
        )
    }

    #[tool(
        description = "Find where something is defined, across every project in the portfolio.\n\nSearches function, class and protocol names by their words: \"montage profile\"\nfinds resolve_montage_profile and MontageProfile. All words rank above some\nwords; an exact name comes first. Returns project, file:line and kind.\nInside one repo, Grep answers more (callers, usages)."
    )]
    async fn symbol_search(
        &self,
        Parameters(a): Parameters<SymbolSearchArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        Ok(
            match self.with_portfolio(|p| {
                p.search_symbols(
                    &a.query,
                    a.n_results,
                    a.project.as_deref(),
                    a.kind.as_deref(),
                )
            }) {
                Ok(v) => list_result(v),
                Err(e) => failed("symbol_search", e),
            },
        )
    }

    #[tool(
        description = "Counts across the portfolio: projects, scanned projects, files, symbols by kind, packages."
    )]
    async fn codegraph_stats(&self) -> Result<CallToolResult, ErrorData> {
        Ok(match self.with_portfolio(|p| p.stats()) {
            Ok(v) => dict_result(v),
            Err(e) => failed("codegraph_stats", e),
        })
    }

    #[tool(
        description = "What one project is made of: stack, languages, directory layers, patterns,\ndependencies and the files with the most definitions."
    )]
    async fn codegraph_explain(
        &self,
        Parameters(a): Parameters<ProjectArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        Ok(match self.with_portfolio(|p| p.explain(&a.project)) {
            Ok(Ok(v)) => dict_result(v),
            Ok(Err(msg)) => dict_result(json!({"error": msg})),
            Err(e) => failed("codegraph_explain", e),
        })
    }

    #[tool(
        description = "Packages used by more than one project, most shared first — or, given a\nname, every project that uses a package matching it."
    )]
    async fn codegraph_shared(
        &self,
        Parameters(a): Parameters<SharedArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        Ok(
            match self.with_portfolio(|p| p.shared_packages(a.package.as_deref())) {
                Ok(v) => list_result(v),
                Err(e) => failed("codegraph_shared", e),
            },
        )
    }

    #[tool(
        description = "Get project information from the registry.\n\nWithout name: returns list of all active projects with stacks.\nWith name: returns detailed info for one project."
    )]
    async fn project_info(
        &self,
        Parameters(a): Parameters<ProjectInfoArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        Ok(match Self::project_info_value(a.name.as_deref()) {
            Ok(Value::Array(items)) => list_result(items),
            Ok(v) => dict_result(v),
            Err(e) => failed("project_info", e),
        })
    }

    #[tool(
        description = "Fetch a URL with browser-like headers to bypass basic bot protection.\n\nReturns page content as plain text (HTML tags stripped) or raw HTML.\nHandles redirects, cookies, and common anti-bot headers automatically."
    )]
    async fn web_fetch(
        &self,
        Parameters(a): Parameters<WebFetchArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        Ok(dict_result(
            crate::web::fetch(&a.url, a.max_length, a.extract_text, a.timeout).await,
        ))
    }

    #[tool(
        description = "Full-text search over indexed external sources (YouTube, Telegram, ProductHunt, TrustMRR).\n\nBM25 with Russian and English stemming; matching is lexical, so use the\nwords the content would use. YouTube hits are the best transcript chunk per\nvideo, with chapter, timecode and a URL that opens at that moment. `score`\nis BM25 (ProductHunt upvotes add up to 15%): higher is better."
    )]
    async fn source_search(
        &self,
        Parameters(a): Parameters<SourceSearchArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        Ok(
            match self.with_sources(|s| s.search(&a.query, a.source.as_deref(), a.n_results)) {
                Ok(v) => list_result(v),
                Err(e) => failed("source_search", e),
            },
        )
    }

    #[tool(
        description = "List all auto-detected topics with video counts.\n\nTags are assigned at indexing from topic keywords in the title and the\nstart of the description. Shared across videos — enables topic clustering."
    )]
    async fn source_tags(
        &self,
        Parameters(a): Parameters<SourceArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        Ok(match self.with_sources(|s| s.list_tags(&a.source)) {
            Ok(v) => list_result(v),
            Err(e) => failed("source_tags", e),
        })
    }

    #[tool(
        description = "Find related videos by shared tags.\n\nGiven a video URL, finds other videos that share auto-detected topics.\nReturns videos sorted by number of shared tags (most related first)."
    )]
    async fn source_related(
        &self,
        Parameters(a): Parameters<RelatedArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let video_id = crate::web::youtube_id(&a.video_url);
        Ok(
            match self.with_sources(|s| s.related_videos(&a.source, &video_id)) {
                Ok(v) => list_result(v),
                Err(e) => failed("source_related", e),
            },
        )
    }

    #[tool(
        description = "List indexed external sources with document counts.\n\nOne index at ~/.solo/sources/search.db. YouTube sources include the\nvideo/chunk breakdown (videos, video_chunks fields)."
    )]
    async fn source_list(&self) -> Result<CallToolResult, ErrorData> {
        Ok(match self.with_sources(|s| s.list_sources()) {
            Ok(v) => list_result(v),
            Err(e) => failed("source_list", e),
        })
    }
}

impl Solograph {
    fn with_portfolio<T>(&self, f: impl FnOnce(&mut Portfolio) -> Result<T>) -> Result<T> {
        let mut st = self.stores();
        if st.portfolio.is_none() {
            st.portfolio = Some(Portfolio::open(None)?);
        }
        f(st.portfolio.as_mut().expect("opened above"))
    }

    fn with_sources<T>(&self, f: impl FnOnce(&SourceIndex) -> Result<T>) -> Result<T> {
        let mut st = self.stores();
        if st.sources.is_none() {
            st.sources = Some(SourceIndex::open(None)?);
        }
        f(st.sources.as_ref().expect("opened above"))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for Solograph {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("solograph", env!("CARGO_PKG_VERSION")))
    }
}
