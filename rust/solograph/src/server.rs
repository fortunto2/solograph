//! The MCP server: five tools, one per question an agent brings (a past
//! conversation, a note, a definition, a talk, a project). Twelve until 0.17: agents
//! called five of them in two months, and with tool search the agent picks from
//! names alone. A list comes back as one JSON text block per item plus
//! `structuredContent: {"result": [...]}`, a dict as one JSON text block.

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use anyhow::Result;
use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerConfig};
use rmcp::{ErrorData, ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::kb::KnowledgeBase;
use crate::paths::{env_path, home};
use crate::portfolio::Portfolio;
use crate::sessions::SessionSearch;
use crate::sources::SourceIndex;

/// Arguments as clients send them, not only as the schema says. Some clients
/// sent numbers and booleans as strings ("10", "true") in 2026, and an empty
/// filter string meant "no filter" to the Python server; strict types turned
/// both into errors or into zero results (review, 28 Oct 2026).
mod lenient {
    use serde::{Deserialize, Deserializer, de::Error};
    use serde_json::Value;

    pub fn usize<'de, D: Deserializer<'de>>(d: D) -> Result<usize, D::Error> {
        match Value::deserialize(d)? {
            Value::Number(n) => n
                .as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| D::Error::custom("not a count")),
            Value::String(s) => s.trim().parse().map_err(D::Error::custom),
            _ => Err(D::Error::custom("expected a number")),
        }
    }

    pub fn bool<'de, D: Deserializer<'de>>(d: D) -> Result<bool, D::Error> {
        match Value::deserialize(d)? {
            Value::Bool(b) => Ok(b),
            Value::String(s) => match s.trim() {
                "true" | "1" | "yes" => Ok(true),
                "false" | "0" | "no" | "" => Ok(false),
                other => Err(D::Error::custom(format!("not a boolean: {other}"))),
            },
            Value::Number(n) => Ok(n.as_i64() != Some(0)),
            _ => Err(D::Error::custom("expected a boolean")),
        }
    }

    /// "" and null are no filter.
    pub fn filter<'de, D: Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
        Ok(Option::<String>::deserialize(d)?.filter(|s| !s.trim().is_empty()))
    }
}

fn five() -> usize {
    5
}
fn twenty() -> usize {
    20
}

#[derive(Deserialize, JsonSchema)]
pub struct KbSearchArgs {
    /// Search words (e.g. "privacy architecture", "агент память")
    query: String,
    /// Number of results (default 5)
    #[serde(default = "five", deserialize_with = "lenient::usize")]
    n_results: usize,
    /// Filter by frontmatter `type` (substring match)
    #[serde(default, deserialize_with = "lenient::filter")]
    doc_type: Option<String>,
    /// Add up to 3 pages linked to or from the top hits ([[wikilinks]]), marked `graph_source`
    #[serde(default, deserialize_with = "lenient::bool")]
    expand_graph: bool,
    /// Only files under this path prefix (e.g. "wiki/")
    #[serde(default, deserialize_with = "lenient::filter")]
    path: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
pub struct SessionSearchArgs {
    /// Search words (e.g. "hubspot webhook", "транскрипция midi")
    query: String,
    /// Number of results (default 5)
    #[serde(default = "five", deserialize_with = "lenient::usize")]
    n_results: usize,
    /// Filter by project name, substring (e.g. "life2film")
    #[serde(default, deserialize_with = "lenient::filter")]
    project: Option<String>,
    /// A session_id from a result (or its first characters): read that session's
    /// passages matching `query` instead of searching
    #[serde(default, deserialize_with = "lenient::filter")]
    session_id: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
pub struct SymbolSearchArgs {
    /// Words or an identifier (e.g. "auth middleware", "MontageProfile")
    query: String,
    /// Limit to one project (exact registry name)
    #[serde(default, deserialize_with = "lenient::filter")]
    project: Option<String>,
    /// "function", "class" or "protocol"
    #[serde(default, deserialize_with = "lenient::filter")]
    kind: Option<String>,
    /// Number of results (default 20)
    #[serde(default = "twenty", deserialize_with = "lenient::usize")]
    n_results: usize,
}

#[derive(Deserialize, JsonSchema)]
pub struct ProjectInfoArgs {
    /// One project: its registry entry, stack, languages, layers, dependencies
    #[serde(default, deserialize_with = "lenient::filter")]
    name: Option<String>,
    /// A package name or part of one: every project that uses it (e.g. "supabase")
    #[serde(default, deserialize_with = "lenient::filter")]
    package: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
pub struct SourceSearchArgs {
    /// Search words (e.g. "startup idea", "revenue growth"). Omit, with no
    /// `related`, for the sources and their topics.
    #[serde(default, deserialize_with = "lenient::filter")]
    query: Option<String>,
    /// Only this source (e.g. "youtube", "telegram")
    #[serde(default, deserialize_with = "lenient::filter")]
    source: Option<String>,
    /// A video URL or id: videos sharing its topics instead of a search
    #[serde(default, deserialize_with = "lenient::filter")]
    related: Option<String>,
    /// Number of results (default 5)
    #[serde(default = "five", deserialize_with = "lenient::usize")]
    n_results: usize,
}

/// What the client is told at connect. With tool search on (Claude Code's
/// default) the agent sees only tool names and this text until it loads a tool,
/// so it says when to reach for each one, search words first.
const INSTRUCTIONS: &str =
    "solograph searches this user's own past work, offline. Before answering \
from scratch, search it when the request leans on something earlier: \"as we did\", \"what did \
we decide\", \"that script\", a name or id from before.
- session_search: past Claude Code conversations, every prompt and answer (decisions, commands, \
paths, ids). The session you are in is left out.
- kb_search: the user's notes and wiki (markdown).
- symbol_search: where a function or class is defined, across all the user's repositories.
- source_search: indexed YouTube talks, Telegram, ProductHunt, TrustMRR.
- project_info: the user's projects: the list, one project's stack and layers, who uses a package.
Matching is by words (BM25 over Russian and English stems), not by meaning: use the words the \
text itself would contain, and try the other language when nothing comes back.";

/// Loads with the session instead of behind tool search: the two tools that
/// answer "what did we do before" (Claude Code's `anthropic/alwaysLoad`).
fn always_load() -> rmcp::model::MetaObject {
    let mut m = serde_json::Map::new();
    m.insert("anthropic/alwaysLoad".into(), json!(true));
    rmcp::model::MetaObject(m)
}

/// A store opened on first use, behind its own lock: one lock for all four made
/// the portfolio counts wait 190 ms behind a session search (review, 28 Oct 2026).
type Slot<T> = std::sync::Arc<Mutex<Option<T>>>;

#[derive(Clone)]
pub struct Solograph {
    kb: Slot<KnowledgeBase>,
    sessions: Slot<SessionSearch>,
    sources: Slot<SourceIndex>,
    portfolio: Slot<Portfolio>,
    tool_router: ToolRouter<Self>,
}

/// The store in `slot`, opened by `open` the first time, handed to `f`.
fn with<S, T>(
    slot: &Mutex<Option<S>>,
    open: impl FnOnce() -> Result<S>,
    f: impl FnOnce(&mut S) -> Result<T>,
) -> Result<T> {
    let mut guard: MutexGuard<'_, Option<S>> = slot.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Some(open()?);
    }
    f(guard.as_mut().expect("opened above"))
}

fn list_result(items: Vec<Value>) -> CallToolResult {
    let mut r = CallToolResult::success(
        items
            .iter()
            .map(|v| ContentBlock::text(v.to_string()))
            .collect(),
    );
    r.structured_content = Some(json!({"result": items}));
    r
}

fn dict_result(v: Value) -> CallToolResult {
    CallToolResult::success(vec![ContentBlock::text(v.to_string())])
}

fn failed(tool: &str, e: anyhow::Error) -> CallToolResult {
    CallToolResult::error(vec![ContentBlock::text(format!(
        "Error executing tool {tool}: {e:#}"
    ))])
}

/// CODEGRAPH_REGISTRY, else ~/.solo/registry.yaml, when it exists.
fn registry_path() -> Option<PathBuf> {
    match env_path("CODEGRAPH_REGISTRY") {
        Some(p) => p.exists().then_some(p),
        None => Some(home().join(".solo").join("registry.yaml")).filter(|p| p.exists()),
    }
}

impl Solograph {
    pub fn new() -> Self {
        Self {
            kb: Default::default(),
            sessions: Default::default(),
            sources: Default::default(),
            portfolio: Default::default(),
            tool_router: Self::tool_router(),
        }
    }

    /// Every project the registry lists, or an error naming the fix.
    fn registry() -> Result<std::result::Result<Vec<Value>, Value>> {
        let Some(path) = registry_path() else {
            return Ok(Err(
                json!({"error": "Registry not found. Set CODEGRAPH_REGISTRY env var or run: solograph scan"}),
            ));
        };
        let data: serde_yaml::Value = serde_yaml::from_str(&std::fs::read_to_string(path)?)?;
        let data = crate::yamlval::to_json(&data);
        Ok(Ok(data["projects"].as_array().cloned().unwrap_or_default()))
    }

    /// One project: its registry entry, then what the portfolio index knows of it.
    /// The registry matches the name in any case and the index by its exact name,
    /// so the index is asked for the registry's spelling.
    fn project(&self, name: &str) -> Result<Value> {
        let lower = name.to_lowercase();
        let entry = match Self::registry()? {
            Ok(projects) => projects.into_iter().find(|p| {
                p["name"]
                    .as_str()
                    .is_some_and(|n| n.to_lowercase() == lower)
            }),
            Err(_) => None,
        };
        let exact = entry
            .as_ref()
            .and_then(|e| e["name"].as_str())
            .unwrap_or(name)
            .to_string();
        // No portfolio index yet (nothing scanned) still leaves the registry entry.
        let made_of = self
            .with_portfolio(|p| p.explain(&exact))
            .unwrap_or_else(|e| Err(format!("{e:#}")));
        Ok(match (entry, made_of) {
            (Some(Value::Object(mut e)), Ok(Value::Object(x))) => {
                for (k, v) in x {
                    e.entry(k).or_insert(v);
                }
                Value::Object(e)
            }
            (Some(Value::Object(mut e)), Err(msg)) => {
                e.insert("index".into(), json!(msg));
                Value::Object(e)
            }
            (Some(e), _) => e,
            (None, Ok(x)) => x,
            (None, Err(msg)) => json!({"error": msg}),
        })
    }

    /// All projects with their stacks, the portfolio's counts, and the ten
    /// packages the most projects share.
    fn portfolio_overview(&self) -> Result<Value> {
        let projects: Vec<Value> = match Self::registry()? {
            Ok(projects) => projects
                .iter()
                .map(|p| {
                    let mut o = Map::new();
                    o.insert("name".into(), p["name"].clone());
                    o.insert(
                        "stacks".into(),
                        p.get("stacks").cloned().unwrap_or(json!([])),
                    );
                    let day = p["last_commit"].as_str().unwrap_or("");
                    o.insert("last_commit".into(), json!(day.get(..10).unwrap_or(day)));
                    if let Some(st) = p["status"].as_str().filter(|s| *s != "active") {
                        o.insert("status".into(), json!(st));
                    }
                    Value::Object(o)
                })
                .collect(),
            Err(e) => return Ok(e),
        };
        let (counts, shared) = self
            .with_portfolio(|p| Ok((p.stats()?, p.shared_packages(None)?)))
            .unwrap_or_else(|e| (json!({"error": format!("{e:#}")}), Vec::new()));
        let shared: Vec<Value> = shared
            .iter()
            .take(10)
            .map(|p| json!({"package": p["package"], "source": p["source"], "projects": p["projects"].as_array().map_or(0, Vec::len)}))
            .collect();
        Ok(json!({"projects": projects, "counts": counts, "most_shared_packages": shared}))
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
        description = "Search the user's notes and wiki: markdown files with YAML frontmatter.\n\nBM25 over Russian and English stems; title, then summary and tags, then body, in that\norder of weight. A Russian word also searches the English terms the KB's alias table\nlists, and a typo is respelt. Edits are picked up on every call. `score` is BM25:\nhigher is better, comparable only within one answer.",
        meta = always_load()
    )]
    async fn kb_search(
        &self,
        Parameters(a): Parameters<KbSearchArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(root) = crate::kb::detect_root() else {
            return Ok(list_result(vec![
                json!({"error": "KB not configured. Set KB_PATH env var."}),
            ]));
        };
        let found = with(
            &self.kb,
            || KnowledgeBase::open(&root, None),
            |kb| {
                kb.search(
                    &a.query,
                    a.n_results,
                    a.doc_type.as_deref(),
                    a.expand_graph,
                    a.path.as_deref(),
                )
            },
        );
        Ok(match found {
            Ok(v) => list_result(v),
            Err(e) => failed("kb_search", e),
        })
    }

    #[tool(
        description = "Search past Claude Code sessions, every prompt and answer: \"how did we do X\",\n\"what did we decide about Y\", a command, path or id used before.\n\nBM25 over Russian and English stems, a typo respelt. Matching is lexical: use the words\nthe conversation would have used, and the other language if nothing comes back. The\nsession you are in is left out. Sessions Claude Code deleted from disk (it keeps 30\ndays) are still here, marked `transcript: deleted`. To read a result further, call again\nwith its `session_id`: the passages of that session that match `query`.",
        meta = always_load()
    )]
    async fn session_search(
        &self,
        Parameters(a): Parameters<SessionSearchArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let found = with(
            &self.sessions,
            || SessionSearch::open(None, None),
            |s| match a.session_id.as_deref() {
                Some(id) => s.read(id, &a.query, a.n_results),
                None => {
                    // Per call: the session asking changes under a running server.
                    s.skip(crate::sessions::calling_session());
                    s.search(&a.query, a.n_results, a.project.as_deref())
                }
            },
        );
        Ok(match found {
            Ok(v) => dict_result(v),
            Err(e) => failed("session_search", e),
        })
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
        description = "The user's projects. With `name`: that project's registry entry (path, status,\nlast commit) and what it is made of (stack, languages, directory layers, dependencies,\nthe files with the most definitions). With `package`: every project using a package\nmatching it. With neither: all projects with their stacks, portfolio counts, and the\npackages most projects share."
    )]
    async fn project_info(
        &self,
        Parameters(a): Parameters<ProjectInfoArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let out = match (a.name.as_deref(), a.package.as_deref()) {
            (Some(name), _) => self.project(name),
            (None, Some(package)) => self
                .with_portfolio(|p| p.shared_packages(Some(package)))
                .map(Value::Array),
            (None, None) => self.portfolio_overview(),
        };
        Ok(match out {
            Ok(Value::Array(items)) => list_result(items),
            Ok(v) => dict_result(v),
            Err(e) => failed("project_info", e),
        })
    }

    #[tool(
        description = "Search indexed external sources: YouTube talks, Telegram, ProductHunt, TrustMRR.\n\nBM25 with Russian and English stems; use the words the content would use. YouTube\nhits are the best transcript chunk per video, with chapter, timecode and a URL that\nopens at that moment. `score` is BM25 (ProductHunt upvotes add up to 15%). With\n`related` instead of a query: videos sharing that video's topics. With neither: the\nsources with their counts and topics."
    )]
    async fn source_search(
        &self,
        Parameters(a): Parameters<SourceSearchArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let source = a.source.as_deref();
        let out = self.with_sources(|s| match (a.query.as_deref(), a.related.as_deref()) {
            (Some(q), _) => s.search(q, source, a.n_results).map(Value::Array),
            (None, Some(video)) => s
                .related_videos(
                    source.unwrap_or("youtube"),
                    &crate::sources::youtube_id(video),
                )
                .map(Value::Array),
            (None, None) => Ok(json!({
                "sources": s.list_sources()?,
                "topics": s.list_tags(source.unwrap_or("youtube"))?,
            })),
        });
        Ok(match out {
            Ok(Value::Array(items)) => list_result(items),
            Ok(v) => dict_result(v),
            Err(e) => failed("source_search", e),
        })
    }
}

impl Solograph {
    fn with_portfolio<T>(&self, f: impl FnOnce(&mut Portfolio) -> Result<T>) -> Result<T> {
        with(&self.portfolio, || Portfolio::open(None), f)
    }

    fn with_sources<T>(&self, f: impl FnOnce(&SourceIndex) -> Result<T>) -> Result<T> {
        with(&self.sources, || SourceIndex::open(None), |s| f(s))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for Solograph {
    /// Every tool call goes through here, so the usage log sees all of them.
    async fn call_tool(
        &self,
        request: rmcp::model::CallToolRequestParams,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::CallToolResponse, ErrorData> {
        let started = std::time::Instant::now();
        let (name, args) = (request.name.to_string(), request.arguments.clone());
        // Clients on the stateless protocol (Claude Code since 2.1.3xx) send no
        // initialize: their name rides in each request's _meta.
        let client = context
            .meta
            .client_info()
            .map(|i| i.name)
            .or_else(|| context.peer.peer_info().map(|i| i.client_info.name.clone()));
        let tcc = rmcp::handler::server::tool::ToolCallContext::new(self, request, context);
        let out = self.tool_router.call(tcc).await;
        let result = match &out {
            Ok(rmcp::model::CallToolResponse::Complete(r)) => Some(r),
            _ => None,
        };
        crate::usage::record_tool(
            &name,
            args.as_ref(),
            result,
            started.elapsed(),
            client.as_deref(),
        );
        out
    }

    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("solograph", env!("CARGO_PKG_VERSION")))
            .with_instructions(INSTRUCTIONS)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{KbSearchArgs, ProjectInfoArgs, SymbolSearchArgs};

    #[test]
    fn strings_for_numbers_and_flags_and_empty_filters_are_understood() {
        let a: KbSearchArgs = serde_json::from_value(
            json!({"query": "x", "n_results": "10", "expand_graph": "true", "path": "", "doc_type": null}),
        )
        .unwrap();
        assert_eq!(
            (a.n_results, a.expand_graph, a.path, a.doc_type),
            (10, true, None, None)
        );
        let s: SymbolSearchArgs =
            serde_json::from_value(json!({"query": "x", "project": "", "kind": " "})).unwrap();
        assert_eq!((s.project, s.kind, s.n_results), (None, None, 20));
        let p: ProjectInfoArgs =
            serde_json::from_value(json!({"package": "", "name": null})).unwrap();
        assert_eq!((p.package, p.name), (None, None));
        assert!(
            serde_json::from_value::<KbSearchArgs>(json!({"query": "x", "n_results": "ten"}))
                .is_err()
        );
    }
}
