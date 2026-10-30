//! Search over Claude Code transcripts, including the ones the disk has deleted.
//! Port of session_search.py: same store, same reader, same coverage report.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::Result;
use memchr::memmem;
use serde_json::{Map, Value, json};

use crate::fts::{Doc, FtsStore, SyncReport};
use crate::paths::{env_path, home};

/// '-Users-john-projects-MyApp-ios' → 'MyApp/ios'; no 'projects' → last segment.
pub fn derive_project_name(dir_name: &str) -> String {
    let parts: Vec<&str> = dir_name.trim_start_matches('-').split('-').collect();
    match parts.iter().position(|p| *p == "projects") {
        Some(i) if i + 1 < parts.len() => parts[i + 1..].join("/"),
        Some(_) => "projects".into(),
        None => parts
            .last()
            .map_or_else(|| dir_name.to_string(), |s| s.to_string()),
    }
}

/// What a slash command was called with: "/solo:research therapist marketplace".
/// Claude Code writes it as tags, and every user text starting with '<' used to
/// be dropped with them (review, 28 Oct 2026).
fn slash_command(text: &str) -> Option<String> {
    let tag = |t: &str| -> Option<&str> {
        let open = format!("<{t}>");
        let start = text.find(&open)? + open.len();
        let end = text[start..].find(&format!("</{t}>"))? + start;
        Some(text[start..end].trim())
    };
    let name = tag("command-name")?;
    let args = tag("command-args").unwrap_or("");
    Some(format!("{name} {args}").trim().to_string())
}

/// Whether a user text block is what the person typed: a prompt (with or without
/// "[Image #N]") or a slash command, not text Claude Code adds in tags
/// (`<task-notification>`, `<local-command-stdout>`) or writes itself
/// ("[Request interrupted by user]"). Skill bodies and image notes are `isMeta`
/// lines, judged before the text.
pub fn typed(text: &str) -> bool {
    slash_command(text).is_some()
        || !(text.starts_with('<') || text.starts_with("[Request interrupted"))
}

/// A path as one shell word.
fn shell_quote(p: &str) -> String {
    if p.chars()
        .all(|c| c.is_ascii_alphanumeric() || "/._-~".contains(c))
    {
        p.to_string()
    } else {
        format!("'{}'", p.replace('\'', "'\\''"))
    }
}

pub fn claude_dir() -> PathBuf {
    env_path("CLAUDE_SESSIONS_DIR").unwrap_or_else(|| home().join(".claude"))
}

pub fn default_db_path() -> PathBuf {
    env_path("CODEGRAPH_SESSIONS_DB")
        .unwrap_or_else(|| home().join(".solo").join("sessions"))
        .join("search.db")
}

/// Batch runs (`claude -p` from /tmp, an agent's scratchpad), not conversations.
const BATCH_DIRS: [&str; 3] = ["-private-tmp", "-tmp-", "-private-var-folders"];

/// (session_id, path) for conversation transcripts; None if the directory cannot be read.
pub fn transcript_files(projects_dir: &Path) -> Option<Vec<(String, PathBuf)>> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(projects_dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    dirs.sort();
    let mut by_id: HashMap<String, PathBuf> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    for dir in dirs {
        let name = dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if !dir.is_dir() || BATCH_DIRS.iter().any(|b| name.starts_with(b)) || name == "-tmp" {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
        paths.sort();
        for p in paths {
            if p.extension().is_some_and(|x| x == "jsonl") && p.is_file() {
                let id = p
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned();
                if by_id.insert(id.clone(), p).is_none() {
                    order.push(id);
                }
            }
        }
    }
    Some(
        order
            .into_iter()
            .map(|id| {
                let p = by_id.remove(&id).unwrap_or_default();
                (id, p)
            })
            .collect(),
    )
}

// Byte markers checked before parsing: tool results are most of a transcript's
// bytes and none of its text, so they are never parsed.
const USER: &[u8] = br#""type":"user""#;
const ASSISTANT: &[u8] = br#""type":"assistant""#;
const TITLE: &[u8] = br#""type":"ai-title""#;
const SKIP_USER: [&[u8]; 2] = [br#""toolUseResult""#, br#""isMeta":true"#];
const TEXT_BLOCK: &[u8] = br#""type":"text""#;
const FILE_TOOLS: [&str; 2] = ["Edit", "Write"];
const FILE_TOOL_MARKERS: [&[u8]; 2] = [br#""name":"Edit""#, br#""name":"Write""#];

fn has(hay: &[u8], needle: &[u8]) -> bool {
    memmem::find(hay, needle).is_some()
}

fn worth_parsing(line: &[u8]) -> bool {
    if has(line, USER) {
        !SKIP_USER.iter().any(|m| has(line, m))
    } else if has(line, ASSISTANT) {
        has(line, TEXT_BLOCK) || FILE_TOOL_MARKERS.iter().any(|m| has(line, m))
    } else {
        has(line, TITLE)
    }
}

/// The searchable text of a transcript from `offset` on, and where reading stopped.
///
/// Prompts, answers and edited files, in order. Skill expansions (isMeta) are
/// skipped. A last line without its newline is still being written: reading
/// stops before it, and `end` says where to resume.
static IMAGE_MARK: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r"^\s*(\[Image #\d+\]\s*)+").expect("static regex")
});

/// A prompt as a title: "[Image #3]" markers dropped, and None for lines Claude
/// Code writes itself ("[Request interrupted by user for tool use]" was the title
/// of 82 archived sessions).
pub fn clean_title(text: &str) -> Option<String> {
    let t = IMAGE_MARK.replace(text, "");
    let t = t.trim();
    if t.is_empty() || t.starts_with("[Request interrupted") {
        return None;
    }
    Some(t.to_string())
}

/// Titles stored before clean_title existed, fixed in place: a system line or an
/// "[Image #N]" prefix becomes the first real prompt. Archived summaries list
/// their prompts as "1. …"; live documents start with theirs.
/// `PRAGMA user_version` of a session index whose titles have been through retitle().
const TITLES_CLEANED: i64 = 1;

fn retitle(store: &FtsStore) -> Result<usize> {
    let rows: Vec<(i64, String, String)> = store
        .db
        .prepare("SELECT rowid, title, body FROM fts WHERE title LIKE '[%'")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<rusqlite::Result<_>>()?;
    static NUMBERED: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"(?m)^\d+\. (.+)$").expect("static regex"));
    let mut fixed = 0;
    for (rowid, title, body) in rows {
        let numbered: Vec<&str> = NUMBERED
            .captures_iter(&body)
            .filter_map(|c| c.get(1))
            .map(|m| m.as_str())
            .collect();
        let candidates: Vec<&str> = if body.contains("\nPrompts:\n") && !numbered.is_empty() {
            numbered
        } else {
            std::iter::once(title.as_str())
                .chain(body.split("\n\n"))
                .collect()
        };
        let Some(new) = candidates.into_iter().find_map(clean_title) else {
            continue;
        };
        let new: String = new.replace('\n', " ").chars().take(200).collect();
        if new != title {
            store.db.execute(
                "UPDATE fts SET title = ? WHERE rowid = ?",
                rusqlite::params![new, rowid],
            )?;
            fixed += 1;
        }
    }
    Ok(fixed)
}

pub fn read_transcript(path: &Path, offset: i64) -> Result<Doc> {
    let mut file = BufReader::with_capacity(1 << 20, File::open(path)?);
    file.seek(SeekFrom::Start(offset as u64))?;
    let (mut parts, mut title, mut first_prompt) =
        (Vec::<String>::new(), String::new(), String::new());
    let mut meta: Map<String, Value> = Map::new();
    let mut end = offset;
    let mut raw = Vec::new();
    loop {
        raw.clear();
        if file.read_until(b'\n', &mut raw)? == 0 || raw.last() != Some(&b'\n') {
            break;
        }
        end += raw.len() as i64;
        if !worth_parsing(&raw) {
            continue;
        }
        let Ok(Value::Object(obj)) = serde_json::from_slice::<Value>(&raw) else {
            continue;
        };
        let typ = obj.get("type").and_then(Value::as_str).unwrap_or("");
        if typ == "ai-title" {
            if let Some(t) = obj
                .get("aiTitle")
                .and_then(Value::as_str)
                .filter(|t| !t.is_empty())
            {
                title = t.to_string();
            }
            continue;
        }
        if obj.get("cwd").is_some_and(crate::fts::truthy) && !meta.contains_key("project_path") {
            meta.insert("project_path".into(), obj["cwd"].clone());
            meta.insert(
                "branch".into(),
                obj.get("gitBranch").cloned().unwrap_or_else(|| json!("")),
            );
        }
        if let Some(ts) = obj.get("timestamp").filter(|v| crate::fts::truthy(v)) {
            meta.entry("started_at").or_insert_with(|| ts.clone());
            meta.insert("ended_at".into(), ts.clone());
        }
        let content = obj.get("message").and_then(|m| m.get("content"));
        let blocks: Vec<Value> = match content {
            Some(Value::String(s)) => vec![json!({"type": "text", "text": s})],
            Some(Value::Array(a)) => a.clone(),
            _ => Vec::new(),
        };
        for block in blocks.iter().filter_map(Value::as_object) {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    let text = block.get("text").and_then(Value::as_str).unwrap_or("");
                    if typ == "user"
                        && let Some(cmd) = slash_command(text)
                    {
                        if first_prompt.is_empty() {
                            first_prompt = cmd.clone();
                        }
                        parts.push(cmd);
                    } else if text.chars().count() > 5 && !text.starts_with('<') {
                        parts.push(text.to_string());
                        if typ == "user" && first_prompt.is_empty() {
                            first_prompt = clean_title(text).unwrap_or_default();
                        }
                    }
                }
                Some("tool_use")
                    if block
                        .get("name")
                        .and_then(Value::as_str)
                        .is_some_and(|n| FILE_TOOLS.contains(&n)) =>
                {
                    let fp = block
                        .get("input")
                        .and_then(|i| i.get("file_path"))
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    if !fp.is_empty() {
                        parts.push(format!("Edited: {fp}"));
                    }
                }
                _ => {}
            }
        }
    }
    if offset == 0 {
        if title.is_empty() {
            title = first_prompt
                .replace('\n', " ")
                .trim()
                .chars()
                .take(200)
                .collect();
        }
    } else {
        for k in ["started_at", "project_path", "branch"] {
            meta.shift_remove(k);
        }
    }
    let label = derive_project_name(
        &path
            .parent()
            .and_then(Path::file_name)
            .unwrap_or_default()
            .to_string_lossy(),
    );
    // The working directory names the project; the Claude dir name split on '-'
    // turned "video-analyzer" into "analyzer" (review, 28 Oct 2026). `kind` holds
    // the full path, so a project filter (substring) matches its real name.
    let cwd = meta
        .get("project_path")
        .and_then(Value::as_str)
        .unwrap_or("");
    let project = Path::new(cwd)
        .file_name()
        .map_or_else(|| label.clone(), |n| n.to_string_lossy().into_owned());
    let kind = if cwd.is_empty() {
        label
    } else {
        cwd.to_string()
    };
    let mut full = Map::new();
    full.insert(
        "session_id".into(),
        json!(path.file_stem().unwrap_or_default().to_string_lossy()),
    );
    full.insert("project".into(), json!(project));
    full.extend(meta);
    Ok(Doc {
        title,
        keys: String::new(),
        body: parts.join("\n\n"),
        kind,
        meta: full,
        end: Some(end),
    })
}

/// Rows written before a session's project was its working directory: `kind`
/// becomes the path and `project` its last part. Writes only when a row needs it,
/// so an index already relabelled costs one read.
fn relabel(store: &FtsStore) -> Result<()> {
    let rows: Vec<(String, String)> = store
        .db
        .prepare(
            "SELECT doc_id, json_extract(meta, '$.project_path') AS pp FROM docs \
             WHERE coalesce(pp, '') != '' AND kind IS NOT pp",
        )?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    if rows.is_empty() {
        return Ok(());
    }
    let tx = store.db.unchecked_transaction()?;
    for (doc_id, pp) in rows {
        let name = Path::new(&pp)
            .file_name()
            .map_or_else(|| pp.clone(), |n| n.to_string_lossy().into_owned());
        tx.execute(
            "UPDATE docs SET kind = ?2, meta = json_set(meta, '$.project', ?3) WHERE doc_id = ?1",
            rusqlite::params![doc_id, pp, name],
        )?;
    }
    tx.commit()?;
    Ok(())
}

/// A session under a folder holding one of these is never indexed.
pub const PRIVATE_MARKERS: [&str; 2] = [".solograph-private", ".metadata_never_index"];

fn session_cwd(path: &Path) -> String {
    let Ok(f) = File::open(path) else {
        return String::new();
    };
    for (i, line) in BufReader::new(f).split(b'\n').enumerate() {
        let Ok(line) = line else { break };
        if has(&line, br#""cwd""#)
            && let Ok(Value::Object(o)) = serde_json::from_slice::<Value>(&line)
        {
            return o
                .get("cwd")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
        }
        if i > 200 {
            break;
        }
    }
    String::new()
}

pub(crate) fn is_private(cwd: &str) -> bool {
    !cwd.is_empty()
        && Path::new(cwd)
            .ancestors()
            .any(|p| PRIVATE_MARKERS.iter().any(|m| p.join(m).exists()))
}

/// Whether a session started in `cwd` stays out of the index. No cwd at all is
/// private too: with nothing to check, the safe answer is no.
fn keep_out(cwd: &str) -> bool {
    cwd.is_empty() || is_private(cwd)
}

const GONE: &str = "deleted from disk; this text is the only copy";

pub struct SessionSearch {
    pub store: FtsStore,
    projects_dir: PathBuf,
    /// Each transcript's first working directory, which never changes once
    /// written. The privacy verdict is not cached: a marker can appear any time.
    cwds: HashMap<PathBuf, String>,
}

/// The Claude Code session asking. Claude Code passes its id to MCP servers and
/// to the shell, but an MCP server outlives `/clear`, which starts a new session
/// with the old id still in the server's environment. So the id is checked
/// against the transcripts of the project directory: the asking session's was
/// written in the last minutes. The environment's id when it is one of those (or
/// when nothing can be checked), else the only live one, else none.
pub fn calling_session() -> Option<String> {
    let env = std::env::var("CLAUDE_CODE_SESSION_ID")
        .ok()
        .filter(|s| !s.is_empty());
    let project = env_path("CLAUDE_PROJECT_DIR").or_else(|| std::env::current_dir().ok())?;
    let live = live_sessions(&claude_dir(), &project, std::time::SystemTime::now());
    pick_session(env, &live)
}

fn pick_session(env: Option<String>, live: &[String]) -> Option<String> {
    match (env, live) {
        (Some(e), _) if live.is_empty() || live.contains(&e) => Some(e),
        (_, [one]) => Some(one.clone()),
        _ => None,
    }
}

/// Sessions of `project` whose transcript changed in the last two minutes, or one
/// of their subagents' (`<id>/subagents/*.jsonl`): a session waiting on a subagent
/// writes nothing itself, and the subagent's calls are its calls. Claude Code keeps
/// them in `projects/<the path with every other character as ->`.
fn live_sessions(claude: &Path, project: &Path, now: std::time::SystemTime) -> Vec<String> {
    let name: String = project
        .to_string_lossy()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let recent = |p: &Path| {
        std::fs::metadata(p)
            .and_then(|m| m.modified())
            .is_ok_and(|m| now.duration_since(m).is_ok_and(|d| d.as_secs() <= 120))
    };
    let files = |dir: PathBuf| {
        std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
    };
    files(claude.join("projects").join(name))
        .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .filter(|p| recent(p) || files(p.with_extension("").join("subagents")).any(|s| recent(&s)))
        .filter_map(|p| Some(p.file_stem()?.to_string_lossy().into_owned()))
        .collect()
}

impl SessionSearch {
    pub fn open(db_path: Option<&Path>, sessions_root: Option<&Path>) -> Result<Self> {
        let db = db_path.map_or_else(default_db_path, Path::to_path_buf);
        let root = sessions_root.map_or_else(claude_dir, Path::to_path_buf);
        let mut store = FtsStore::open(&db)?;
        store.redact_existing()?;
        // Once per index, not per open: the scan reads every row of the FTS table,
        // bodies included (tens of MB), which made each CLI call ~2 s. New documents
        // get a clean title when they are written.
        let version: i64 = store
            .db
            .query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if version < TITLES_CLEANED {
            retitle(&store)?;
            store
                .db
                .execute_batch(&format!("PRAGMA user_version = {TITLES_CLEANED}"))?;
        }
        relabel(&store)?;
        Ok(Self {
            store,
            projects_dir: root.join("projects"),
            cwds: HashMap::new(),
        })
    }

    /// Judged per transcript, every sync. It used to be cached per Claude project
    /// dir from that dir's first transcript, for the server's lifetime (review,
    /// 28 Oct 2026): a cwd-less first file made a private session searchable, and
    /// a marker added while the server ran was ignored.
    fn cwd_of(&mut self, path: &Path) -> String {
        if let Some(c) = self.cwds.get(path) {
            return c.clone();
        }
        let c = session_cwd(path);
        // Not cached while empty: the first lines of a new transcript carry none.
        if !c.is_empty() {
            self.cwds.insert(path.to_path_buf(), c.clone());
        }
        c
    }

    /// Archived sessions (transcript deleted) whose project folder is private now.
    fn archived_private(&self, verdict: &mut HashMap<String, bool>) -> Result<Vec<String>> {
        let mut stmt = self.store.db.prepare(
            "SELECT doc_id, json_extract(meta, '$.project_path') FROM docs WHERE gone = 1",
        )?;
        let rows: Vec<(String, Option<String>)> = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        // An archived row with no recorded folder was imported from the old vector
        // store; there is nothing to check it against, and it is the only copy.
        Ok(rows
            .into_iter()
            .filter_map(|(id, pp)| {
                let pp = pp.filter(|p| !p.is_empty())?;
                let private = *verdict.entry(pp.clone()).or_insert_with(|| is_private(&pp));
                private.then_some(id)
            })
            .collect())
    }

    /// Sync the index with the transcripts on disk. None = directory unreadable.
    pub fn sync(&mut self) -> Result<Option<(SyncReport, usize, usize)>> {
        let Some(files) = transcript_files(&self.projects_dir) else {
            return Ok(None);
        };
        let mut verdict: HashMap<String, bool> = HashMap::new();
        let mut private: HashSet<String> = HashSet::new();
        for (id, p) in &files {
            let cwd = self.cwd_of(p);
            if *verdict.entry(cwd.clone()).or_insert_with(|| keep_out(&cwd)) {
                private.insert(id.clone());
            }
        }
        let archived = self.archived_private(&mut verdict)?;
        let mut gone: HashSet<String> = private.clone();
        gone.extend(archived);
        self.store.forget(&gone)?;
        let files: Vec<(String, PathBuf)> = files
            .into_iter()
            .filter(|(id, _)| !private.contains(id))
            .collect();
        let parse = |p: &Path| read_transcript(p, 0).map(Some);
        let append = |p: &Path, off: i64| read_transcript(p, off);
        let report = self.store.sync(&files, &parse, true, Some(&append))?;
        Ok(Some((report, files.len(), private.len())))
    }

    /// Sessions for `query`, without `skip`: the session asking, whose prompt holds
    /// the very words searched for, so it came first in 7 of 18 agent searches
    /// (Oct 2026).
    pub fn search(
        &mut self,
        query: &str,
        n_results: usize,
        project: Option<&str>,
        skip: Option<&str>,
    ) -> Result<Value> {
        let synced = self.sync()?;
        let mut results = Vec::new();
        let extra = usize::from(skip.is_some());
        let hits = self.ranked(query, n_results.saturating_add(extra), project)?;
        for hit in hits
            .into_iter()
            .filter(|h| h["doc_id"].as_str() != skip)
            .take(n_results)
        {
            let doc_id = hit["doc_id"].as_str().unwrap_or("").to_string();
            let mut r = Map::new();
            r.insert("session_id".into(), json!(doc_id));
            r.insert(
                "project".into(),
                hit.get("project")
                    .cloned()
                    .unwrap_or_else(|| hit["kind"].clone()),
            );
            r.insert(
                "started_at".into(),
                hit.get("started_at").cloned().unwrap_or_else(|| json!("")),
            );
            r.insert("title".into(), hit["title"].clone());
            r.insert("snippet".into(), hit["snippet"].clone());
            r.insert("score".into(), hit["score"].clone());
            if hit["gone"] == true {
                r.insert("transcript".into(), json!(GONE));
            } else if let Some(pp) = hit.get("project_path").filter(|v| crate::fts::truthy(v)) {
                let pp = pp.as_str().map_or_else(|| pp.to_string(), str::to_string);
                r.insert(
                    "resume".into(),
                    json!(format!(
                        "cd {} && claude --resume {doc_id}",
                        shell_quote(&pp)
                    )),
                );
            }
            results.push(Value::Object(r));
        }
        let cov = coverage(synced.as_ref(), self.store.count()?);
        Ok(json!({"results": results, "coverage": cov}))
    }
}

impl SessionSearch {
    /// One session read for `query`: the `n` passages that hold most of its words.
    /// `session` is a session_id from a search, or its first characters.
    pub fn read(&mut self, session: &str, query: &str, n: usize) -> Result<Value> {
        self.sync()?;
        let rows: Vec<(String, Option<String>, i64, String, String)> = self
            .store
            .db
            .prepare(
                "SELECT d.doc_id, d.meta, d.gone, f.title, f.body FROM docs d \
                 JOIN fts f ON f.rowid = d.id WHERE substr(d.doc_id, 1, length(?1)) = ?1 LIMIT 2",
            )?
            .query_map([session.trim()], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
            })?
            .collect::<rusqlite::Result<_>>()?;
        let (doc_id, meta, gone, title, body) = match rows.as_slice() {
            [one] => one,
            [] => return Ok(json!({"error": format!("No session {session} in the index")})),
            _ => return Ok(json!({"error": format!("{session} starts more than one session id")})),
        };
        let meta: Value = meta
            .as_deref()
            .and_then(|m| serde_json::from_str(m).ok())
            .unwrap_or(Value::Null);
        let expr = any_word(query, self.respelt(query).as_deref());
        let passages: Vec<Value> = crate::text::passages(body, &expr, n, 700)
            .into_iter()
            .map(|(part, text)| json!({"part": part, "text": text}))
            .collect();
        Ok(json!({
            "session_id": doc_id,
            "project": meta.get("project").cloned().unwrap_or(Value::Null),
            "started_at": meta.get("started_at").cloned().unwrap_or(Value::Null),
            "title": title,
            "parts": body.split("\n\n").count(),
            "passages": passages,
            "transcript": if *gone == 1 { GONE } else { "on disk" },
        }))
    }
}

/// Any word of `query` in either spelling: the widest search pass, and what a read
/// looks for. `fixed` is the query respelt, when it had a typo.
fn any_word(query: &str, fixed: Option<&str>) -> String {
    crate::text::union(
        &crate::text::match_expression(query),
        &crate::text::match_expression(fixed.unwrap_or(query)),
    )
}

impl SessionSearch {
    /// The query with its typos fixed (fts/spell.rs), when it had any. A failed
    /// lookup is no fix: the query as written still searches.
    fn respelt(&self, query: &str) -> Option<String> {
        self.store.respell(query, &[]).ok().flatten()
    }

    /// All the query's words before some of them. Measured on 37 known-answer
    /// sessions (27 Oct 2026): recall@1 26 → 29, recall@5 30 → 32. Title weight
    /// 1 or 10 instead of 3 and a recency boost did not help, and were left out.
    /// A query whose words never meet in one session falls back to OR, as before.
    fn ranked(
        &self,
        query: &str,
        n: usize,
        project: Option<&str>,
    ) -> Result<Vec<Map<String, Value>>> {
        // Passes in turn, each filling what the one before left: every word as
        // written, every word in either spelling, any word in either. A typo drops
        // its word from the first (fts/spell.rs); a session that holds the words as
        // written, typos and all ("придуматьк"), stays first.
        let fixed = self.respelt(query);
        let mut passes = vec![crate::text::all_terms(query)];
        if let Some(q) = &fixed {
            passes.push(crate::text::all_terms_either(query, q));
        }
        passes.push(any_word(query, fixed.as_deref()));
        passes.dedup();
        let mut hits: Vec<Map<String, Value>> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for expr in &passes {
            if hits.len() >= n {
                break;
            }
            for h in self.store.search_expr(expr, n, project)? {
                if hits.len() < n && seen.insert(h["doc_id"].to_string()) {
                    hits.push(h);
                }
            }
        }
        Ok(hits)
    }
}

/// What this answer could see. Never "complete" when we could not look.
pub fn coverage(synced: Option<&(SyncReport, usize, usize)>, indexed: i64) -> Value {
    let Some((r, on_disk, private)) = synced else {
        return json!({
            "status": "unknown",
            "indexed": indexed,
            "why": "the transcript directory is missing or unreadable; only the archive was searched",
        });
    };
    let mut cov = Map::new();
    cov.insert("on_disk".into(), json!(on_disk));
    cov.insert("indexed".into(), json!(r.indexed));
    cov.insert("archived".into(), json!(r.gone));
    cov.insert("read_now".into(), json!(r.read));
    cov.insert("private_skipped".into(), json!(private));
    if r.failed.is_empty() {
        cov.insert("status".into(), json!("complete"));
    } else {
        cov.insert("status".into(), json!("partial"));
        cov.insert(
            "failed".into(),
            json!(r.failed.iter().take(10).collect::<Vec<_>>()),
        );
        cov.insert(
            "why".into(),
            json!(format!(
                "{} transcript(s) could not be read, so this answer is not about them",
                r.failed.len()
            )),
        );
    }
    Value::Object(cov)
}

#[cfg(test)]
mod tests;
