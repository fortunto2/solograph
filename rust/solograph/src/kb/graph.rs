//! The [[wikilink]] graph of the KB, from the links each page's sync recorded:
//! always as fresh as the search index, with nothing to rebuild.
//!
//! Replaces scripts/kb_graph.py (a JSON file `expand_graph` read, fresh only when a
//! script ran: it sat at 36 nodes from 7 Feb to 27 Oct 2026 while the wiki passed 160
//! pages) and the grep pipelines behind `make wiki-graph`, `make wiki-orphans` and
//! wiki-lint's broken-link check.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::Result;
use serde_json::{Value, json};

use super::KnowledgeBase;

#[derive(Debug, Clone, Default)]
pub struct Page {
    pub title: String,
    pub tags: Vec<String>,
    pub links: Vec<String>,
}

#[derive(Debug, Default)]
pub struct Graph {
    pub pages: BTreeMap<String, Page>,
    /// (from, to), resolved, each pair once, no self-links.
    pub edges: BTreeSet<(String, String)>,
    /// (from, slug) that resolve to no page.
    pub broken: Vec<(String, String)>,
    /// The wiki folder (`Config::pages`): a slug is a page there first.
    wiki: String,
}

/// The catalog (`index.md`) links to every page and `_*.md` files are tooling: as
/// link sources they would make every page a neighbour of every other.
pub fn is_meta(id: &str) -> bool {
    let name = id.rsplit('/').next().unwrap_or(id);
    name == "index.md" || name.starts_with('_')
}

pub(super) fn stem(id: &str) -> &str {
    let name = id.rsplit('/').next().unwrap_or(id);
    name.strip_suffix(".md").unwrap_or(name)
}

fn dir(id: &str) -> &str {
    id.rsplit_once('/').map_or("", |(d, _)| d)
}

impl Graph {
    pub fn build(pages: BTreeMap<String, Page>, wiki: &str) -> Self {
        let mut by_stem: HashMap<String, Vec<&str>> = HashMap::new();
        for id in pages.keys() {
            by_stem
                .entry(stem(id).to_lowercase())
                .or_default()
                .push(id.as_str());
        }
        let resolve = |slug: &str, from: &str| -> Option<String> {
            let in_wiki = format!("{wiki}{slug}.md");
            if pages.contains_key(&in_wiki) {
                return Some(in_wiki);
            }
            let hits = by_stem.get(slug)?;
            match hits.as_slice() {
                [one] => Some((*one).to_string()),
                many => many
                    .iter()
                    .find(|h| dir(h) == dir(from))
                    .map(|h| (*h).to_string()),
            }
        };
        let mut edges = BTreeSet::new();
        let mut broken = Vec::new();
        for (from, p) in &pages {
            for slug in &p.links {
                match resolve(slug, from) {
                    Some(to) if &to != from => {
                        edges.insert((from.clone(), to));
                    }
                    Some(_) => {}
                    None => broken.push((from.clone(), slug.clone())),
                }
            }
        }
        Self {
            pages,
            edges,
            broken,
            wiki: wiki.to_string(),
        }
    }

    fn in_scope(id: &str, prefix: Option<&str>) -> bool {
        prefix.is_none_or(|p| id.starts_with(p))
    }

    /// Up to `cap` pages linked to or from the hits, best first: a neighbour of
    /// several hits, or of a higher-ranked one, scores more (Σ 1/rank). Catalog and
    /// tooling pages are never neighbours or the reason for one.
    pub fn neighbours(&self, hits: &[Value], prefix: Option<&str>, cap: usize) -> Vec<Value> {
        let ids: Vec<&str> = hits.iter().filter_map(|h| h["file"].as_str()).collect();
        let mut score: HashMap<&str, (f64, &str)> = HashMap::new();
        for (rank, hit) in ids.iter().enumerate() {
            let w = 1.0 / (rank + 1) as f64;
            for (a, b) in &self.edges {
                let other = if a == hit {
                    b.as_str()
                } else if b == hit {
                    a.as_str()
                } else {
                    continue;
                };
                if is_meta(a) || is_meta(other) || ids.contains(&other) {
                    continue;
                }
                if !Self::in_scope(other, prefix) {
                    continue;
                }
                let e = score.entry(other).or_insert((0.0, hit));
                e.0 += w;
            }
        }
        let indeg = self.in_degree();
        let mut ranked: Vec<(&str, f64, &str)> =
            score.into_iter().map(|(k, (s, via))| (k, s, via)).collect();
        ranked.sort_by(|x, y| {
            y.1.total_cmp(&x.1)
                .then_with(|| indeg.get(y.0).cmp(&indeg.get(x.0)))
                .then_with(|| x.0.cmp(y.0))
        });
        ranked
            .into_iter()
            .take(cap)
            .map(|(id, _, via)| {
                let p = self.pages.get(id).cloned().unwrap_or_default();
                let via_title = self.pages.get(via).map_or(via, |v| v.title.as_str());
                json!({
                    "file": id,
                    "title": p.title,
                    "tags": p.tags.join(","),
                    "graph_source": format!("via link: {via_title}"),
                })
            })
            .collect()
    }

    /// Inbound links per page, counting only real pages as sources.
    pub fn in_degree(&self) -> HashMap<&str, usize> {
        let mut d: HashMap<&str, usize> = HashMap::new();
        for (a, b) in &self.edges {
            if !is_meta(a) {
                *d.entry(b.as_str()).or_default() += 1;
            }
        }
        d
    }

    /// Hubs, connectors, orphans, broken links, pages missing from the catalog, top
    /// tags — and, with `page`, its backlinks and outgoing links. `prefix` scopes
    /// every list to pages under it (e.g. `wiki/`).
    pub fn report(&self, prefix: Option<&str>, page: Option<&str>, top: usize) -> Value {
        let scoped: Vec<&String> = self
            .pages
            .keys()
            .filter(|id| Self::in_scope(id, prefix))
            .collect();
        // Pages with a nonzero degree, most first.
        let ranked = |deg: &HashMap<&str, usize>| -> Vec<(&str, usize)> {
            let mut v: Vec<(&str, usize)> = scoped
                .iter()
                .filter(|id| !is_meta(id))
                .filter_map(|id| deg.get(id.as_str()).map(|n| (id.as_str(), *n)))
                .collect();
            v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
            v
        };
        let mut outdeg: HashMap<&str, usize> = HashMap::new();
        for (a, _) in &self.edges {
            *outdeg.entry(a.as_str()).or_default() += 1;
        }
        let hubs = ranked(&self.in_degree());
        let connectors = ranked(&outdeg);
        // Orphan: nothing links here at all — the catalog counts, as it always did
        // for `make wiki-orphans`, so a page is orphaned when it is neither linked
        // nor in the catalog.
        let linked: BTreeSet<&str> = self.edges.iter().map(|(_, b)| b.as_str()).collect();
        let orphans: Vec<&str> = scoped
            .iter()
            .filter(|id| !is_meta(id) && !linked.contains(id.as_str()))
            .map(|id| id.as_str())
            .collect();
        let broken: Vec<Value> = self
            .broken
            .iter()
            // The catalog's dead links count; `_*` tooling pages hold example links.
            .filter(|(from, _)| Self::in_scope(from, prefix) && !stem(from).starts_with('_'))
            .map(|(from, slug)| json!({"from": from, "link": slug}))
            .collect();
        let catalog = format!("{}index.md", prefix.unwrap_or(""));
        let in_catalog: BTreeSet<&str> = self
            .edges
            .iter()
            .filter(|(a, _)| *a == catalog)
            .map(|(_, b)| b.as_str())
            .collect();
        let not_in_index: Vec<&str> = if self.pages.contains_key(&catalog) {
            scoped
                .iter()
                .filter(|id| !is_meta(id) && !in_catalog.contains(id.as_str()))
                .map(|id| stem(id))
                .collect()
        } else {
            Vec::new()
        };
        let mut tags: HashMap<&str, usize> = HashMap::new();
        for id in scoped.iter().filter(|id| !is_meta(id)) {
            for t in &self.pages[id.as_str()].tags {
                *tags.entry(t.as_str()).or_default() += 1;
            }
        }
        let mut tags: Vec<(&str, usize)> = tags.into_iter().collect();
        tags.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        let pair = |v: &[(&str, usize)], key: &str| -> Vec<Value> {
            v.iter()
                .take(top)
                .map(|(id, n)| {
                    json!({"page": id, "title": self.pages.get(*id).map_or("", |p| p.title.as_str()), key: n})
                })
                .collect()
        };
        let mut out = json!({
            "scope": prefix.unwrap_or(""),
            "pages": scoped.iter().filter(|id| !is_meta(id)).count(),
            "links": self.edges.iter().filter(|(a, b)| Self::in_scope(a, prefix) && Self::in_scope(b, prefix)).count(),
            "hubs": pair(&hubs, "in"),
            "connectors": pair(&connectors, "out"),
            "orphans": orphans,
            "broken": broken,
            "not_in_index": not_in_index,
            "tags": tags.iter().take(top).map(|(t, n)| json!({"tag": t, "count": n})).collect::<Vec<_>>(),
        });
        if let Some(slug) = page {
            out["page"] = self.page_links(slug);
        }
        out
    }

    /// Backlinks and outgoing links of one page, by slug or id.
    pub fn page_links(&self, slug: &str) -> Value {
        let id = if self.pages.contains_key(slug) {
            slug.to_string()
        } else {
            self.pages
                .keys()
                .find(|id| stem(id).eq_ignore_ascii_case(slug) && id.starts_with(&self.wiki))
                .or_else(|| {
                    self.pages
                        .keys()
                        .find(|id| stem(id).eq_ignore_ascii_case(slug))
                })
                .cloned()
                .unwrap_or_default()
        };
        let back: Vec<&str> = self
            .edges
            .iter()
            .filter(|(a, b)| *b == id && !is_meta(a))
            .map(|(a, _)| a.as_str())
            .collect();
        let out: Vec<&str> = self
            .edges
            .iter()
            .filter(|(a, _)| *a == id)
            .map(|(_, b)| b.as_str())
            .collect();
        json!({"id": id, "backlinks": back, "links": out})
    }
}

impl KnowledgeBase {
    /// The link graph of the whole KB, synced first.
    pub fn graph(&mut self) -> Result<Graph> {
        self.sync()?;
        self.link_graph()
    }

    /// The link graph as the index holds it now.
    pub fn link_graph(&self) -> Result<Graph> {
        let mut stmt = self.store.db.prepare(
            "SELECT d.doc_id, d.meta, f.title FROM docs d LEFT JOIN fts f ON f.rowid = d.id",
        )?;
        let mut pages = BTreeMap::new();
        for row in stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Option<String>>(1)?,
                r.get::<_, Option<String>>(2)?,
            ))
        })? {
            let (id, meta, title) = row?;
            let meta: Value =
                serde_json::from_str(meta.as_deref().unwrap_or("{}")).unwrap_or_default();
            let links = meta["links"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            let tags = meta["tags"]
                .as_str()
                .unwrap_or("")
                .split(',')
                .map(str::trim)
                .filter(|t| !t.is_empty())
                .map(String::from)
                .collect();
            pages.insert(
                id,
                Page {
                    title: title.unwrap_or_default(),
                    tags,
                    links,
                },
            );
        }
        Ok(Graph::build(pages, &self.config.pages))
    }
}

#[cfg(test)]
mod tests;
