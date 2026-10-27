//! Read-only questions over the portfolio that the CLI asks and the MCP does not:
//! one project's dependencies, and names defined in several places.

use anyhow::Result;
use serde_json::{Value, json};

use super::Portfolio;

/// Entry points and route verbs repeat by design; noise when no name is asked for.
const DUPES_STOPLIST: &str =
    "'main','POST','GET','PUT','PATCH','DELETE','handler','default','constructor'";

pub struct DupesQuery<'a> {
    pub matching: Option<&'a str>,
    pub project: Option<&'a str>,
    pub min: usize,
    pub limit: usize,
}

impl Portfolio {
    /// A project's packages, by source then name.
    pub fn dependencies(&self, project: &str) -> Result<Vec<Value>> {
        let mut stmt = self.db.prepare(
            "SELECT name, version, source FROM packages WHERE project = ? ORDER BY source, name",
        )?;
        let rows = stmt.query_map([project], |r| {
            Ok(json!({
                "name": r.get::<_, String>(0)?,
                "version": r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                "source": r.get::<_, Option<String>>(2)?.unwrap_or_default(),
            }))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Names defined in several places — copy-paste as a class, which grep cannot
    /// raise. This is what Epiphan's `ep index query` was used for (a shared-secret
    /// check copied into five API routes, an HTML escape drifted in one of four
    /// copies); the call graph left with 0.8, definitions did not. Dunder methods and
    /// generated code are noise and are left out.
    pub fn dupes(&self, q: &DupesQuery) -> Result<Vec<Value>> {
        let mut filter = vec![
            r"s.name NOT LIKE '\_\_%' ESCAPE '\'".to_string(),
            "length(s.name) > 3".into(),
            "f.path NOT LIKE '%generated%'".into(),
            "f.path NOT LIKE '%_pb2%'".into(),
        ];
        let mut args: Vec<String> = Vec::new();
        match q.matching {
            Some(m) => {
                filter.push("s.name LIKE ?".into());
                args.push(format!("%{m}%"));
            }
            None => {
                filter.push(format!("s.name NOT IN ({DUPES_STOPLIST})"));
                filter.push("coalesce(f.lang, '') <> 'hcl'".into());
            }
        }
        if let Some(p) = q.project {
            filter.push("f.project = ?".into());
            args.push(p.to_string());
        }
        let sql = format!(
            "SELECT s.name, s.kind, count(*) AS n, count(DISTINCT f.project), \
             group_concat(f.project || '/' || f.path || ':' || s.line, char(9)) \
             FROM symbols s JOIN files f ON f.id = s.file_id WHERE {} \
             GROUP BY s.name, s.kind HAVING n >= {} ORDER BY n DESC, s.name LIMIT {}",
            filter.join(" AND "),
            q.min,
            q.limit
        );
        let mut stmt = self.db.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(args.iter()), |r| {
            let places: String = r.get(4)?;
            Ok(json!({
                "name": r.get::<_, String>(0)?,
                "kind": r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                "count": r.get::<_, i64>(2)?,
                "repos": r.get::<_, i64>(3)?,
                "places": places.split('\t').collect::<Vec<_>>(),
            }))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }
}
