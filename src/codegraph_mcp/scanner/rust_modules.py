"""Resolve Rust `use` paths to the files that define them.

Rust was the one language whose internal imports resolved to nothing — 0 of
1223 in this workspace. The generic resolver looks for the module path as a
*path fragment* (`.utils` → `utils`, which works for Python and JS), and a
Rust module path is not a path: `crate::sgr_types::tasks` lives in
`crates/va-agent/src/sgr_types/tasks.rs`, or in `…/tasks/mod.rs`, and the
crate it belongs to is named by a directory whose dashes became underscores.

So the mapping has to be built from the file layout, once per scan:

    crates/va-agent/src/montage.rs           → va_agent :: montage
    crates/va-agent/src/sgr_types/tasks.rs   → va_agent :: sgr_types::tasks
    crates/va-agent/src/actions/mod.rs       → va_agent :: actions
    src/pipeline/neural.rs                   → <root crate> :: pipeline::neural

Then a `use` resolves by walking its segments from longest to shortest,
because the tail of a path is usually a symbol rather than a module:
`crate::sgr_types::tasks::montage_tool_defs` is the module `sgr_types::tasks`
plus a function. Longest-first also means `a::b::c` prefers `a/b/c.rs` over
`a/b.rs`, which is what Rust itself does.

Path dependencies are the reason this matters here: the app reaches the
sibling `video-*` crates through them, and without resolution the graph shows
a file importing nothing at all.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RustModule:
    crate: str  # `va_agent` — underscores, as `use` spells it
    module: str  # `sgr_types::tasks`, empty for a crate root
    file: str  # path relative to the project


def _crate_and_module(rel_path: str) -> RustModule | None:
    """Map one .rs file to the crate and module path a `use` would name."""
    if not rel_path.endswith(".rs"):
        return None
    parts = rel_path.split("/")
    if "src" not in parts:
        return None

    src_at = parts.index("src")
    # The crate is the directory holding src/ — `crates/va-agent/src/…` is
    # crate va_agent. A src/ at the top belongs to the root package, whose
    # name we do not know from the path; "" matches any crate below.
    crate = parts[src_at - 1].replace("-", "_") if src_at > 0 else ""

    tail = parts[src_at + 1 :]
    if not tail:
        return None
    # lib.rs and main.rs are the crate root, not a module named "lib".
    if tail[-1] in ("lib.rs", "main.rs") or tail[-1] == "mod.rs":
        module_parts = tail[:-1]
    else:
        module_parts = tail[:-1] + [tail[-1][:-3]]

    return RustModule(crate=crate, module="::".join(module_parts), file=rel_path)


def build_index(files) -> dict[tuple[str, str], str]:
    """(crate, module) → file, for every Rust file in the project."""
    index: dict[tuple[str, str], str] = {}
    for f in files:
        if getattr(f, "lang", None) != "rust":
            continue
        m = _crate_and_module(f.path)
        if m:
            index[(m.crate, m.module)] = m.file
    return index


def resolve(use_path: str, source_file: str, index: dict[tuple[str, str], str]) -> str | None:
    """The file a `use` path names, or None.

    `crate::` and `self::` stay inside the importing file's own crate;
    `super::` is treated the same way, since one level up is still that crate
    and the longest-match walk finds the right module anyway. A leading
    identifier that is not one of those is a crate name — which is how a
    path dependency on a sibling workspace resolves.
    """
    segments = [s for s in use_path.split("::") if s]
    if not segments:
        return None

    own_crate = ""
    m = _crate_and_module(source_file)
    if m:
        own_crate = m.crate

    if segments[0] in ("crate", "self", "super"):
        crate = own_crate
        segments = segments[1:]
    else:
        crate = segments[0]
        segments = segments[1:]
        # `use serde::…` is an external crate unless a file in this project
        # actually belongs to a crate of that name.
        if not any(c == crate for c, _ in index):
            return None

    # Longest first: the tail of a use path is usually a symbol, not a module.
    for cut in range(len(segments), -1, -1):
        module = "::".join(segments[:cut])
        hit = index.get((crate, module))
        if hit and hit != source_file:
            return hit
        # A crate root's files can sit under a src/ with no crate directory
        # (the workspace's own package), recorded with crate "".
        hit = index.get(("", module))
        if hit and hit != source_file:
            return hit
    return None
