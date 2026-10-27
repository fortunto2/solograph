"""Dependency manifests. Found while porting to Rust (27 Oct 2026): the Package.swift
parser took `.git` off a URL with rstrip(".git"), which strips any trailing '.',
'g', 'i' and 't' — so apple/swift-log.git came out as "swift-lo"."""

from __future__ import annotations

from codegraph_mcp.scanner.deps import scan_deps


def test_a_swift_package_keeps_the_g_at_the_end_of_its_name(tmp_path):
    (tmp_path / "Package.swift").write_text(
        '.package(url: "https://github.com/apple/swift-log.git", from: "1.0.0"),\n'
        '.package(url: "https://github.com/x/Alamofire", from: "5.0.0")\n'
    )
    assert [p.name for p in scan_deps(tmp_path)] == ["swift-log", "Alamofire"]
