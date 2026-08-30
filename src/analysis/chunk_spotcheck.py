from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chunking.chunk import chunk_id_for_unit


WINDOW_RADIUS_CHARS = 1400
ARTICLE_CONFIG_ID = "hier_article"
BASE_HIER_CONFIG_ID = "hier_paragraph"

VISIBLE_CONFIGS = [
    ("fs_64_12", "fixed_size", "Fixed 64 / 12"),
    ("fs_256_50", "fixed_size", "Fixed 256 / 50"),
    ("sem_50_64", "semantic", "Semantic 50 / 64"),
    ("sem_50_128", "semantic", "Semantic 50 / 128"),
    ("hier_paragraph", "hierarchical", "Hierarchical paragraph"),
]


def write_chunk_spotcheck_outputs(
    *,
    config_path: Path,
    canonical_texts_path: Path,
    chunks_dir: Path,
    xref_graph_path: Path,
    output_json_path: Path,
    output_html_path: Path,
    window_radius_chars: int = WINDOW_RADIUS_CHARS,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    canonical_texts = json.loads(canonical_texts_path.read_text(encoding="utf-8"))
    xref_graph = json.loads(xref_graph_path.read_text(encoding="utf-8"))
    chunk_sets = load_chunk_sets(chunks_dir)
    edges_by_source = build_edges_by_source(xref_graph)
    chunks_by_id = {
        chunk["chunk_id"]: chunk
        for chunks in chunk_sets.values()
        for chunk in chunks
    }

    documents = []
    for document_code in config["corpus"]["document_codes"]:
        documents.append(
            build_document_spotcheck(
                document_code=document_code,
                canonical_text=canonical_texts[document_code]["text"],
                chunk_sets=chunk_sets,
                chunks_by_id=chunks_by_id,
                edges_by_source=edges_by_source,
                window_radius_chars=window_radius_chars,
            )
        )

    report = {
        "window_radius_chars": window_radius_chars,
        "selection_rule": (
            "For each document, select the hier_paragraph chunk nearest the "
            "document midpoint among chunks with both a parent article and at "
            "least one direct outgoing xref; fall back to xref-bearing chunks, "
            "then parent-bearing chunks, then the nearest hier_paragraph chunk."
        ),
        "visible_configs": [
            {"config_id": config_id, "label": label}
            for config_id, _, label in VISIBLE_CONFIGS
        ],
        "documents": documents,
    }
    html_report = strip_trailing_whitespace(render_html(report))
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output_html_path.write_text(html_report, encoding="utf-8")
    return report


def load_chunk_sets(chunks_dir: Path) -> dict[str, list[dict[str, Any]]]:
    chunk_sets: dict[str, list[dict[str, Any]]] = {}
    for config_id, strategy, _ in VISIBLE_CONFIGS + [
        (ARTICLE_CONFIG_ID, "hierarchical", "Hierarchical article")
    ]:
        path = chunks_dir / strategy / f"{config_id}.json"
        chunk_sets[config_id] = json.loads(path.read_text(encoding="utf-8"))
    return chunk_sets


def build_edges_by_source(xref_graph: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    edges_by_source: dict[str, list[dict[str, Any]]] = {}
    for edge in xref_graph["edges"]:
        edges_by_source.setdefault(edge["source_unit_id"], []).append(edge)
    return edges_by_source


def build_document_spotcheck(
    *,
    document_code: str,
    canonical_text: str,
    chunk_sets: dict[str, list[dict[str, Any]]],
    chunks_by_id: dict[str, dict[str, Any]],
    edges_by_source: dict[str, list[dict[str, Any]]],
    window_radius_chars: int,
) -> dict[str, Any]:
    base_chunk = select_anchor_chunk(
        document_code=document_code,
        canonical_text_length=len(canonical_text),
        base_chunks=chunk_sets[BASE_HIER_CONFIG_ID],
        edges_by_source=edges_by_source,
    )
    anchor_start, anchor_end = chunk_span(base_chunk)
    anchor_center = (anchor_start + anchor_end) // 2
    window_start = max(0, anchor_center - window_radius_chars)
    window_end = min(len(canonical_text), anchor_center + window_radius_chars)

    return {
        "document_code": document_code,
        "canonical_char_length": len(canonical_text),
        "window": {
            "start": window_start,
            "end": window_end,
            "text": canonical_text[window_start:window_end],
        },
        "anchor": summarize_chunk(base_chunk),
        "tracks": [
            build_track(
                config_id=config_id,
                label=label,
                chunks=chunk_sets[config_id],
                document_code=document_code,
                window_start=window_start,
                window_end=window_end,
                anchor_center=anchor_center,
            )
            for config_id, _, label in VISIBLE_CONFIGS
        ],
        "enrichment": build_enrichment_summary(
            base_chunk=base_chunk,
            chunks_by_id=chunks_by_id,
            edges_by_source=edges_by_source,
            window_start=window_start,
            window_end=window_end,
        ),
    }


def select_anchor_chunk(
    *,
    document_code: str,
    canonical_text_length: int,
    base_chunks: list[dict[str, Any]],
    edges_by_source: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    candidates = [
        chunk
        for chunk in base_chunks
        if chunk["metadata"]["document_code"] == document_code
    ]
    with_parent_and_xrefs = [
        chunk
        for chunk in candidates
        if chunk["metadata"].get("parent_chunk_id")
        and outgoing_edges_for_chunk(chunk, edges_by_source)
    ]
    with_xrefs = [
        chunk
        for chunk in candidates
        if outgoing_edges_for_chunk(chunk, edges_by_source)
    ]
    with_parent = [
        chunk
        for chunk in candidates
        if chunk["metadata"].get("parent_chunk_id")
    ]
    pool = with_parent_and_xrefs or with_xrefs or with_parent or candidates
    midpoint = canonical_text_length // 2
    return min(
        pool,
        key=lambda chunk: (
            abs(chunk_center(chunk) - midpoint),
            chunk["metadata"].get("legal_order", 0),
            chunk["chunk_id"],
        ),
    )


def build_track(
    *,
    config_id: str,
    label: str,
    chunks: list[dict[str, Any]],
    document_code: str,
    window_start: int,
    window_end: int,
    anchor_center: int,
) -> dict[str, Any]:
    visible_chunks = [
        chunk
        for chunk in chunks
        if chunk["metadata"]["document_code"] == document_code
        and intervals_overlap(chunk_span(chunk), (window_start, window_end))
    ]
    containing_anchor = [
        chunk
        for chunk in visible_chunks
        if chunk_span(chunk)[0] <= anchor_center <= chunk_span(chunk)[1]
    ]
    anchor_chunk_id = ""
    if containing_anchor:
        anchor_chunk_id = min(containing_anchor, key=chunk_span_length)["chunk_id"]

    return {
        "config_id": config_id,
        "label": label,
        "visible_chunk_count": len(visible_chunks),
        "anchor_chunk_id": anchor_chunk_id,
        "chunks": [
            summarize_visible_chunk(
                chunk,
                window_start=window_start,
                window_end=window_end,
                anchor_chunk_id=anchor_chunk_id,
            )
            for chunk in visible_chunks
        ],
    }


def build_enrichment_summary(
    *,
    base_chunk: dict[str, Any],
    chunks_by_id: dict[str, dict[str, Any]],
    edges_by_source: dict[str, list[dict[str, Any]]],
    window_start: int,
    window_end: int,
) -> dict[str, Any]:
    parent_chunk = None
    parent_chunk_id = base_chunk["metadata"].get("parent_chunk_id")
    if parent_chunk_id:
        parent_chunk = chunks_by_id.get(parent_chunk_id)

    xrefs = []
    seen_targets: set[str] = set()
    for edge in outgoing_edges_for_chunk(base_chunk, edges_by_source):
        target_chunk_id = chunk_id_for_unit(
            ARTICLE_CONFIG_ID,
            edge["target_unit_id_normalized"],
        )
        target_chunk = chunks_by_id.get(target_chunk_id)
        if not target_chunk or target_chunk_id in seen_targets:
            continue
        seen_targets.add(target_chunk_id)
        target_start, target_end = chunk_span(target_chunk)
        xrefs.append(
            {
                "edge": edge,
                "target_chunk": summarize_chunk(target_chunk),
                "target_in_window": intervals_overlap(
                    (target_start, target_end),
                    (window_start, window_end),
                ),
            }
        )

    return {
        "base_chunk": summarize_chunk(base_chunk),
        "parent_chunk": summarize_chunk(parent_chunk) if parent_chunk else None,
        "xref_count": len(xrefs),
        "xref_targets": xrefs,
    }


def outgoing_edges_for_chunk(
    chunk: dict[str, Any],
    edges_by_source: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for source_unit_id in source_unit_candidates(chunk):
        edges.extend(edges_by_source.get(source_unit_id, []))
    return edges


def source_unit_candidates(chunk: dict[str, Any]) -> list[str]:
    # Mirror context assembly's xref-source rule.
    metadata = chunk["metadata"]
    seen: set[str] = set()
    candidates = []
    for unit_id in metadata["source_unit_ids"]:
        if unit_id not in seen:
            seen.add(unit_id)
            candidates.append(unit_id)
    return candidates


def summarize_visible_chunk(
    chunk: dict[str, Any],
    *,
    window_start: int,
    window_end: int,
    anchor_chunk_id: str,
) -> dict[str, Any]:
    start, end = chunk_span(chunk)
    window_length = window_end - window_start
    visible_start = max(start, window_start)
    visible_end = min(end, window_end)
    return {
        **summarize_chunk(chunk),
        "is_anchor_chunk": chunk["chunk_id"] == anchor_chunk_id,
        "left_pct": percentage(visible_start - window_start, window_length),
        "width_pct": percentage(visible_end - visible_start, window_length),
    }


def summarize_chunk(chunk: dict[str, Any] | None) -> dict[str, Any]:
    if chunk is None:
        return {}
    start, end = chunk_span(chunk)
    return {
        "chunk_id": chunk["chunk_id"],
        "config_id": chunk["config_id"],
        "strategy": chunk["strategy"],
        "text": chunk["text"],
        "token_count": chunk["token_count"],
        "canonical_start": start,
        "canonical_end": end,
        "source_unit_ids": chunk["metadata"].get("source_unit_ids", []),
        "answer_unit_ids": chunk["metadata"].get("answer_unit_ids", []),
        "parent_chunk_id": chunk["metadata"].get("parent_chunk_id"),
        "hierarchy_level": chunk["metadata"].get("hierarchy_level"),
    }


def chunk_span(chunk: dict[str, Any]) -> tuple[int, int]:
    starts = [interval[0] for interval in chunk["canonical_intervals"]]
    ends = [interval[1] for interval in chunk["canonical_intervals"]]
    return min(starts), max(ends)


def chunk_center(chunk: dict[str, Any]) -> int:
    start, end = chunk_span(chunk)
    return (start + end) // 2


def chunk_span_length(chunk: dict[str, Any]) -> int:
    start, end = chunk_span(chunk)
    return end - start


def intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def percentage(value: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return max(0.0, min(100.0, value * 100 / total))


def render_html(report: dict[str, Any]) -> str:
    nav = "\n".join(
        f'<a href="#doc-{escape_attr(doc["document_code"])}">{escape(doc["document_code"])}</a>'
        for doc in report["documents"]
    )
    sections = "\n".join(render_document_section(doc) for doc in report["documents"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Q4EU Chunk Spot Checks</title>
  <style>
    :root {{
      --ink: #17202a;
      --muted: #5d6673;
      --line: #d7dde5;
      --paper: #ffffff;
      --bg: #f6f7f9;
      --accent: #1565c0;
      --fixed: #3478c7;
      --semantic: #228b6b;
      --hier: #8a5a0a;
      --parent: #7b4dbb;
      --xref: #b84a62;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 24px 28px 18px;
      background: #102033;
      color: #fff;
    }}
    header h1 {{ margin: 0 0 8px; font-size: 26px; }}
    header p {{ margin: 0; max-width: 1000px; color: #d7e2ef; }}
    nav {{
      position: sticky;
      top: 0;
      z-index: 5;
      display: flex;
      gap: 8px;
      padding: 10px 28px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,.96);
      backdrop-filter: blur(8px);
    }}
    nav a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 650;
      padding: 4px 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
    }}
    main {{ padding: 20px 28px 48px; }}
    section.doc {{
      margin: 0 0 28px;
      padding: 20px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
    }}
    h2 {{ margin: 0 0 8px; font-size: 22px; }}
    h3 {{ margin: 20px 0 10px; font-size: 17px; }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 10px 0 16px;
      color: var(--muted);
      font-size: 13px;
    }}
    .pill {{
      padding: 3px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fafbfc;
    }}
    .excerpt {{
      margin: 10px 0 18px;
      padding: 14px;
      max-height: 280px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfe;
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
    }}
    mark {{
      padding: 0 2px;
      background: #fff2a8;
      border-bottom: 2px solid #c58a00;
    }}
    .track-card {{
      margin: 12px 0;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    .track-card h4 {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin: 0 0 10px;
      font-size: 15px;
    }}
    .track-count {{ color: var(--muted); font-weight: 500; font-size: 12px; }}
    .chunk-row {{
      display: grid;
      grid-template-columns: minmax(220px, 360px) 1fr;
      gap: 12px;
      align-items: center;
      margin: 6px 0;
    }}
    .chunk-label {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    .barline {{
      position: relative;
      height: 16px;
      border: 1px solid #e2e6ec;
      border-radius: 5px;
      background: linear-gradient(90deg, #f8fafc, #eef2f6);
      overflow: hidden;
    }}
    .bar {{
      position: absolute;
      top: 2px;
      height: 10px;
      min-width: 2px;
      border-radius: 4px;
      background: var(--fixed);
    }}
    .semantic .bar {{ background: var(--semantic); }}
    .hierarchical .bar {{ background: var(--hier); }}
    .bar.anchor {{
      outline: 2px solid #111827;
      outline-offset: 1px;
    }}
    details {{
      margin: 7px 0;
      border-top: 1px solid #eef1f5;
      padding-top: 7px;
    }}
    summary {{ cursor: pointer; color: var(--accent); font-weight: 650; }}
    pre {{
      margin: 8px 0 0;
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      background: #f7f9fb;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      max-height: 220px;
      overflow: auto;
    }}
    .enriched {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
    }}
    .enrich-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }}
    .enrich-card.parent {{ border-color: #c6b6e4; }}
    .enrich-card.xref {{ border-color: #e7b6c2; }}
    .enrich-card h4 {{ margin: 0 0 6px; font-size: 15px; }}
    .small {{ color: var(--muted); font-size: 12px; }}
    .empty {{ color: var(--muted); font-style: italic; }}
  </style>
</head>
<body>
  <header>
    <h1>Q4EU Chunk Spot Checks</h1>
    <p>{escape(report["selection_rule"])} Window radius: {report["window_radius_chars"]} canonical characters. Each document uses the same window across all strategies.</p>
  </header>
  <nav>{nav}</nav>
  <main>{sections}</main>
</body>
</html>
"""


def render_document_section(doc: dict[str, Any]) -> str:
    code = doc["document_code"]
    anchor = doc["anchor"]
    window = doc["window"]
    tracks = "\n".join(render_track(track) for track in doc["tracks"])
    excerpt = render_excerpt(
        window["text"],
        window_start=window["start"],
        anchor_start=anchor["canonical_start"],
        anchor_end=anchor["canonical_end"],
    )
    return f"""
<section class="doc" id="doc-{escape_attr(code)}">
  <h2>{escape(code)}</h2>
  <div class="meta">
    <span class="pill">window {window["start"]}-{window["end"]}</span>
    <span class="pill">anchor {escape(anchor["chunk_id"])}</span>
    <span class="pill">{anchor["token_count"]} tokens</span>
    <span class="pill">answer units: {escape(", ".join(anchor["answer_unit_ids"]))}</span>
  </div>
  <h3>Canonical Text Around Anchor</h3>
  <div class="excerpt">{excerpt}</div>
  <h3>Chunk Boundary Tracks</h3>
  {tracks}
  <h3>Hierarchical Enrichment Around Anchor</h3>
  {render_enrichment(doc["enrichment"])}
</section>
"""


def render_excerpt(
    text: str,
    *,
    window_start: int,
    anchor_start: int,
    anchor_end: int,
) -> str:
    start = max(0, anchor_start - window_start)
    end = min(len(text), anchor_end - window_start)
    if start >= end:
        return escape(text)
    return (
        escape(text[:start])
        + "<mark>"
        + escape(text[start:end])
        + "</mark>"
        + escape(text[end:])
    )


def render_track(track: dict[str, Any]) -> str:
    rows = "\n".join(render_chunk_row(chunk) for chunk in track["chunks"])
    details = "\n".join(render_chunk_details(chunk) for chunk in track["chunks"])
    empty = '<div class="empty">No chunks overlap this window.</div>' if not rows else ""
    return f"""
<div class="track-card {escape_attr(strategy_class(track["config_id"]))}">
  <h4>{escape(track["label"])} <span class="track-count">{track["visible_chunk_count"]} overlapping chunks; anchor chunk: {escape(track["anchor_chunk_id"] or "none")}</span></h4>
  {rows or empty}
  {details}
</div>
"""


def render_chunk_row(chunk: dict[str, Any]) -> str:
    anchor_class = " anchor" if chunk["is_anchor_chunk"] else ""
    title = (
        f'{chunk["chunk_id"]} | {chunk["token_count"]} tokens | '
        f'{", ".join(chunk["answer_unit_ids"])}'
    )
    return f"""
<div class="chunk-row">
  <div class="chunk-label" title="{escape_attr(title)}">{escape(chunk["chunk_id"])}</div>
  <div class="barline">
    <div class="bar{anchor_class}" title="{escape_attr(title)}" style="left:{chunk["left_pct"]:.3f}%; width:{chunk["width_pct"]:.3f}%"></div>
  </div>
</div>
"""


def render_chunk_details(chunk: dict[str, Any]) -> str:
    return f"""
<details>
  <summary>{escape(chunk["chunk_id"])} | {chunk["token_count"]} tokens | {escape(", ".join(chunk["answer_unit_ids"]))}</summary>
  <pre>{escape(chunk["text"])}</pre>
</details>
"""


def render_enrichment(enrichment: dict[str, Any]) -> str:
    base = render_enrichment_card("Base paragraph", enrichment["base_chunk"], "base")
    parent = (
        render_enrichment_card("Parent article", enrichment["parent_chunk"], "parent")
        if enrichment["parent_chunk"]
        else '<div class="enrich-card parent"><h4>Parent article</h4><div class="empty">No parent chunk for this anchor.</div></div>'
    )
    xrefs = "\n".join(render_xref_card(item) for item in enrichment["xref_targets"])
    if not xrefs:
        xrefs = '<div class="enrich-card xref"><h4>Xref targets</h4><div class="empty">No direct xref targets for this anchor.</div></div>'
    return f'<div class="enriched">{base}{parent}{xrefs}</div>'


def render_enrichment_card(title: str, chunk: dict[str, Any], css_class: str) -> str:
    return f"""
<div class="enrich-card {escape_attr(css_class)}">
  <h4>{escape(title)}</h4>
  <div class="small">{escape(chunk["chunk_id"])} | {chunk["token_count"]} tokens | {escape(", ".join(chunk["answer_unit_ids"]))}</div>
  <pre>{escape(chunk["text"])}</pre>
</div>
"""


def render_xref_card(item: dict[str, Any]) -> str:
    edge = item["edge"]
    chunk = item["target_chunk"]
    location = "inside window" if item["target_in_window"] else "outside window"
    return f"""
<div class="enrich-card xref">
  <h4>Xref target: {escape(edge["raw_match"])}</h4>
  <div class="small">source {escape(edge["source_unit_id"])} -> {escape(edge["target_unit_id_normalized"])} ({escape(location)})</div>
  <div class="small">{escape(chunk["chunk_id"])} | {chunk["token_count"]} tokens</div>
  <pre>{escape(chunk["text"])}</pre>
</div>
"""


def strategy_class(config_id: str) -> str:
    if config_id.startswith("sem_"):
        return "semantic"
    if config_id.startswith("hier_"):
        return "hierarchical"
    return "fixed"


def escape(value: Any) -> str:
    return html.escape(str(value), quote=False)


def escape_attr(value: Any) -> str:
    return html.escape(str(value), quote=True)


def strip_trailing_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate chunk spot-check report.")
    parser.add_argument("--config-path", type=Path, default=Path("config/config.yaml"))
    parser.add_argument(
        "--canonical-texts-path",
        type=Path,
        default=Path("data/parsed/canonical_texts.json"),
    )
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument(
        "--xref-graph-path",
        type=Path,
        default=Path("data/parsed/xref_graph.json"),
    )
    parser.add_argument(
        "--output-json-path",
        type=Path,
        default=Path("data/analysis/chunk_spotcheck.json"),
    )
    parser.add_argument(
        "--output-html-path",
        type=Path,
        default=Path("reports/diagnostics/chunk_spotcheck.html"),
    )
    parser.add_argument(
        "--window-radius-chars",
        type=int,
        default=WINDOW_RADIUS_CHARS,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = write_chunk_spotcheck_outputs(
        config_path=args.config_path,
        canonical_texts_path=args.canonical_texts_path,
        chunks_dir=args.chunks_dir,
        xref_graph_path=args.xref_graph_path,
        output_json_path=args.output_json_path,
        output_html_path=args.output_html_path,
        window_radius_chars=args.window_radius_chars,
    )
    print(
        "Chunk spot-check report complete: "
        f"{len(report['documents'])} documents, "
        f"{args.output_html_path}."
    )


if __name__ == "__main__":
    main()
