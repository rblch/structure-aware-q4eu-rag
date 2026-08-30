from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


DOCUMENTS = {
    "B": "[Law_EU]qa_overview/oke/documents/bruss.akn",
    "RI": "[Law_EU]qa_overview/oke/documents/rome_i.akn",
    "RII": "[Law_EU]qa_overview/oke/documents/rome_ii.akn",
    "G": "[Law_EU]qa_overview/oke/documents/gdpr.akn",
    "E": "[Law_EU]qa_overview/oke/documents/eidas.akn",
    "W": "[Law_EU]qa_overview/oke/documents/warrant.html",
}

EVALUATE_PATH = "[Law_EU]qa_overview/oke/evaluate.py"
LICENSE_CANDIDATES = ("LICENSE", "LICENSE.md", "COPYING")


@dataclass(frozen=True)
class SourceFile:
    logical_name: str
    repo_path: str
    local_path: Path


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def raw_url(repo: str, commit: str, path: str) -> str:
    quoted_path = quote(path, safe="/")
    return f"https://raw.githubusercontent.com/{repo}/{commit}/{quoted_path}"


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "q4eu-rag-importer"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def write_source_file(repo: str, commit: str, source_file: SourceFile) -> dict:
    url = raw_url(repo, commit, source_file.repo_path)
    content = fetch_bytes(url)
    source_file.local_path.parent.mkdir(parents=True, exist_ok=True)
    source_file.local_path.write_bytes(content)
    return {
        "logical_name": source_file.logical_name,
        "repo_path": source_file.repo_path,
        "local_path": source_file.local_path.as_posix(),
        "url": url,
        "bytes": len(content),
        "sha256": sha256_bytes(content),
    }


def fetch_license(repo: str, commit: str, output_dir: Path) -> dict:
    attempts = []
    for candidate in LICENSE_CANDIDATES:
        url = raw_url(repo, commit, candidate)
        attempts.append(url)
        try:
            content = fetch_bytes(url)
        except HTTPError as exc:
            if exc.code == 404:
                continue
            raise

        local_path = output_dir / candidate
        local_path.write_bytes(content)
        return {
            "status": "fetched",
            "repo_path": candidate,
            "local_path": local_path.as_posix(),
            "url": url,
            "bytes": len(content),
            "sha256": sha256_bytes(content),
        }

    return {"status": "not_found", "attempted_urls": attempts}


def validate_commit(commit: str) -> None:
    if not commit or commit == "REQUIRED":
        raise SystemExit("A pinned source commit is required.")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise SystemExit("The source commit must be a 40-character lowercase SHA.")


def source_files(output_dir: Path) -> list[SourceFile]:
    files = [
        SourceFile("evaluate.py", EVALUATE_PATH, output_dir / "evaluate.py"),
    ]
    files.extend(
        SourceFile(code, repo_path, output_dir / "documents" / Path(repo_path).name)
        for code, repo_path in DOCUMENTS.items()
    )
    return files


def assert_snapshot_matches_config(
    source_metadata_path: Path,
    config: dict,
) -> None:
    metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    pinned_commit = config["dataset"]["source_commit"]
    if metadata["source_commit"] != pinned_commit:
        raise ValueError(
            f"Snapshot commit {metadata['source_commit']} does not match the "
            f"pinned dataset.source_commit {pinned_commit} in config.yaml."
        )
    mismatched = [
        file_info["local_path"]
        for file_info in metadata["files"]
        if not Path(file_info["local_path"]).exists()
        or sha256_bytes(Path(file_info["local_path"]).read_bytes())
        != file_info["sha256"]
    ]
    if mismatched:
        raise ValueError(
            "Raw source files are missing or differ from the hashes recorded "
            f"at fetch time: {mismatched}"
        )


def fetch_discolqa(repo: str, branch: str, commit: str, output_dir: Path) -> dict:
    validate_commit(commit)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [
        write_source_file(repo, commit, source_file)
        for source_file in source_files(output_dir)
    ]
    metadata = {
        "source_repo": repo,
        "source_branch": branch,
        "source_commit": commit,
        "import_timestamp_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "files": files,
        "license": fetch_license(repo, commit, output_dir),
        "excluded_upstream_assets": [
            "large DiscoLQA model files",
            "derived caches",
            "runtime outputs",
        ],
    }

    metadata_path = output_dir / "source_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch pinned DiscoLQA Q4EU inputs.")
    parser.add_argument(
        "--repo",
        default="Francesco-Sovrano/DiscoLQA",
        help="GitHub repository in owner/name form.",
    )
    parser.add_argument("--branch", default="main", help="Recorded source branch.")
    parser.add_argument("--commit", required=True, help="Pinned upstream commit SHA.")
    parser.add_argument(
        "--output-dir",
        default="data/raw/discolqa",
        type=Path,
        help="Output directory for raw DiscoLQA inputs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = fetch_discolqa(args.repo, args.branch, args.commit, args.output_dir)
    print(
        "Fetched "
        f"{len(metadata['files'])} source files from {metadata['source_repo']} "
        f"at {metadata['source_commit']}."
    )


if __name__ == "__main__":
    main()

