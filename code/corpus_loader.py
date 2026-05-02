"""
Corpus Loader — reads and chunks the markdown support corpus.

Walks data/{hackerrank,claude,visa}/ recursively, parses YAML frontmatter,
and produces Document objects ready for embedding.
"""

import os
import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Document:
    """A single chunk of a support article."""
    doc_id: str
    title: str
    domain: str  # "hackerrank", "claude", "visa"
    breadcrumbs: list[str] = field(default_factory=list)
    content: str = ""
    source_path: str = ""
    source_url: str = ""
    chunk_index: int = 0


# Regex to split YAML frontmatter from markdown body
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and body from a markdown file."""
    match = FRONTMATTER_RE.match(text)
    if match:
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        body = text[match.end():]
    else:
        meta = {}
        body = text
    return meta, body


def detect_domain(filepath: str, data_root: str) -> str:
    """Infer domain from the file's path relative to data/."""
    rel = os.path.relpath(filepath, data_root)
    parts = Path(rel).parts
    if parts:
        first = parts[0].lower()
        if first in ("hackerrank", "claude", "visa"):
            return first
    return "unknown"


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    """
    Split text into chunks by paragraph boundaries, respecting max_chars.
    Uses overlap to maintain context between chunks.
    """
    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph exceeds max, save current and start new
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current.strip())
            # Keep overlap from end of current chunk
            if overlap > 0 and len(current) > overlap:
                current = current[-overlap:] + "\n\n" + para
            else:
                current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    # If no chunks were created, return the whole text as one chunk
    if not chunks and text.strip():
        chunks = [text.strip()]

    return chunks


def load_corpus(data_dir: str) -> list[Document]:
    """
    Load and chunk all markdown files from the data directory.

    Returns a list of Document objects, each representing a chunk of
    a support article with its metadata.
    """
    documents = []
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    md_files = sorted(data_path.rglob("*.md"))
    print(f"  Found {len(md_files)} markdown files in {data_dir}")

    for filepath in md_files:
        filepath_str = str(filepath)

        # Skip index files — they're just tables of contents
        if filepath.name == "index.md":
            continue

        try:
            text = filepath.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as e:
            print(f"  Warning: Could not read {filepath}: {e}")
            continue

        meta, body = parse_frontmatter(text)
        domain = detect_domain(filepath_str, data_dir)
        title = meta.get("title", filepath.stem)
        breadcrumbs = meta.get("breadcrumbs", [])
        if isinstance(breadcrumbs, str):
            breadcrumbs = [breadcrumbs]
        source_url = meta.get("source_url", "")

        # Chunk the body
        chunks = chunk_text(body, max_chars=1500, overlap=200)

        for i, chunk_content in enumerate(chunks):
            doc = Document(
                doc_id=f"{domain}:{filepath.stem}:{i}",
                title=title,
                domain=domain,
                breadcrumbs=breadcrumbs,
                content=chunk_content,
                source_path=filepath_str,
                source_url=source_url,
                chunk_index=i,
            )
            documents.append(doc)

    print(f"  Created {len(documents)} document chunks from {len(md_files)} files")

    # Summary by domain
    domain_counts = {}
    for doc in documents:
        domain_counts[doc.domain] = domain_counts.get(doc.domain, 0) + 1
    for domain, count in sorted(domain_counts.items()):
        print(f"    {domain}: {count} chunks")

    return documents
