"""Chunk a docling-parsed markdown file into header-tagged, token-bounded chunks.

Run: python3 chunk_docling_md.py <path_to_docling.md> [--max-tokens 512]

Rules implemented:
  - The FIRST "## ..." line in the document is the paper title. Everything
    between it and the "## Abstract" header (author names, affiliations,
    emails) is dropped entirely -- it is never embedded.
  - From "## Abstract" onward, EVERY "## ..." line is a candidate section
    header -- no leading number required (this was wrong in an earlier
    version and is now removed; "Abstract" itself has no number).
  - A header is still filtered out as noise -- WITHOUT requiring a number --
    if the block immediately following it is an image/formula/caption/table
    rather than real prose. This is how the OCR-injected pseudo-headings
    ("## Scaled Dot-Product Attention", "## Input-Input Layer5" -- both are
    mis-parsed figure-internal labels) get ignored: a genuine section always
    opens with at least a sentence of real content, these don't.
  - Numbered headers ("## 3 Model Architecture", "## 3.2 Attention") still
    determine section vs subsection by their numbering depth. Unnumbered
    headers ("Abstract", "References") are always treated as depth-1
    (top-level) sections.
  - EVERY chunk's embedded text starts with the paper title and section
    path, so the paper name itself is part of what gets embedded, not just
    metadata sitting next to it.
  - Within a section, paragraphs are stuffed until the next one would push
    the chunk over `max_tokens`; a new chunk then starts with the SAME
    section header.
  - Inline citations like "[12]" / "[2, 19]" are stripped.
  - Paragraphs that are pure footnote markers -- starting with *, †, or ‡
    (the "Equal contribution" / "Work performed while at ..." notes) -- are
    dropped outright.
  - Images, undecoded formulas, figure/table captions, and markdown tables
    are dropped. If that drop leaves the previous paragraph mid-sentence
    (it doesn't end in . ! ? or :), the next paragraph is glued onto it
    instead of starting a new paragraph unit.
  - Chunking STOPS entirely at the first "References"/"Bibliography" header
    -- nothing after it is processed, not even if real section-like content
    somehow follows.
"""

import argparse
import json
import re
from pathlib import Path

HEADER_RE = re.compile(r"^#{1,6}\s+(\S.*)$")
NUMBER_PREFIX_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")

IMAGE_MARKER_RE = re.compile(r"^(<!--\s*image\s*-->|!\[.*?\]\(.*?\))$", re.IGNORECASE)
FORMULA_MARKER_RE = re.compile(r"^<!--\s*formula-not-decoded\s*-->$", re.IGNORECASE)
CAPTION_RE = re.compile(r"^(Figure|Table)\s+\d+\s*:", re.IGNORECASE)
TABLE_ROW_RE = re.compile(r"^\|")

CITATION_RE = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")
WHITESPACE_RE = re.compile(r"\s{2,}")

STOP_SECTION_TITLES = {"references", "bibliography"}
DROPPED_KINDS = {"image", "formula", "caption", "table", "junk"}
KNOWN_UNNUMBERED_SECTIONS = {
    "abstract", "limitations", "acknowledgements", "acknowledgments",
    "appendix", "related work", "ethics statement", "broader impact",
}

_WORD_RE = re.compile(r"\S+")


def count_tokens(text: str) -> int:
    """Rough approximation: 1.3 tokens per whitespace-separated word."""
    return int(len(_WORD_RE.findall(text)) * 1.3)


def is_junk_start(text: str) -> bool:
    """Drop anything that doesn't start with a letter -- catches stray
    reference/URL lines like '13 https://...' or '[14 https://...]', plus
    footnote markers (*, †, ‡) -- while tolerating a markdown list bullet
    ('- ...') so real bulleted content isn't lost."""
    stripped = re.sub(r"^-\s+", "", text)
    return not stripped[:1].isalpha()


def is_real_section(title: str):
    """Return (accept, clean_title, depth). Numbered headers are always
    accepted; unnumbered ones only if they're a known paper section --
    otherwise it's noise (e.g. a mis-tagged author name)."""
    m = NUMBER_PREFIX_RE.match(title)
    if m:
        numbering, clean_title = m.groups()
        return True, clean_title, numbering.count(".") + 1
    if title.lower() in KNOWN_UNNUMBERED_SECTIONS:
        return True, title, 1
    return False, title, 1


def strip_citations(text: str) -> str:
    text = CITATION_RE.sub("", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def ends_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?", ":"))


def split_blocks(md_text: str) -> list[str]:
    raw_blocks = re.split(r"\n\s*\n", md_text)
    return [" ".join(b.split("\n")).strip() for b in raw_blocks if b.strip()]


def classify_block(block: str) -> str:
    if HEADER_RE.match(block):
        return "header"
    if IMAGE_MARKER_RE.match(block):
        return "image"
    if FORMULA_MARKER_RE.match(block):
        return "formula"
    if CAPTION_RE.match(block):
        return "caption"
    if TABLE_ROW_RE.match(block):
        return "table"
    if is_junk_start(block):
        return "junk"
    return "content"


def make_chunk(paper_title, section, paragraphs, chunks):
    """Turn the buffered paragraphs into one chunk and append it to `chunks`.
    Mutates `chunks` directly (lists are mutable) -- no nonlocal needed."""
    if not paragraphs:
        return
    body = "\n\n".join(paragraphs)
    text = f"paper : {paper_title}\nsection : {section}\n\n{body}"
    chunks.append({
        "chunk_id": len(chunks),
        "paper_title": paper_title,
        "section": section,
        "text": text,
        "token_count": count_tokens(text),
    })


def chunk_docling_markdown(md_text: str, max_tokens: int = 512) -> list[dict]:
    blocks = split_blocks(md_text)

    chunks = []
    paper_title = None
    in_frontmatter = False
    top_section = None       # current top-level section, for subsection paths
    current_section = None
    buffer = []               # paragraphs waiting to be flushed into a chunk
    buffer_tokens = 0

    i = 0
    while i < len(blocks):
        block = blocks[i]
        kind = classify_block(block)

        if kind == "header":
            title = HEADER_RE.match(block).group(1).strip()

            if paper_title is None:
                paper_title = title
                in_frontmatter = True
                i += 1
                continue

            if in_frontmatter:
                if title.lower() == "abstract":
                    in_frontmatter = False
                else:
                    i += 1  # still in the author/affiliation block -- drop
                    continue

            if title.lower() in STOP_SECTION_TITLES:
                make_chunk(paper_title, current_section, buffer, chunks)
                break  # hard stop: nothing after References is processed

            accept, clean_title, depth = is_real_section(title)
            if not accept:
                i += 1  # noise header (e.g. a mis-tagged author name)
                continue

            make_chunk(paper_title, current_section, buffer, chunks)
            buffer, buffer_tokens = [], 0
            if depth == 1:
                top_section = clean_title
                current_section = clean_title
            else:
                current_section = f"{top_section} : {clean_title}"
            i += 1
            continue

        if in_frontmatter or paper_title is None:
            i += 1
            continue

        if kind in DROPPED_KINDS:
            i += 1
            continue

        # kind == "content"
        text = strip_citations(block)
        if not text:
            i += 1
            continue

        if buffer and not ends_sentence(buffer[-1]):
            buffer[-1] = f"{buffer[-1]} {text}"
            buffer_tokens += count_tokens(text)
            i += 1
            continue

        new_tokens = count_tokens(text)
        if buffer and buffer_tokens + new_tokens > max_tokens:
            make_chunk(paper_title, current_section, buffer, chunks)
            buffer, buffer_tokens = [], 0
        buffer.append(text)
        buffer_tokens += new_tokens
        i += 1

    make_chunk(paper_title, current_section, buffer, chunks)
    return chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("md_path")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--out", default="test/chunks.json")
    args = parser.parse_args()

    md_text = Path(args.md_path).read_text(encoding="utf-8")
    chunks = chunk_docling_markdown(md_text, max_tokens=args.max_tokens)

    Path(args.out).write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(chunks)} chunks written to {args.out}")
    for c in chunks:
        print(f"  [{c['chunk_id']:>2}] ({c['token_count']:>3} tok)  {c['section']}")