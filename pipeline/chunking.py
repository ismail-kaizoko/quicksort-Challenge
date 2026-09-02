import argparse
import json
import re
from pathlib import Path


HEADER_RE = re.compile(r"^#{1,6}\s+(\S.*)$")
NUMBER_PREFIX_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")

IMAGE_MARKER_RE = re.compile(r"^<!--\s*image\s*-->$", re.IGNORECASE)
FORMULA_MARKER_RE = re.compile(r"^<!--\s*formula-not-decoded\s*-->$", re.IGNORECASE)
CAPTION_RE = re.compile(r"^(Figure|Table)\s+\d+\s*:", re.IGNORECASE)
TABLE_ROW_RE = re.compile(r"^\|")
FOOTNOTE_MARKER_RE = re.compile(r"^[\*\u2217\u2020\u2021]")  # *, ∗, †, ‡

CITATION_RE = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")
WHITESPACE_RE = re.compile(r"\s{2,}")

STOP_SECTION_TITLES = {"references", "bibliography"}
DROPPED_KINDS = {"image", "formula", "caption", "table", "footnote"}



_WORD_RE = re.compile(r"\S+")


def count_tokens(text: str) -> int:
    """Rough approximation: 1.3 tokens per whitespace-separated word."""
    return int(len(_WORD_RE.findall(text)) * 1.3)


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
    if FOOTNOTE_MARKER_RE.match(block):
        return "footnote"
    return "content"


def chunk_docling_markdown(md_text: str, max_tokens: int = 512) -> list[dict]:
    """Returns a list of {chunk_id, paper_title, section, text, token_count}."""
    blocks = split_blocks(md_text)

    chunks: list[dict] = []
    paper_title: str | None = None
    in_frontmatter = False  # True while skipping the author/affiliation block
    current_section_title: str | None = None
    current_section_path: str | None = None
    current_paragraphs: list[str] = []
    current_tokens = 0

    def flush():
        nonlocal current_paragraphs, current_tokens
        if current_paragraphs:
            body = "\n\n".join(current_paragraphs)
            embed_text = f"{paper_title}\nSection: {current_section_path}\n\n{body}"
            chunks.append({
                "chunk_id": len(chunks),
                "paper_title": paper_title,
                "section": current_section_path,
                "text": embed_text,
                "token_count": count_tokens(embed_text),
            })
        current_paragraphs = []
        current_tokens = 0

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
                    # fall through: process as the first real section below
                else:
                    i += 1  # still in the author/affiliation block -- drop
                    continue

            if title.lower() in STOP_SECTION_TITLES:
                flush()
                break  # hard stop: nothing after References is processed

            # Noise-header check: a real section never opens with a dropped
            # visual as its very first block.
            next_kind = classify_block(blocks[i + 1]) if i + 1 < len(blocks) else "content"
            if next_kind in {"image", "formula", "caption", "table"}:
                i += 1  # ignore this header line only, keep current section
                continue

            flush()
            m = NUMBER_PREFIX_RE.match(title)
            if m:
                numbering, real_title = m.groups()
                depth = numbering.count(".") + 1
            else:
                real_title, depth = title, 1

            if depth == 1:
                current_section_title = real_title
                current_section_path = real_title
            else:
                current_section_path = f"{current_section_title} : {real_title}"
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

        if current_paragraphs and not ends_sentence(current_paragraphs[-1]):
            current_paragraphs[-1] = f"{current_paragraphs[-1]} {text}"
            current_tokens += count_tokens(text)
            i += 1
            continue

        new_tokens = count_tokens(text)
        if current_paragraphs and current_tokens + new_tokens > max_tokens:
            flush()  # same current_section_path carries into the new chunk
        current_paragraphs.append(text)
        current_tokens += new_tokens
        i += 1

    flush()
    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("md_path")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--out", default="chunks.json")
    args = parser.parse_args()

    md_text = Path(args.md_path).read_text(encoding="utf-8")
    chunks = chunk_docling_markdown(md_text, max_tokens=args.max_tokens)

    Path(args.out).write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(chunks)} chunks written to {args.out}")
    for c in chunks:
        print(f"  [{c['chunk_id']:>2}] ({c['token_count']:>3} tok)  {c['section']}")