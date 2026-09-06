"""Batch-process a folder of raw PDFs into one combined chunks.json,
optionally captioning figures along the way.

Pipeline: PDF -> (docling) -> markdown in temp/ -> chunks -> one merged JSON.

Reuses the logic already built and tested in:
  - parse_pdfs.py         (build_converter, parse_pdf)
  - chunk_docling_md.py   (chunk_docling_markdown)
  - caption_figures.py    (find_figure_image_pairs, load_model,
                            caption_all_figures, build_figure_chunks)
All files must sit next to this script (or be on PYTHONPATH).

Run:
    python3 run_pipeline.py --input-dir ./pdfs --output-dir ./output
    python3 run_pipeline.py --input-dir ./pdfs --output-dir ./output --caption-figures
"""
import argparse
import json
from pathlib import Path

from parsing import build_converter, parse_pdf
from chunking import chunk_docling_markdown
from captioning import (
    find_figure_image_pairs,
    load_model,
    caption_all_figures,
    build_figure_chunks,
)


def process_paper(md_path: Path, max_tokens: int, caption_model, caption_tokenizer) -> list[dict]:
    """Chunk one paper's markdown; add captioned-figure chunks if a model was loaded."""
    md_text = md_path.read_text(encoding="utf-8")
    chunks = chunk_docling_markdown(md_text, max_tokens=max_tokens)

    if caption_model is not None:
        paper_title = chunks[0]["paper_title"] if chunks else md_path.stem
        pairs = find_figure_image_pairs(md_text, md_path.parent)
        figures = caption_all_figures(pairs, caption_model, caption_tokenizer)
        chunks += build_figure_chunks(figures, paper_title, next_chunk_id=len(chunks))

    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Folder of raw PDF papers")
    parser.add_argument("--output-dir", required=True, help="Folder to write temp/ and chunks.json into")
    parser.add_argument("--caption-figures", action="store_true",
                         help="Also caption figure images with moondream2 (slow on CPU)")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    temp_dir = output_dir / "temp"
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"no .pdf files found in {input_dir}")
        return

    # Parse every PDF to markdown first (docling's models are loaded once,
    # via build_converter(), and reused across all files -- same reasoning
    # as loading the caption model once below).
    converter = build_converter()
    md_files = []
    for pdf_path in pdf_files:
        print(f"parsing {pdf_path.name} ...")
        md_files.append(parse_pdf(converter, pdf_path, temp_dir))

    caption_model, caption_tokenizer = (load_model() if args.caption_figures else (None, None))

    all_chunks = []
    for md_path in md_files:
        print(f"chunking {md_path.name} ...")
        paper_chunks = process_paper(md_path, args.max_tokens, caption_model, caption_tokenizer)
        all_chunks.extend(paper_chunks)
        print(f"  -> {len(paper_chunks)} chunks")

    # chunk_id was assigned per-paper by chunk_docling_markdown -- renumber
    # sequentially now that everything lives in one combined file.
    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = i

    out_path = output_dir / "chunks.json"
    out_path.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(all_chunks)} total chunks across {len(pdf_files)} papers -> {out_path}")
    print(f"(intermediate markdown + images kept in {temp_dir} for inspection)")


if __name__ == "__main__":
    main()