"""Caption figures referenced in a docling markdown file and merge them into
an existing chunks.json.

Looks for the pattern:
    Figure N: <original caption text>

    ![Image](<path/to/image>.png)

For each match: resolves the image path, captions it with moondream2 using
the paper's own caption as context, and appends one new chunk to chunks.json.

Run:
    pip install transformers torch pillow einops timm --break-system-packages
    python3 caption_figures.py docling.md chunks.json --out chunks_enriched.json
"""
import argparse
import json
import re
from pathlib import Path
import torch

from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4


# "Figure 1: caption text" -- also matches "Table N:" the same way
CAPTION_RE = re.compile(r"^(Figure|Table)\s+(\d+)\s*:\s*(.+)$", re.IGNORECASE)
# "![Image](path/to/file.png)" -- docling's image reference syntax
IMAGE_REF_RE = re.compile(r"^!\[.*?\]\((.+?)\)$")



def find_figure_image_pairs(md_text: str, md_dir: Path) -> list[dict]:
    """Scan the markdown for 'Figure N: caption' immediately followed by an
    image reference, and resolve each image to a real file path."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", md_text) if b.strip()]

    pairs = []
    for i, block in enumerate(blocks):
        caption_match = CAPTION_RE.match(block)
        if not caption_match or i + 1 >= len(blocks):
            continue
        image_match = IMAGE_REF_RE.match(blocks[i + 1])
        if not image_match:
            continue

        label, number, caption_text = caption_match.groups()
        raw_path = image_match.group(1).replace("\\", "/")  # Windows -> POSIX
        pairs.append({
            "label": f"{label} {number}",
            "caption": caption_text.strip(),
            "image_path": (md_dir / raw_path).resolve(),
        })
    return pairs


def load_model():
    print(f"Loading model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
    )
    model.to(DEVICE)
    model.eval()
    return model, processor


def caption_batch(model, tokenizer, batch: list[dict]) -> list[str]:
    """One batched moondream2 call. Each image gets its OWN prompt built
    from its OWN paper caption -- real textual context, not a generic
    "describe this image" ask with no grounding."""
    images = [Image.open(p["image_path"]).convert("RGB") for p in batch]
    prompts = [
        f'This is {p["label"]} from a research paper. '
        f'Its original caption is: "{p["caption"]}". '
        f"Describe what the figure shows and its key insight, in 2-3 sentences."
        for p in batch
    ]
    return model.batch_answer(images=images, prompts=prompts, tokenizer=tokenizer)


def caption_all_figures(pairs: list[dict], model, tokenizer) -> list[dict]:
    """Caption every figure in mini-batches so CPU memory stays bounded
    regardless of how many figures the paper has."""
    results = []
    for start in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[start:start + BATCH_SIZE]
        descriptions = caption_batch(model, tokenizer, batch)
        for p, description in zip(batch, descriptions):
            results.append({**p, "description": description})
    return results


def build_figure_chunks(figures: list[dict], paper_title: str, next_chunk_id: int) -> list[dict]:
    """Turn captioned figures into chunk records matching the existing schema."""
    new_chunks = []
    for i, fig in enumerate(figures):
        body = f'{fig["label"]}: {fig["caption"]}\n\n{fig["description"]}'
        text = f"paper : {paper_title}\nsection : {fig['label']}\n\n{body}"
        new_chunks.append({
            "chunk_id": next_chunk_id + i,
            "paper_title": paper_title,
            "section": fig["label"],
            "text": text,
            "token_count": int(len(text.split()) * 1.3),
        })
    return new_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("md_path")
    parser.add_argument("chunks_json")
    parser.add_argument("--out", default="chunks_enriched.json")
    args = parser.parse_args()

    md_path = Path(args.md_path)
    md_text = md_path.read_text(encoding="utf-8")
    existing_chunks = json.loads(Path(args.chunks_json).read_text(encoding="utf-8"))
    paper_title = existing_chunks[0]["paper_title"] if existing_chunks else md_path.stem

    pairs = find_figure_image_pairs(md_text, md_path.parent)
    print(f"found {len(pairs)} figure/image pairs")

    if len(pairs) ==0 : 
        print("no figures found!")

    else : 
        model, tokenizer = load_model()
        figures = caption_all_figures(pairs, model, tokenizer)

        new_chunks = build_figure_chunks(figures, paper_title, next_chunk_id=len(existing_chunks))
        enriched = existing_chunks + new_chunks

        Path(args.out).write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{len(new_chunks)} figure chunks added -> {args.out} ({len(enriched)} total chunks)")