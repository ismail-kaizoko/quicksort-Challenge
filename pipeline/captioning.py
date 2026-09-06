import argparse
import json
import re
from pathlib import Path
import torch

from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM

import yaml



PROMPT = """
You are analyzing a figure extracted from a scientific research paper.

Your goal is to create a technically useful description for a Retrieval
Augmented Generation (RAG) system.

For ALL figures:
- Describe what the figure represents.
- Extract important visible labels and terminology.
- Explain the main scientific purpose of the figure.
"""

_config = yaml.safe_load(Path(__file__).parent.joinpath("config.yaml").read_text())["captioning"]
MODEL_ID = _config["captioning"]["model_id"]
BATCH_SIZE = _config["captioning"]["batch_size"]


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # CUDA can use float16 for better speed/memory.
    # CPU should use float32.
    dtype = (
        torch.float16
        if device == "cuda"
        else torch.float32
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype )


    model.eval()

    print("Model loaded successfully.")

    return model, processor


def caption_batch(model, processor, batch: list[dict]) -> list[str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "content": batch,
                },
                {
                    "type": "text",
                    "text": PROMPT,
                },
            ],
        }
    ]

    # Processor handles image loading + tokenization
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(device)

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    # Remove the input tokens from generated output
    input_length = inputs["input_ids"].shape[-1]

    generated_ids = generated_ids[:, input_length:]

    result = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return result.strip()

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