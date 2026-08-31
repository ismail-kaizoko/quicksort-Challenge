"""Caption extracted figures with Florence-2-base (CPU-friendly, ~230M params).

pip install transformers torch pillow einops timm --break-system-packages
"""
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration

MODEL_ID = "florence-community/Florence-2-base"



# "<MORE_DETAILED_CAPTION>" gives richer descriptions than the plain
# "<CAPTION>" task prompt — worth the extra tokens for figures like
# architecture diagrams where the plain mode tends to under-describe.
TASK_PROMPT = "<MORE_DETAILED_CAPTION>"


def load_model():
    """Load Florence-2 in float32 on CPU (no GPU/quantization needed at this size)."""

    model = Florence2ForConditionalGeneration.from_pretrained(MODEL_ID).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    return model, processor


def caption_image(model, processor, image_path: Path) -> str:
    """Run detailed captioning on a single image and return the parsed caption text."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=TASK_PROMPT, images=image, return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=200,
            num_beams=3,
        )
    raw_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    # post_process_generation strips Florence-2's special tokens and returns
    # a dict keyed by task prompt — this is model-specific, not a generic API.
    parsed = processor.post_process_generation(
        raw_text, task=TASK_PROMPT, image_size=(image.width, image.height)
    )
    return parsed[TASK_PROMPT]


def caption_folder(folder: str, out_json: str = "figure_captions.json") -> None:
    """Caption every image in `folder` and write {filename: caption} to `out_json`."""
    model, processor = load_model()
    image_paths = sorted(
        p for p in Path(folder).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    results = {}
    for p in image_paths:
        caption = caption_image(model, processor, p)
        results[p.name] = caption
        print(f"{p.name}: {caption}")
    Path(out_json).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    caption_folder("./output/images/unstructured")