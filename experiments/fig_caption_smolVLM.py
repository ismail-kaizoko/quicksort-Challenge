from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

INPUT_FOLDER = Path("data/figs")
OUTPUT_FOLDER = Path("./output/captions/smolvlm")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# Reduce this if generation is too slow
MAX_NEW_TOKENS = 700


# ============================================================
# PROMPT
# ============================================================

PROMPT = """
You are analyzing a figure extracted from a scientific research paper.

Your goal is to create a technically useful description for a Retrieval
Augmented Generation (RAG) system.

First identify the figure type:
- neural network architecture
- flowchart
- graph/chart
- table
- mathematical visualization
- other scientific figure

Then analyze it carefully.

For ALL figures:
- Describe what the figure represents.
- Extract important visible labels and terminology.
- Explain the main scientific purpose of the figure.

If it is an ARCHITECTURE DIAGRAM or FLOWCHART:
- Explain the components.
- Explain the connections between components.
- Describe the data flow step by step.
- Explain the role of important modules.

If it is a GRAPH or CHART:
- Identify the X-axis and Y-axis.
- Identify the compared methods or models.
- Describe important trends.
- Extract numerical values only when clearly visible.
- State the main conclusion.

If it is a TABLE:
- Identify the columns and rows.
- Identify the evaluation metrics.
- Extract important results.
- Identify the best-performing methods when clearly visible.

IMPORTANT:
- Be technically precise.
- Do not hallucinate information.
- Do not invent numerical values.
- Use the terminology visible in the figure.
- Write a detailed but concise description suitable for semantic retrieval.
"""


# ============================================================
# MODEL
# ============================================================

def load_model():

    print("Loading SmolVLM2...")

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
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cpu":
        model = model.to(device)

    model.eval()

    print("Model loaded successfully.")

    return model, processor, device


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(
    image_path: Path,
    model,
    processor,
    device: str,
) -> str:

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "path": str(image_path),
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

    # Move tensors to the correct device
    if device == "cuda":
        inputs = inputs.to(model.device)
    else:
        inputs = inputs.to(device)

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    # Remove the input tokens from generated output
    input_length = inputs["input_ids"].shape[-1]

    generated_ids = generated_ids[:, input_length:]

    result = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return result.strip()


# ============================================================
# PROCESS FOLDER
# ============================================================

def caption_folder(folder_path: Path):

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Input folder does not exist: {folder_path}"
        )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = [
        path
        for path in folder_path.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"\nFound {len(images)} images.")

    if not images:
        return

    # Load model ONCE
    model, processor, device = load_model()

    for index, image_path in enumerate(images, start=1):

        print("\n" + "=" * 70)
        print(f"[{index}/{len(images)}] Processing: {image_path.name}")
        print("=" * 70)

        try:

            result = analyze_image(
                image_path=image_path,
                model=model,
                processor=processor,
                device=device,
            )

            print("\nRESULT:\n")
            print(result)

            # Save individual caption
            output_path = (
                OUTPUT_FOLDER
                / f"{image_path.stem}.txt"
            )

            output_path.write_text(
                result,
                encoding="utf-8",
            )

            print(f"\nSaved: {output_path}")

        except Exception as e:

            print(
                f"\nERROR processing "
                f"{image_path.name}:\n{e}"
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    caption_folder(INPUT_FOLDER)
