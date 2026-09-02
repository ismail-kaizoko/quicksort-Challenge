from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def load_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    return model, processor


def analyze_image(image_path, prompt, model, processor):

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
        )

    # Remove the input tokens from the output
    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output


PROMPT = """
Analyze this scientific figure carefully.

Determine what type of figure it is: architecture diagram,
flowchart, chart, graph, table, or another scientific visualization.

Then provide:

1. A precise description of what the figure shows.
2. The important components, labels, and relationships.
3. If it is an architecture/flowchart:
   - explain the data flow step by step
   - explain the role of each major component
4. If it is a chart/graph:
   - identify the X and Y axes
   - identify compared methods/models
   - extract important trends and numerical results when clearly visible
   - state the main conclusion
5. If it is a table:
   - identify rows, columns, metrics, and important results
   - identify the best-performing methods when possible.

Do not invent information or numerical values that are not clearly visible.
Be technically precise. This description will be used in a RAG system.
"""


def caption_folder(folder_path):

    folder = Path(folder_path)

    output_dir = Path("./output/captions/qwen")
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model()

    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}

    for image_path in folder.iterdir():

        if image_path.suffix.lower() not in image_extensions:
            continue

        print(f"\nProcessing: {image_path.name}")

        try:
            result = analyze_image(
                image_path,
                PROMPT,
                model,
                processor,
            )

            print(result)

            output_path = output_dir / f"{image_path.stem}.txt"

            output_path.write_text(
                result,
                encoding="utf-8",
            )

        except Exception as e:
            print(f"ERROR processing {image_path.name}: {e}")


if __name__ == "__main__":
    caption_folder("./data/figs")