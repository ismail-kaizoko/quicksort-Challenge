""" 
Testing here three of the most common used pdf_parsing libraries : PyMuPDF, pdfplumber, and unstructured
Experiments tested on "Attention is all you need" paper.


Additional infos : 
- Unstructured package need these shell commands, to install if not provided : 
- TesseractOCR : you can run in powershell : " winget install UB-Mannheim.TesseractOCR && $env:Path += ";C:\Program Files\Tesseract-OCR"  "
Poppler forthe "hi_res" mode.


"""

import re
import sys
import time
from pathlib import Path




def run_pymupdf(path: str):
    t0 = time.perf_counter()
    import fitz
    doc = fitz.open(path)
    pages = [{"page_number": i + 1, "text": p.get_text("text")} for i, p in enumerate(doc)]
    doc.close()
    elapsed = time.perf_counter() - t0
    return pages, elapsed



def save_text(name: str, text: str, out_dir: str = "./output") -> None:
    """Write the full extracted text to <out_dir>/<name>.txt for manual review."""
    Path(out_dir).mkdir(exist_ok=True)
    out_path = Path(out_dir) / f"{name}.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"  saved: {out_path}")



def run_pdfplumber(path: str):
    t0 = time.perf_counter()
    import pdfplumber
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            pages.append({
                "page_number": i + 1,
                "text": page.extract_text() or "",
                "tables": page.extract_tables(),
            })
    elapsed = time.perf_counter() - t0
    return pages, elapsed


def run_unstructured(path: str, strategy):
    t0 = time.perf_counter()
    from unstructured.partition.pdf import partition_pdf
    # choice of parameters is justified following the documentation in this url : https://unstructured.readthedocs.io/en/main/core/partition.html#partition-pdf
    img_path = f"./output/images/unstructured/{strategy}"
    Path(img_path).mkdir(exist_ok=True, parents=True)

    if strategy == "hi_res":
        infer_table_structure = True
    else : infer_table_structure = False

    elements = partition_pdf(filename=path, 
                             strategy=strategy, 
                             extract_images_in_pdf=True,
                             extract_image_block_types=["Image", "Table"],
                             extract_image_block_output_dir=img_path,
                             infer_table_structure = infer_table_structure,
                             languages=["eng"],
                             max_partition = None
                             )
    elapsed = time.perf_counter() - t0
    pages = {}
    for el in elements:
        pn = el.metadata.page_number or 0
        pages.setdefault(pn, []).append({"category": el.category, "text": str(el)})
    return pages, elapsed


def summarize(name: str, pages, elapsed: float, is_unstructured=False):
    if is_unstructured:
        all_text = "\n".join(e["text"] for els in pages.values() for e in els)
    else:
        all_text = "\n".join(p["text"] for p in pages)
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(f"  time:        {elapsed:.2f}s")
    print(f"  total chars: {len(all_text)}")
    return all_text


if __name__ == "__main__":
    pdf_path = "data/Attention_Is_All_You_Need.pdf"


    # --- PyMuPDF ---
    pm_pages, pm_time = run_pymupdf(pdf_path)
    pm_text = summarize("PyMuPDF", pm_pages, pm_time)
    save_text("pymudf", pm_text)



    # --- pdfplumber ---
    pp_pages, pp_time = run_pdfplumber(pdf_path)
    pp_text = summarize("pdfplumber", pp_pages, pp_time)
    save_text("pdfplumber", pp_text)



    # --- unstructured ---
    strats = ["hi_res", "ocr_only", "fast"]
    for strat in strats :
        print(f"\n unsing the '{strat}' strategy")
        un_pages, un_time = run_unstructured(pdf_path, strategy=strat)
        un_text = summarize(f"unstructured_{strat}", un_pages, un_time, is_unstructured=True)
        save_text(f"unstructured_{strat}", un_text)
