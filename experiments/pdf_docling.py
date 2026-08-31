"""
Expermimenting with the Docling library, see documentation : https://docling-project.github.io/docling/ 

"""

import time
from pathlib import Path
import json

from docling.document_converter import DocumentConverter
from docling.datamodel.document import PictureItem

from docling_core.types.doc import ImageRefMode, PictureItem, TableItem



ouput_dir = Path("./output/docling/")
ouput_dir.mkdir(exist_ok=True, parents=True)
t0 = time.perf_counter()
converter = DocumentConverter()
document = converter.convert("data/Attention_Is_All_You_Need.pdf").document



# Export to different data types : mardown,json....
md_document = document.export_to_markdown()
md_path = ouput_dir / "docling.md"
md_path.write_text(md_document, encoding = "utf-8")

dict_document = document.export_to_dict()
json_path = ouput_dir / "docling.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(
        dict_document,
        f,
        indent=2,
        ensure_ascii=False,
    )    


elapsed = time.perf_counter() - t0
print(elapsed)
