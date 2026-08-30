import time
from docling.document_converter import DocumentConverter



t0 = time.perf_counter()
converter = DocumentConverter()
result = converter.convert("data/Attention_Is_All_You_Need.pdf")


result = result.document.export_to_markdown()
elapsed = time.perf_counter() - t0

print(result, elapsed)
