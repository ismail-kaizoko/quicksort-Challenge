""" Parsing documents following the experimetns done in the experiment/ folder using the Docling package
Docling documentation link : https://docling-project.github.io/docling/  
Inspired code in page : https://docling-project.github.io/docling/_generated/examples/export_figures/?utm_source=chatgpt.com
"""

import logging
import os
import time
from pathlib import Path

from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.settings import DEFAULT_PAGE_RANGE
from docling.document_converter import DocumentConverter, PdfFormatOption



def build_converter(caption_figure : bool) -> DocumentConverter:
    """Same PdfPipelineOptions as the original script: keep picture/table
    images so they can be exported, and run formula enrichment."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_picture_images = caption_figure
    # could be usefull if we hve a strong model that understands tables. discarded for now for simplicity
    # pipeline_options.generate_table_images = True
    pipeline_options.do_formula_enrichment = True
 
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )



def parse_pdf(converter: DocumentConverter, input_doc_path: Path, output_dir: Path) -> Path:
    """Convert one PDF to markdown (+ referenced image files) under output_dir.
 
    Returns:
        Path to the generated .md file (images land alongside it in a
        "<stem>_artifacts" folder, which is what caption_figures.py's path
        resolution expects).
    """
    conv_res = converter.convert(input_doc_path)
 
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_res.input.file.stem
    md_path = output_dir / f"{doc_filename}.md"
    conv_res.document.save_as_markdown(md_path, image_mode=ImageRefMode.REFERENCED)
    return md_path
