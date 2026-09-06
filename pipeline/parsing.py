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


# # Under CI we limit the conversion to a representative page range to keep the
# # example fast; locally the full document is processed.
# IS_CI = os.environ.get("CI", "").lower() in ("true", "1", "yes")
# CI_PAGE_RANGE = (3, 4)

IMAGE_RESOLUTION_SCALE = 2.0



def main(input_doc_path : Path, output_dir : Path):

    # Keep page/element images so they can be exported. The `images_scale` controls
    # the rendered image resolution (scale=1 ~ 72 DPI). The `generate_*` toggles
    # decide which elements are enriched with images.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = True
    pipeline_options.do_formula_enrichment = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    
    conv_res = doc_converter.convert(input_doc_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_res.input.file.stem

    # # Save images of figures and tables
    # table_counter = 0
    # picture_counter = 0

    # for element, _level in conv_res.document.iterate_items():

    #     # saving tables as png
    #     if isinstance(element, TableItem) and element is not None:
    #         image = element.get_image(conv_res.document)
    #         if image is not None:
    #             table_counter += 1
    #             filename = (
    #                 output_dir/ f"{doc_filename}-table-{table_counter}.png"
    #             )
    #             image.save(filename, "PNG")
    #         else:
    #             print("Warning: Table image is None")

    #     # saving images as png
    #     if isinstance(element, PictureItem) and element is not None:
    #         image = element.get_image(conv_res.document)
    #         if image is not None:
    #             picture_counter += 1
    #             filename = (
    #                 output_dir/ f"{doc_filename}-picture-{picture_counter}.png"
    #             )
    #             image.save(filename, "PNG")
    #         else:
    #             print("Warning: Picture image is None")


    # Save markdown with externally referenced pictures
    md_filename = output_dir / f"{doc_filename}.md"
    conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)




if __name__ == "__main__":
    main(input_doc_path=Path("./../data/ChunkNorris.pdf"), output_dir=Path("./test/"))



