import os
from src.textSummarizer.logging import logger
from pathlib import Path
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import io

# Text extraction logic
def iter_text_per_page(pdf_file : Path, password='', page_numbers: list=None, maxpages=0, caching=True, codec='utf-8', laparams=None):
    """Extract text from PDF file per page."""
    if laparams is None:
        laparams = LAParams()

    with open(pdf_file, "rb") as fp:
        rsrcmgr = PDFResourceManager(caching=caching)
        idx = 1
        for page in PDFPage.get_pages(fp, page_numbers, maxpages=maxpages, password=password, caching=caching):
            with io.StringIO() as output_string:
                device = TextConverter(rsrcmgr, output_string, codec=codec, laparams=laparams)
                interpreter = PDFPageInterpreter(rsrcmgr, device)
                interpreter.process_page(page)
                yield idx, output_string.getvalue()
                idx += 1

# PDF categorization based on page count
def categorize_pdf(pdf_path: Path):
    """Categorizes a PDF as 'Short', 'Medium', or 'Long' based on page count."""
    try:
        with open(pdf_path, "rb") as fh:
            page_count = sum(1 for _ in PDFPage.get_pages(fh))
    except Exception as e:
        logger.error(f"Error reading page count for {pdf_path}: {e}")
        return 'Error', 0

    if page_count <= 2:
        return 'Short', page_count
    elif 3 <= page_count <= 12:
        return 'Medium', page_count
    else:
        return 'Long', page_count

# Process large PDFs in batches
def process_large_pdf(pdf_path: Path , page_count: int, batch_size=10):
    """Processes large PDFs in batches."""
    total_pages = page_count
    text_data = []

    for batch_start in range(0, total_pages, batch_size):
        batch_end = min(batch_start + batch_size, total_pages)
        page_numbers = list(range(batch_start, batch_end))
        for _, page_text in iter_text_per_page(pdf_path, page_numbers=page_numbers):
            text_data.append(page_text)
    
    return "\n".join(text_data)
