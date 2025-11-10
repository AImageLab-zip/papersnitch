from pypdf import PdfReader

from pathlib import Path

PDF_DIR = Path(__file__).resolve().parent / "media" / "pdf"

pdf_name = "miccai_2025_0308_paper.pdf"
reader = PdfReader(PDF_DIR / pdf_name)
number_of_pages = len(reader.pages)
pages = reader.pages
text = ""
for page in reader.pages:
    text = text + page.extract_text()

with open(PDF_DIR / "miccai_2025_0308_paper.txt", "w") as text_file:
    text_file.write(text)
