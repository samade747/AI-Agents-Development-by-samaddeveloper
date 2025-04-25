## 3. `tools/resume_tool.py`

import io
from PyPDF2 import PdfReader
from fastapi import UploadFile

def parse_pdf(file: UploadFile) -> str:
    """Extract text from uploaded PDF resume."""
    contents = file.file.read()
    reader = PdfReader(io.BytesIO(contents))
