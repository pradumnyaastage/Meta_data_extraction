# src/extract_text.py

import os
from docx import Document
from PIL import Image
import pytesseract

def extract_text_from_docx(path):
    doc = Document(path)
    return " ".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])

def extract_text_from_png(path):
    image = Image.open(path)
    text = pytesseract.image_to_string(image)
    return text.strip()

def extract_text_from_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        return extract_text_from_docx(path)
    elif ext == ".png":
        return extract_text_from_png(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
