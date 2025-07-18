import os
import pdfplumber
from docx import Document
from typing import Dict, Any, List

class FileProcessor:
    @staticmethod
    def extract_text_from_file(file_path: str) -> str:
        try:
            max_file_size = 5 * 1024 * 1024
            if os.path.getsize(file_path) > max_file_size:
                raise ValueError("File size exceeds maximum limit of 5MB")

            file_ext = os.path.splitext(file_path)[1].lower()

            if file_ext == '.pdf':
                with pdfplumber.open(file_path) as pdf:
                    return "\n".join(
                        page.extract_text() for page in pdf.pages
                        if page.extract_text()
                    )
            elif file_ext in ['.docx', '.doc']:
                doc = Document(file_path)
                return "\n".join(paragraph.text for paragraph in doc.paragraphs)
            elif file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

        except Exception as e:
            raise Exception(f"Error extracting text from {file_path}: {str(e)}")