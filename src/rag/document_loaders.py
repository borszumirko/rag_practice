import os
import tempfile

from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
    PyPDFLoader,
)

from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()


def load_text_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"Hello this is a smaple text file.\nThis file is used to...")
        temp_file_path = temp_file.name

    try:
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        for doc in documents:
            print(doc.page_content)
            print(doc)
    finally:
        os.remove(temp_file_path)


def pdf_loader(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} from PDF")

    for i, doc in enumerate(documents):
        print(f"Doc {i+1} preview: {doc.page_content[:100]}")
        print(f"metadata: {doc.metadata}")


pdf_loader(
    pdf_path="/home/mirko-borszukovszki/Documents/python_test/rag/docs/langchain_demo.pdf"
)
